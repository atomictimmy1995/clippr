"""Stage 1: get the source video (a YouTube link or a local file) and extract its audio."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clippr.config import Settings
from clippr.media import extract_audio, probe, run
from clippr.schemas import Chapter, VideoInfo

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}


class DownloadError(RuntimeError):
    pass


def validate_youtube_url(url: str) -> str:
    """Accept only links to a single YouTube video. Returns the cleaned URL."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DownloadError("Link must start with http:// or https://")
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise DownloadError("Only YouTube links are supported")
    path = parsed.path.rstrip("/")
    if host == "youtu.be":
        if not path.strip("/"):
            raise DownloadError("That youtu.be link has no video ID")
    elif path == "/watch":
        if not parse_qs(parsed.query).get("v"):
            raise DownloadError("That YouTube link has no video ID (missing ?v=)")
    elif not any(path.startswith(p) and len(path) > len(p) for p in ("/shorts/", "/live/", "/embed/")):
        raise DownloadError("That doesn't look like a link to a single YouTube video")
    return url


def is_local_source(source: str) -> bool:
    return "://" not in source and Path(source).expanduser().is_file()


def fetch(source: str, job_dir: Path, settings: Settings, progress: Progress) -> VideoInfo:
    """Download or copy the video to ``job_dir/source.mp4`` and write ``audio.wav``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "source.mp4"
    meta: dict = {}
    if is_local_source(source):
        _import_local(Path(source).expanduser(), target)
        meta["title"] = Path(source).stem
    else:
        meta = _download_youtube(validate_youtube_url(source), job_dir, target, settings, progress)

    p = probe(target)
    if not p.has_audio:
        raise DownloadError("The video has no audio track, so there is nothing to transcribe")
    if p.duration > settings.max_video_min * 60:
        raise DownloadError(f"Video is longer than the {settings.max_video_min}-minute limit")

    progress(0.95, "Extracting audio")
    extract_audio(target, job_dir / "audio.wav")
    chapters = [
        Chapter(title=c.get("title") or "", start=float(c["start_time"]), end=float(c["end_time"]))
        for c in meta.get("chapters") or []
        if c.get("start_time") is not None and c.get("end_time") is not None
    ]
    return VideoInfo(
        source=source,
        title=meta.get("title") or "",
        uploader=meta.get("uploader") or "",
        duration=p.duration,
        width=p.width,
        height=p.height,
        fps=p.fps,
        chapters=chapters,
    )


def _import_local(src: Path, target: Path) -> None:
    p = probe(src)
    if src.suffix.lower() == ".mp4" and p.video_codec == "h264":
        shutil.copyfile(src, target)
        return
    tmp = target.with_suffix(".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(tmp)])
    tmp.replace(target)


def _download_youtube(url: str, job_dir: Path, target: Path, settings: Settings, progress: Progress) -> dict:
    import yt_dlp  # imported lazily: it is slow to import and only needed here

    base_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
    }
    if settings.ytdlp_cookies:
        base_opts["cookiefile"] = str(settings.ytdlp_cookies)

    progress(0.0, "Checking video")
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"YouTube refused the request: {_clean(e)}") from e
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise DownloadError("Live streams and premieres can't be clipped until they have finished")
    duration = info.get("duration") or 0
    if duration > settings.max_video_min * 60:
        raise DownloadError(f"Video is longer than the {settings.max_video_min}-minute limit")

    h = settings.max_height

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress(min(0.9, 0.9 * done / total), "Downloading video")

    opts = {
        **base_opts,
        # Prefer H.264 so OpenCV can decode it everywhere; fall back to anything up to max height.
        "format": (
            f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={h}]+ba/b[height<={h}]/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(job_dir / "download.%(ext)s"),
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Download failed: {_clean(e)}") from e

    files = sorted(job_dir.glob("download.*"))
    if not files:
        raise DownloadError("Download finished but no file was written")
    downloaded = files[0]
    if downloaded.suffix == ".mp4" and probe(downloaded).video_codec == "h264":
        downloaded.replace(target)
    else:
        progress(0.9, "Converting video")
        _import_local(downloaded, target)
        downloaded.unlink(missing_ok=True)
    return info


def _clean(e: Exception) -> str:
    msg = str(e).replace("ERROR: ", "")
    msg = msg.split("; please report this issue")[0]  # yt-dlp's bug-report boilerplate
    return msg.strip()[:300]
