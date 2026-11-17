"""Stage 7: cut, reframe to 9:16, burn in captions and encode one clip."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from clippr.media import MediaError
from clippr.pipeline.reframe import CropPlan, iter_frames

log = logging.getLogger(__name__)


def _blurred_background(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(out_w / w, out_h / h)
    bw, bh = int(np.ceil(w * scale)), int(np.ceil(h * scale))
    # Blur at low resolution: much faster and looks the same.
    small = cv2.resize(frame, (max(1, bw // 8), max(1, bh // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    big = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (bw - out_w) // 2, (bh - out_h) // 2
    return (big[y0 : y0 + out_h, x0 : x0 + out_w] * 0.55).astype(np.uint8)


def compose(frame: np.ndarray, mode: str, cx: float, plan: CropPlan, out_w: int, out_h: int) -> np.ndarray:
    if mode == "crop":
        x0 = int(round(cx - plan.crop_w / 2))
        x0 = max(0, min(plan.src_w - plan.crop_w, x0))
        crop = frame[:, x0 : x0 + plan.crop_w]
        interp = cv2.INTER_AREA if crop.shape[0] > out_h else cv2.INTER_LINEAR
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)
    # "fit": whole frame scaled to the output width, centred over a blurred copy of itself
    canvas = _blurred_background(frame, out_w, out_h)
    h, w = frame.shape[:2]
    fw, fh = out_w, int(round(h * out_w / w))
    if fh > out_h:
        fw, fh = int(round(w * out_h / h)), out_h
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    x0, y0 = (out_w - fw) // 2, (out_h - fh) // 2
    canvas[y0 : y0 + fh, x0 : x0 + fw] = fg
    return canvas


def _filter_escape(value: str) -> str:
    """Escape a value for use inside an ffmpeg filtergraph option."""
    value = value.replace("\\", "/")
    for ch in ("'", ":", ",", ";", "[", "]"):
        value = value.replace(ch, "\\" + ch)
    return value


def render_clip(source: Path, start: float, end: float, plan: CropPlan, ass_path: Path | None,
                out_path: Path, thumb_path: Path, *, out_w: int, out_h: int, fps: float,
                fonts_dir: Path, preset: str = "fast", crf: int = 20,
                progress: Callable[[float], None] = lambda f: None) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise MediaError(f"OpenCV could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fps
    dur = end - start
    frames = iter_frames(cap, start, end, fps)
    first = next(frames, None)
    if first is None:
        cap.release()
        raise MediaError("No video frames in the clip range")
    t_first = first[0]

    clip_dir = out_path.parent
    tmp_out = out_path.with_suffix(".tmp.mp4")
    vf = "format=yuv420p"
    if ass_path is not None:
        fonts_rel = os.path.relpath(fonts_dir, clip_dir)
        vf = f"ass={_filter_escape(ass_path.name)}:fontsdir={_filter_escape(fonts_rel)}," + vf
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}", "-i", "pipe:0",
        "-ss", f"{start + t_first:.3f}", "-t", f"{dur - t_first:.3f}", "-i", str(source.resolve()),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart", tmp_out.name,
    ]
    thumb_at = dur * 0.3
    thumb_written = False
    with tempfile.TemporaryFile() as errlog:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errlog, cwd=clip_dir)
        try:
            for t, frame in _chain(first, frames):
                mode, cx = plan.at(t)
                out = compose(frame, mode, cx, plan, out_w, out_h)
                if not thumb_written and t >= thumb_at:
                    cv2.imwrite(str(thumb_path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    thumb_written = True
                proc.stdin.write(out.tobytes())
                progress(min(1.0, t / dur))
        except BrokenPipeError:
            pass  # ffmpeg exited early; its error is reported below
        finally:
            cap.release()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            code = proc.wait()
        if code != 0:
            errlog.seek(0)
            tmp_out.unlink(missing_ok=True)
            raise MediaError(f"ffmpeg failed to encode the clip: {errlog.read().decode(errors='replace')[-1500:]}")
    tmp_out.replace(out_path)
    if not thumb_written:
        log.warning("clip too short for a thumbnail at %.1fs", thumb_at)


def _chain(first, rest):
    yield first
    yield from rest
