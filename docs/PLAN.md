# Clippr: YouTube link → captioned 9:16 key-moment clips

## Context

The `clippr` repo is empty (only `README.md`), so this is a new build. The goal: paste a YouTube link, and the app:

1. downloads the video and audio,
2. transcribes the audio with word-level timestamps,
3. finds the key moments and cuts them into clips of about 1 minute,
4. reframes each clip to vertical 9:16 so it follows the speaker,
5. burns in captions.

**Decisions you made:** a web app (with a CLI on the same core), local Whisper for transcription, Claude plus audio/scene signals to pick moments, and 9:16 vertical output at 1080×1920.

**Free to use.** Nothing has to be paid for. Every tool is free and open source and runs on your own machine. Claude is optional:
- If `ANTHROPIC_API_KEY` is set, Claude picks the moments. This costs about $0.10 or less per hour of video and gives the best picks.
- If it isn't, a free local model picks them through Ollama.
- If Ollama isn't installed either, free rules based on the audio and scene signals pick them.

The app shows which method picked each job's clips.

**Things to know up front:**
- **YouTube's terms.** Downloading from YouTube can break their terms of service. The UI will ask the user to confirm they own the video or have the rights to clip it.
- **yt-dlp breaks often.** YouTube changes often break `yt-dlp`, so it needs regular updates. Some videos need a cookies file, so there will be a setting for one.
- **This build machine may be blocked from YouTube.** Tests here use local video files. You'll do the real-URL check on your own machine.

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language / deps | Python 3.12+, `uv`, `pyproject.toml` | All the media and ML libraries are Python |
| Web | FastAPI + Jinja2 templates + HTMX, HTML5 `<video>` previews | One language and no JS build step; enough for paste-a-link, progress and preview/download |
| Jobs | SQLite (SQLModel) job table + one separate `clippr worker` process that claims queued jobs | No Redis or broker needed. Slow GPU/CPU work stays off the web process. |
| Download | `yt-dlp` Python API (not shell): `bv*[height<=1080]+ba/b`, merged to mp4; metadata (title, duration, chapters) saved | Standard tool; the API call avoids shell injection |
| Audio | `ffmpeg` → 16 kHz mono WAV | Input format Whisper wants |
| Transcription | `faster-whisper`, `word_timestamps=True`, `vad_filter=True`; `large-v3-turbo` on CUDA, `small` int8 on CPU (configurable, auto-detect) | Local, free, gives word timings for captions and clean cut points |
| Signals | Per-second loudness (numpy RMS over WAV), shot cuts (`scenedetect` ContentDetector), speech rate (words/sec) | Cheap hints for Claude; shot cuts are reused by reframing |
| Moment picking | Three interchangeable pickers behind one interface. **Claude** (optional, paid): Anthropic Python SDK, `claude-opus-5-5`, `client.beta.messages.parse(output_format=MomentList)`. **Local** (free): Ollama at `localhost:11434` with a JSON-schema `format`, model set by `OLLAMA_MODEL`. **Rules** (free, no AI): ranked by the signals. | Free by default; better picks if you add a key |
| Face tracking | OpenCV `cv2.FaceDetectorYN` (YuNet ONNX, ~300 KB, downloaded on first run) | No MediaPipe wheel problems on new Python versions |
| Render | OpenCV frame loop → crop/resize → raw frames piped to `ffmpeg` (libx264 + AAC) with `-vf ass=` captions burned in the same pass | One encode per clip; exact control of the crop for each frame |
| Captions | Generated `.ass` (word-highlight style) + sidecar `.srt`; bundled OFL font in `assets/fonts/` | Gives the "active word" look used on Shorts and TikTok |

## Project layout

```
pyproject.toml  .env.example  Dockerfile  docker-compose.yml (web + worker, shared ./data volume; optional `ollama` service)
assets/fonts/            # e.g. Montserrat ExtraBold (OFL)
src/clippr/
  config.py              # pydantic-settings: DATA_DIR, WHISPER_MODEL/DEVICE, SELECTOR (auto|claude|local|rules), ANTHROPIC_MODEL, OLLAMA_URL/MODEL, CLIP_MIN/MAX_S, MAX_CLIPS, MAX_VIDEO_MIN, YTDLP_COOKIES
  db.py, models.py       # SQLModel Job(id, url, status, stage, progress, error, options), Clip(job_id, idx, start, end, title, hook, score, paths)
  pipeline/
    download.py          # validate URL host (youtube.com / youtu.be only), enforce max duration, yt-dlp, extract WAV
    transcribe.py        # faster-whisper → Transcript{segments[], words[{w,start,end}]} → transcript.json
    signals.py           # loudness curve, scene cuts, speech rate → signals.json
    moments/
      __init__.py        # select_moments(): picks the selector, then the shared post-processing → moments.json
      prompt.py          # transcript formatting + instructions, shared by both AI pickers
      claude.py          # paid, optional
      local.py           # Ollama, free; splits long transcripts into windows
      rules.py           # free, no AI
      postprocess.py     # snap boundaries, clamp length, remove overlaps, rank
    reframe.py           # face detection, primary-subject tracking, smoothing → crop_path.npy per clip
    captions.py          # word chunks → .ass (highlighted active word) + .srt
    render.py            # cut + crop + scale 1080x1920 + burn captions + mux audio; thumbnail jpg
    runner.py            # runs the stages in order, writes progress, skips a stage whose output already exists (resume)
  worker.py              # loop: claim job (UPDATE … WHERE status='queued' … RETURNING), run, record failure
  cli.py                 # `clippr run <url|file>`, `clippr worker`, `clippr serve`
  web/app.py, routes.py, templates/, static/
tests/
```

Each job writes its files to `data/jobs/<job_id>/`: `source.mp4`, `audio.wav`, `transcript.json`, `signals.json`, `moments.json`, then `clips/clip_NN.{mp4,srt,ass,jpg}`.

## Pipeline detail

**1. Download.**
- Accept only YouTube hosts.
- Reject private or live videos, and anything over `MAX_VIDEO_MIN` (default 180).
- Map yt-dlp progress hooks to job progress.

**2. Transcribe.**
- Store word timings as the single source of truth for timing.
- Report progress from segment end times divided by the video duration.

**3. Signals.**
- Produce a per-second loudness z-score, the list of shot-cut timestamps, and words/sec per transcript segment.
- Combine them into short tags on each segment, such as `[loud]`, `[cut]` and `[fast]`.

**4. Moment selection (`moments/`).**

**Choosing the picker.** With `SELECTOR=auto` (the default), the app uses Claude if `ANTHROPIC_API_KEY` is set. Otherwise it uses the local model if the Ollama server responds and has `OLLAMA_MODEL` pulled. Otherwise it uses the rules. The chosen picker is stored on the job and shown in the UI.

**Shared prompt (Claude and local).**
- The transcript goes in as compact lines: `[12:03.4–12:09.8] text  {loud,cut}`. Video title and chapters go too.
- The prompt asks for moments that make sense on their own: a strong opening line, a complete thought, a payoff.
- It asks for 1.5× the clips needed, so there is room to drop bad ones.
- **Schema (Pydantic `MomentList`).** A list of `{start_s, end_s, title, hook, rationale, score 0–100}`.

**Claude picker (`claude.py`, paid, optional).**
- A 1-hour video is about 15k tokens, well within the 1M context.
- Run `count_tokens` as a guard. If the input is too big, fail clearly rather than truncate it.
- **Request settings.**
  - `model="claude-opus-5-5"`, `output_config={"effort": "high"}` (this model's default is `medium`), `max_tokens=16000`, non-streaming.
  - `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`. This retries automatically on another model if a request is refused.
  - Check `stop_reason` for `refusal` or `max_tokens` before reading `parsed_output`.
- **Prompt caching.** Put the stable system prompt and the transcript block first, with `cache_control: {"type": "ephemeral"}`. The per-request part goes after it: number of clips, length, and an optional user hint such as "focus on the product demo". "Regenerate" and "find more clips" then reuse the cached transcript.
- **Cost.** About $0.10 or less per hour of video at $4 / $20 per million input/output tokens.
- `ANTHROPIC_API_KEY` comes from the environment.

**Local picker (`local.py`, free).**
- Calls Ollama's `/api/chat` over HTTP (`httpx`) with `format` set to the `MomentList` JSON schema, `stream: false` and `options.num_ctx` set from config. The default model is a mid-size open-weight model such as an 8B Qwen or Llama; set it with `OLLAMA_MODEL`.
- Local models have smaller context windows and are weaker over long input, so the transcript is split into roughly 10-minute windows with 1 minute of overlap. The picker gets candidates for each window, then post-processing merges and ranks them.
- The output is validated with Pydantic. If it's invalid, the picker retries once and then falls back to the rules picker.
- Needs about 8 GB of RAM (CPU) or VRAM (GPU). It's slower than Claude, and the picks are noticeably weaker.

**Rules picker (`rules.py`, free, no AI).**
- Slides a 45–60 s window across the transcript in sentence-aligned steps.
- Scores each window from the loudness peaks, speech rate, laughter/applause gaps flagged by Whisper and shot-cut density. Windows that start mid-sentence or after a long silence get a penalty.
- Titles come from the first sentence. The hook is the highest-energy line.

**Post-processing (`postprocess.py`, every picker).** This is plain, testable code:
- snap the start to the nearest sentence/word start and the end to the nearest sentence end;
- force the length into `[CLIP_MIN_S=30, CLIP_MAX_S=60]`;
- drop clips that overlap by more than 20%;
- sort by score and keep the top `MAX_CLIPS` (default 5).

**5. Reframe (`reframe.py`).**
- **Detection.** Sample frames at 5 fps inside each clip and run YuNet face detection.
- **Choosing who to follow.** Link detections into tracks by IoU. Within each shot (cut times come from step 3), follow the main track, scored by size × time on screen × centredness.
- **Crop window.** Width is `src_h * 9/16`, at full height.
  - Centre x follows the subject, with a dead zone (no movement for small shifts) and an exponential or One-Euro smoothing filter.
  - At shot cuts it jumps instantly instead of smoothing across.
  - Values between samples are interpolated for every frame.
- **Fallback.** For shots with no face, or very wide shots (two speakers far apart, screen recordings), use a "fit" layout: the full frame scaled to the 1080 width, centred over a blurred, zoomed copy of itself.

**6. Captions (`captions.py`).**
- Shift word times to start from the clip start.
- Group words into chunks of 2–4 words (about 18 characters or less, never across a pause longer than 0.6 s).
- Write ASS events with `\k`/colour override tags so the word being spoken is highlighted.
- Style: bold, about 80 px, white text with a black outline. Position it about 68% down the frame, clear of the platform UI safe zones.
- Also write an `.srt` file.

**7. Render (`render.py`).**
- Read the source segment with OpenCV, apply the crop path for each frame and resize to 1080×1920.
- Pipe raw frames to ffmpeg. The command takes the audio from `source.mp4` for `[start, end]`, uses `-vf ass=clip.ass:fontsdir=assets/fonts` and `-c:v libx264 -crf 20 -preset medium -c:a aac -movflags +faststart`.
- Save a thumbnail from the frame with the highest score.

## Web app

**Pages**
- **Home:** a URL field, number of clips, an optional "focus" hint, and a checkbox to confirm rights.
- **Job page:** a stage/progress bar that HTMX polls every 2 s.
- **Clip grid:** `<video>` preview, title, hook, score, and Download MP4 / SRT buttons.

**API** (the HTML routes call the same functions)
- `POST /api/jobs {url, max_clips, focus}` → `{job_id}`
- `GET /api/jobs/{id}` → status, stage, progress, error, clips
- `GET /api/jobs/{id}/clips/{n}.mp4|.srt` (paths checked against the job directory)
- `POST /api/jobs/{id}/regenerate` re-runs selection using the cached transcript, then renders
- `DELETE /api/jobs/{id}` removes the DB rows and the job directory

**Operational behaviour**
- Logs are structured and tagged with job_id and stage.
- A failed stage stores its error on the job and shows it in the UI.
- A retry resumes from the last finished stage.
- Source media older than N days is cleaned up.

## Build order (milestones)

1. **Core CLI:** config, download, transcribe. `clippr run <url|file>` writes `transcript.json`.
2. **Selection:** signals, the rules picker and post-processing first (free, works with no setup), then the local picker, then the Claude picker. The CLI prints the chosen moments and which picker chose them.
3. **Render:** plain centre-crop cut first, then captions, then face-tracked reframing and the blur-fit fallback.
4. **Web + worker:** SQLite jobs, the worker loop, FastAPI pages and API, progress polling.
5. **Polish:** Dockerfile and compose (ffmpeg included, optional CUDA image), regenerate, cleanup, README.

## Verification

**Unit tests (pytest).** These cover the pure logic:
- boundary snapping, overlap removal and length clamping, run on a real-shaped `moments.json` + `transcript.json` fixture;
- caption chunking and ASS timing maths;
- smoothing and dead-zone behaviour, and hard cuts at shot boundaries;
- URL host validation;
- picker choice under `SELECTOR=auto` (key set / Ollama up / neither);
- the rules picker's scores on the fixture signals;
- the local picker's window splitting and merging;
- the job-claim query (only one worker gets each job).

**Integration test.**
- Run the pipeline on a short local test video with speech and faces (CC-licensed or self-recorded, kept in `tests/fixtures/`). It skips the download stage.
- Use `ffprobe` to check that each clip:
  - is 1080×1920 and h264/aac,
  - lasts no more than 60 s,
  - has an audio stream.
- Also check that the `.srt` timings fall inside the clip.
- The test always runs with the free rules picker. The local picker test runs only when Ollama is reachable, and the Claude test only when `ANTHROPIC_API_KEY` is set; otherwise those two are skipped.

**Manual end-to-end.**
- Start `clippr serve` and `clippr worker`, open the UI and paste a YouTube URL on your machine.
- Watch the stages move forward, then play the clips. Check that:
  - the speaker stays in frame;
  - the captions are in sync and the highlighted word is right;
  - the clips start and end on clean sentence boundaries.
- A Playwright smoke test submits a job against the local-file path and screenshots the finished clip grid.

## Changes made during the build

- **Shot cuts** come from ffmpeg's built-in scene-change score (`select='gt(scene,0.3)'`), not PySceneDetect. It's one dependency fewer and gives the same result.
- **The font** is Inter ExtraBold (OFL), bundled inside the package at `src/clippr/assets/fonts/`, so it ships with installs.
- **Face detection fallback.** If the YuNet model can't be downloaded, Clippr falls back to OpenCV's bundled `haarcascade_frontalface_alt2`. On the test video, alt2 found the face in every sample, where the default cascade found none.
- **Retry vs regenerate.** A failed job has a **Try again** button, which resumes from the last finished stage. **Regenerate** picks new moments from the cached transcript.
- **Speed.** Tracking + rendering a 52-second clip from a 1080p source took 42 s on a 4-core CPU.
