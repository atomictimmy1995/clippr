# clippr

Turn a YouTube link into short, captioned, vertical (9:16) clips of its best moments.

Paste a link and Clippr will:

1. **Download** the video and audio with yt-dlp.
2. **Transcribe** the audio on your machine with Whisper (faster-whisper), keeping a timestamp for every word.
3. **Pick the key moments.** It finds 30–60 second stretches that make sense on their own, then snaps their start and end to sentence boundaries.
4. **Reframe** each clip to 1080×1920. The crop follows the speaker's face; shots with no face, or with two people too far apart to fit, use the whole frame over a blurred copy of itself.
5. **Caption** each clip. Captions are burned in and the word being spoken is highlighted; you also get an `.srt` file.

**It's free to use.** Everything runs locally with free, open-source tools. Using Claude to pick moments is optional (see [Moment pickers](#moment-pickers)).

> Only clip videos you own or have permission to use. Downloading from YouTube may break YouTube's terms of service.

## Quick start

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on your PATH.

```bash
uv sync
uv run clippr serve        # web app on http://127.0.0.1:8000
uv run clippr worker       # in a second terminal: processes the jobs
```

Open http://127.0.0.1:8000 and paste a link. The page shows progress, then a grid of clips you can preview and download.

### Command line

```bash
uv run clippr run "https://www.youtube.com/watch?v=..." -n 5 --focus "the funniest moments"
uv run clippr run ./my-talk.mp4          # local files work too
```

Clips are written to `data/jobs/<id>/clips/` as `clip_01.mp4`, `clip_01.srt` and `clip_01.jpg` (thumbnail).

### Docker (CPU)

```bash
docker compose up                        # web + worker on http://localhost:8000
docker compose --profile local-ai up     # also starts Ollama for the free local AI picker
docker compose exec ollama ollama pull qwen3:8b
```

## Moment pickers

Clippr has three interchangeable pickers. With the default `CLIPPR_SELECTOR=auto` it uses the first one available. The page and the CLI show which one picked each job's clips.

| Picker | Cost | Needs | Quality |
|---|---|---|---|
| **Claude** | ~$0.10 or less per hour of video | `ANTHROPIC_API_KEY` | Best: understands which moments make sense on their own |
| **Local AI** | Free | [Ollama](https://ollama.com) running with `CLIPPR_OLLAMA_MODEL` pulled (default `qwen3:8b`, ~8 GB RAM/VRAM) | Good, slower |
| **Rules** | Free | Nothing | Basic: ranks sentence-aligned windows by loudness, speaking pace, pauses, laughter and camera cuts |

If a picker fails in `auto` mode (for example, a bad API key), Clippr falls back to the next one and shows a note on the page.

To use the free local AI picker:

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen3:8b
```

To use Claude, put `ANTHROPIC_API_KEY=...` in `.env`. The transcript is sent as a cached prompt prefix, so "Regenerate clips" is cheap.

## Configuration

Settings come from environment variables or a `.env` file. See [`.env.example`](.env.example) for all of them. The ones you're most likely to change:

| Variable | Default | |
|---|---|---|
| `CLIPPR_SELECTOR` | `auto` | `auto`, `claude`, `local` or `rules` |
| `CLIPPR_WHISPER_MODEL` | `auto` | `large-v3-turbo` on an NVIDIA GPU, `small` on CPU. Any faster-whisper model name works |
| `CLIPPR_LANGUAGE` | detect | e.g. `en` |
| `CLIPPR_MAX_CLIPS` | `5` | |
| `CLIPPR_CLIP_MIN_S` / `CLIPPR_CLIP_MAX_S` | `30` / `60` | clip length in seconds |
| `CLIPPR_MAX_VIDEO_MIN` | `180` | longest video accepted |
| `CLIPPR_YTDLP_COOKIES` | none | cookies file, if YouTube asks you to sign in |
| `CLIPPR_RETENTION_DAYS` | `7` | source videos of finished jobs are deleted after this; clips stay |

### Things to know

- **First run downloads models.**
  - The Whisper model (about 500 MB for `small`, 1.6 GB for `large-v3-turbo`) comes from Hugging Face.
  - The YuNet face detector (about 300 KB) comes from GitHub.
  - If YuNet can't be downloaded, Clippr uses OpenCV's built-in face detector instead, which is less accurate.
- **Speed.** On a CPU, transcription with the `small` model usually runs several times faster than real time, and tracking + rendering a 60-second clip from a 1080p video takes about a minute on a 4-core CPU. An NVIDIA GPU makes transcription much faster.
- **Keep yt-dlp up to date.** YouTube changes often break it: run `uv lock --upgrade-package yt-dlp && uv sync`.
- **Keep it on your own machine.** The web app has no login. Run it locally (the default binds to `127.0.0.1`), not on a public server.
- **One worker at a time.** Run a single `clippr worker`. Jobs interrupted by a crash or restart are picked up again when the worker restarts.

## How it works

```
src/clippr/
  pipeline/
    download.py      # validate link, yt-dlp, extract 16 kHz WAV
    transcribe.py    # faster-whisper with word timestamps
    signals.py       # loudness per second, camera cuts (ffmpeg scene score), tags per line
    moments/         # picker choice + claude.py / local.py / rules.py + shared post-processing
    reframe.py       # face detection → tracks → per-shot smoothed crop path (or "fit")
    captions.py      # word chunks → .ass with active-word highlight, and .srt
    render.py        # OpenCV frames → crop/fit → ffmpeg (libx264 + AAC, captions burned in)
    runner.py        # runs stages in order; finished stages are skipped, so jobs resume
  db.py, worker.py   # SQLite job table; the worker claims queued jobs one at a time
  web/               # FastAPI + Jinja2 + htmx pages and a JSON API
```

Each job keeps its intermediate files in `data/jobs/<id>/`: `source.mp4`, `audio.wav`, `transcript.json`, `signals.json` and `moments.json`. The rendered clips go in `clips/`.

### JSON API

| Method | Path | |
|---|---|---|
| `POST` | `/api/jobs` | `{"url": "...", "max_clips": 5, "focus": "...", "confirm_rights": true}` → `{"job_id"}` |
| `GET` | `/api/jobs/{id}` | status, progress, picker and clips (with video, SRT and thumbnail URLs) |
| `POST` | `/api/jobs/{id}/regenerate` | pick new moments (optionally `{"focus": "..."}`); reuses the transcript |
| `DELETE` | `/api/jobs/{id}` | delete the job and its files |

## Tests

```bash
uv run pytest                                  # unit + integration tests (~1 min)
CLIPPR_TEST_WHISPER=1 uv run pytest -k whisper # also run real Whisper transcription
```

The integration tests render real clips from `tests/fixtures/talk.mp4`. This is a 58-second synthetic talk: a face that drifts across the frame, then a hard cut to a slide. `tests/fixtures/make_fixture.py` regenerates it.

Some tests run only when their requirements are present:
- the Claude test needs `ANTHROPIC_API_KEY`;
- the local AI test needs Ollama to be running.

## Licences

- **Bundled font:** Inter (SIL Open Font License, `src/clippr/assets/fonts/LICENSE-Inter.txt`).
- **Bundled htmx:** 0BSD (`src/clippr/web/static/HTMX-LICENSE.txt`).
- **Test fixture photo:** NASA's public-domain astronaut portrait, taken from scikit-image's sample data.
