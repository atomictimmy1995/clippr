"""Regenerate the test video and its transcript.

Needs espeak-ng (speech) and scikit-image (for its public-domain NASA astronaut photo):

    uv run --with scikit-image python tests/fixtures/make_fixture.py

Writes talk.mp4 (960x540, ~70 s) and talk.transcript.json next to this file. The first shot has
one face moving left to right (to test speaker tracking); after a hard cut, the second shot is a
slide with no face (to test the blurred "fit" layout). Word times in the transcript are
approximate: each sentence's spoken time is shared across its words by length.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from skimage import data

HERE = Path(__file__).resolve().parent
W, H, FPS = 960, 540, 25
GAP = 0.45  # silence between sentences
CUT_AT_SENTENCE = 9  # the cut to the slide happens before this sentence

SENTENCES = [
    "Welcome back to the channel, today we are talking about how rockets actually reach orbit.",
    "Here is the surprising part.",
    "Getting to space is easy, but staying there is incredibly hard.",
    "To stay in orbit you need to move sideways at almost eight kilometres every second.",
    "That is about twenty three times faster than a passenger jet.",
    "So why do rockets go straight up first?",
    "They climb out of the thick lower atmosphere, where drag wastes the most fuel.",
    "Then they tip over and spend almost all of their energy going sideways.",
    "Honestly, that changed how I think about every launch I watch.",  # loud
    "Now let's look at the numbers on this slide.",
    "Most of a rocket is fuel, often more than ninety percent of its weight at launch.",
    "That is why engineers obsess over every single kilogram.",
    "Every kilogram you save on the rocket is a kilogram of payload you can carry instead.",
    "Thanks for watching, and remember to look up at the next launch.",
]
LOUD = {8}


def tts(text: str, path: Path) -> None:
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "160", "-w", str(path), text], check=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32), rate


def trim(samples: np.ndarray, thresh: float = 300) -> np.ndarray:
    idx = np.nonzero(np.abs(samples) > thresh)[0]
    return samples[idx[0] : idx[-1] + 1] if len(idx) else samples


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    pieces, segments, t = [], [], 0.5
    rate = 22050
    pieces.append(np.zeros(int(0.5 * rate), dtype=np.float32))
    cut_time = 0.0
    for i, text in enumerate(SENTENCES):
        if i == CUT_AT_SENTENCE:
            cut_time = t - GAP / 2
        wav = tmp / f"s{i}.wav"
        tts(text, wav)
        s, rate = read_wav(wav)
        s = trim(s)
        gain = 1.0 if i in LOUD else 0.35  # the loud line stands out in the loudness signal
        pieces.append(s * gain)
        dur = len(s) / rate
        words = text.split()
        weights = np.array([len(w) + 2 for w in words], dtype=float)
        bounds = np.concatenate([[0], np.cumsum(weights) / weights.sum()]) * dur
        segments.append({
            "start": round(t, 3), "end": round(t + dur, 3), "text": text,
            "words": [{"w": w, "start": round(t + bounds[k], 3), "end": round(t + bounds[k + 1] - 0.02, 3), "p": 0.95}
                      for k, w in enumerate(words)],
        })
        t += dur
        pieces.append(np.zeros(int(GAP * rate), dtype=np.float32))
        t += GAP
    audio = np.clip(np.concatenate(pieces), -32768, 32767).astype(np.int16)
    total = len(audio) / rate
    audio_path = tmp / "audio.wav"
    with wave.open(str(audio_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.tobytes())

    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:300, 100:340]  # head and shoulders
    face = cv2.resize(face, (200, 250))
    frames_path = tmp / "video.mp4"
    n_frames = int(total * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-i", str(audio_path), "-c:v", "libx264", "-preset", "slow", "-crf", "30",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-shortest", str(HERE / "talk.mp4")],
        stdin=subprocess.PIPE,
    )
    yy, xx = np.mgrid[0:H, 0:W]
    studio = np.dstack([40 + 30 * xx / W, 50 + 20 * yy / H, 70 + 40 * xx / W]).astype(np.uint8)
    slide = np.full((H, W, 3), (235, 235, 235), np.uint8)
    cv2.rectangle(slide, (60, 60), (W - 60, 150), (90, 60, 20), -1)
    cv2.putText(slide, "ROCKET MASS BUDGET", (90, 122), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    for k, (label, pct, colour) in enumerate([("Fuel", 0.9, (40, 120, 230)), ("Structure", 0.07, (90, 90, 90)),
                                              ("Payload", 0.03, (60, 170, 60))]):
        y = 220 + k * 90
        cv2.putText(slide, label, (90, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        cv2.rectangle(slide, (300, y), (300 + int(560 * pct), y + 55), colour, -1)
    for i in range(n_frames):
        ts = i / FPS
        if ts < cut_time:
            frame = studio.copy()
            x = int(80 + (W - 360) * (ts / cut_time)) + int(6 * np.sin(ts * 3))  # drifts left to right
            y = 150
            frame[y : y + 250, x : x + 200] = face
        else:
            frame = slide
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()

    transcript = {"language": "en", "duration": round(total, 3), "segments": segments}
    (HERE / "talk.transcript.json").write_text(json.dumps(transcript, indent=1))
    print(f"wrote talk.mp4 ({total:.1f}s, cut at {cut_time:.2f}s) and talk.transcript.json")


if __name__ == "__main__":
    main()
