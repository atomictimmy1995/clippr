"""Stage 5: work out where to point a 9:16 crop window over time so it follows the speaker.

For each clip we sample frames at 5 fps, detect faces, link them into tracks, and in every shot
follow the main track with a dead zone and zero-phase smoothing. Shots with no usable face, or
with two people too far apart to fit in one crop, use a "fit" layout instead: the whole frame
over a blurred copy of itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
from pydantic import BaseModel

log = logging.getLogger(__name__)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SAMPLE_FPS = 5.0
DETECT_WIDTH = 640
DEAD_ZONE = 0.05  # fraction of source width the subject can drift before the crop follows
EMA_ALPHA = 0.25  # per-sample smoothing factor (zero-phase, so no lag)
MIN_PRESENCE = 0.25  # a face must be in this share of a shot's samples to be followed


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def area(self) -> float:
        return self.w * self.h


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    """YuNet (accurate, downloaded once) with OpenCV's bundled Haar cascade as an offline fallback."""

    def __init__(self, models_dir: Path):
        self.kind = "haar"
        self._yunet = None
        self._haar = None
        path = models_dir / YUNET_FILE
        if not path.exists():
            self._download(path)
        if path.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 50)
                self.kind = "yunet"
            except cv2.error as e:
                log.warning("could not load YuNet model, using Haar cascade: %s", e)
        if self._yunet is None:
            # "alt2" finds clear frontal faces far more reliably than the "default" cascade.
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    @staticmethod
    def _download(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            with httpx.stream("GET", YUNET_URL, follow_redirects=True, timeout=30) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            if tmp.stat().st_size < 100_000:  # an error page, not the model
                raise ValueError("downloaded file is too small to be the model")
            tmp.replace(path)
            log.info("downloaded YuNet face model to %s", path)
        except (httpx.HTTPError, OSError, ValueError) as e:
            log.info("YuNet face model unavailable (%s); using the bundled Haar cascade", e)
            tmp.unlink(missing_ok=True)

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale != 1.0 else frame
        boxes: list[Box] = []
        if self._yunet is not None:
            self._yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._yunet.detect(small)
            for row in faces if faces is not None else []:
                boxes.append(Box(row[0] / scale, row[1] / scale, row[2] / scale, row[3] / scale, float(row[14])))
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            min_side = max(20, small.shape[0] // 18)
            rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
            for x, y, bw, bh in rects if len(rects) else []:
                boxes.append(Box(x / scale, y / scale, bw / scale, bh / scale))
        return boxes


@dataclass
class Track:
    samples: dict[int, Box] = field(default_factory=dict)

    def last(self) -> tuple[int, Box]:
        k = max(self.samples)
        return k, self.samples[k]


def build_tracks(detections: list[list[Box]], max_gap: int = 3) -> list[Track]:
    """Greedy frame-to-frame linking of face boxes by overlap and horizontal distance."""
    tracks: list[Track] = []
    for k, boxes in enumerate(detections):
        used: set[int] = set()
        for b in sorted(boxes, key=lambda x: x.area, reverse=True):
            best, best_cost = None, None
            for ti, t in enumerate(tracks):
                lk, lb = t.last()
                if ti in used or lk == k or k - lk > max_gap:
                    continue
                overlap = iou(b, lb)
                dist = abs(b.cx - lb.cx) / max(lb.w, 1.0)
                if overlap > 0.2 or dist < 0.6:
                    cost = dist - overlap
                    if best_cost is None or cost < best_cost:
                        best, best_cost = ti, cost
            if best is None:
                tracks.append(Track({k: b}))
                used.add(len(tracks) - 1)
            else:
                tracks[best].samples[k] = b
                used.add(best)
    return tracks


def smooth_path(values: np.ndarray, dead_zone: float, lo: float, hi: float, alpha: float = EMA_ALPHA) -> np.ndarray:
    if len(values) == 0:
        return values
    v = values.astype(np.float64)
    if len(v) >= 5:  # median filter kills single-sample detector jitter
        padded = np.pad(v, 2, mode="edge")
        v = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    held = np.empty_like(v)
    cur = v[0]
    for i, x in enumerate(v):  # dead zone: only move when the subject leaves the comfort zone
        if abs(x - cur) > dead_zone:
            cur = x - np.sign(x - cur) * dead_zone
        held[i] = cur

    def ema(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        acc = a[0]
        for i, x in enumerate(a):
            acc = alpha * x + (1 - alpha) * acc
            out[i] = acc
        return out

    pad = min(len(held), 15)  # edge padding so the filter settles instead of lagging at the ends
    padded = np.concatenate([np.full(pad, held[0]), held, np.full(pad, held[-1])])
    smoothed = ema(ema(padded)[::-1])[::-1][pad : pad + len(held)]  # forward + backward: no lag
    return np.clip(smoothed, lo, hi)


class Shot(BaseModel):
    start: float  # seconds from the clip start
    end: float
    mode: Literal["crop", "fit"]
    times: list[float] = []
    cx: list[float] = []


class CropPlan(BaseModel):
    src_w: int
    src_h: int
    crop_w: int
    detector: str = ""
    shots: list[Shot]

    def at(self, t: float) -> tuple[str, float]:
        shot = next((s for s in self.shots if s.start <= t < s.end), self.shots[-1])
        if shot.mode == "fit" or not shot.times:
            return shot.mode, self.src_w / 2
        return "crop", float(np.interp(t, shot.times, shot.cx))


def crop_width(src_w: int, src_h: int, out_w: int, out_h: int) -> int:
    w = int(round(src_h * out_w / out_h))
    return w - (w % 2)


def plan_shot(start: float, end: float, times: list[float], dets: list[list[Box]],
              src_w: int, crop_w: int) -> Shot:
    if not times:
        return Shot(start=start, end=end, mode="fit")
    tracks = build_tracks(dets)
    n = len(times)
    usable = [t for t in tracks if len(t.samples) / n >= MIN_PRESENCE]
    if not usable:
        return Shot(start=start, end=end, mode="fit")

    def weight(t: Track) -> float:
        return sum(b.area * (1 - 0.5 * abs(b.cx / src_w - 0.5)) for b in t.samples.values())

    # Two people who are both on screen most of the time and too far apart for one crop.
    strong = [t for t in usable if len(t.samples) / n >= 0.5]
    if len(strong) >= 2:
        centers = [float(np.mean([b.cx for b in t.samples.values()])) for t in strong]
        widths = [float(np.mean([b.w for b in t.samples.values()])) for t in strong]
        if max(centers) - min(centers) + max(widths) > crop_w * 0.9:
            return Shot(start=start, end=end, mode="fit")

    primary = max(usable, key=weight)
    known_k = sorted(primary.samples)
    raw = np.interp(range(n), known_k, [primary.samples[k].cx for k in known_k])
    path = smooth_path(raw, DEAD_ZONE * src_w, crop_w / 2, src_w - crop_w / 2)
    return Shot(start=start, end=end, mode="crop", times=[round(t, 3) for t in times],
                cx=[round(float(c), 1) for c in path])


def plan_crop(source: Path, start: float, end: float, scene_cuts: list[float], detector: FaceDetector,
              *, out_w: int, out_h: int, progress: Callable[[float], None] = lambda f: None) -> CropPlan:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {source}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = end - start
        cw = crop_width(src_w, src_h, out_w, out_h)
        if cw >= src_w:  # already vertical (or narrower): nothing to crop
            return CropPlan(src_w=src_w, src_h=src_h, crop_w=src_w, detector=detector.kind,
                            shots=[Shot(start=0, end=dur, mode="fit")])

        step = max(1, round(fps / SAMPLE_FPS))
        times: list[float] = []
        dets: list[list[Box]] = []
        for i, (t, frame) in enumerate(iter_frames(cap, start, end, fps)):
            if i % step == 0:
                times.append(t)
                dets.append(detector.detect(frame))
                progress(min(1.0, t / dur))
    finally:
        cap.release()

    bounds = [0.0] + sorted(c - start for c in scene_cuts if start + 0.2 < c < end - 0.2) + [dur]
    shots = []
    for s0, s1 in zip(bounds, bounds[1:]):
        idx = [i for i, t in enumerate(times) if s0 <= t < s1]
        shots.append(plan_shot(s0, s1, [times[i] for i in idx], [dets[i] for i in idx], src_w, cw))
    return CropPlan(src_w=src_w, src_h=src_h, crop_w=cw, detector=detector.kind, shots=shots)


def iter_frames(cap: cv2.VideoCapture, start: float, end: float, fps: float):
    """Yield (seconds from clip start, frame) for every frame in [start, end)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 1.0) * 1000)
    half = 0.5 / fps
    while True:
        ok = cap.grab()
        if not ok:
            return
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < start - half:
            continue
        if t >= end - half:
            return
        ok, frame = cap.retrieve()
        if not ok:
            return
        yield t - start, frame
