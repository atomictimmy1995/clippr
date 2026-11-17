import numpy as np

from clippr.pipeline.reframe import Box, CropPlan, Shot, build_tracks, crop_width, plan_shot, smooth_path


def test_crop_width_is_even_and_9_16():
    assert crop_width(1920, 1080, 1080, 1920) == 608
    assert crop_width(960, 540, 1080, 1920) == 304


def test_dead_zone_ignores_small_jitter():
    rng = np.random.default_rng(0)
    raw = 500 + rng.normal(0, 8, 50)  # +-8 px jitter, 48 px dead zone
    out = smooth_path(raw, dead_zone=48, lo=0, hi=1000)
    assert out.max() - out.min() < 5


def test_path_follows_big_moves_and_stays_in_bounds():
    raw = np.concatenate([np.full(20, 200.0), np.full(20, 800.0)])
    out = smooth_path(raw, dead_zone=40, lo=150, hi=810)
    assert out[0] < 300 and out[-1] > 700
    assert out.min() >= 150 and out.max() <= 810
    assert np.all(np.diff(out) >= -1e-6)  # monotone: no overshoot back and forth


def test_tracks_link_moving_face_and_keep_people_apart():
    dets = [[Box(100 + 10 * k, 100, 80, 80), Box(800, 100, 80, 80)] for k in range(10)]
    tracks = build_tracks(dets)
    assert len(tracks) == 2
    assert sorted(len(t.samples) for t in tracks) == [10, 10]


def test_plan_shot_follows_the_main_face():
    times = [i * 0.2 for i in range(20)]
    dets = [[Box(100 + 30 * i, 200, 120, 120)] for i in range(20)]
    shot = plan_shot(0, 4, times, dets, src_w=1920, crop_w=606)
    assert shot.mode == "crop"
    assert shot.cx[-1] > shot.cx[0] + 250  # the crop moves with the face
    for (box,), cx in zip(dets, shot.cx):  # and the face is always fully inside the crop
        assert cx - 303 <= box.x and box.x + box.w <= cx + 303


def test_plan_shot_uses_fit_for_no_face_or_two_far_apart_people():
    times = [i * 0.2 for i in range(20)]
    assert plan_shot(0, 4, times, [[] for _ in times], 1920, 606).mode == "fit"
    two = [[Box(150, 200, 120, 120), Box(1600, 200, 120, 120)] for _ in times]
    assert plan_shot(0, 4, times, two, 1920, 606).mode == "fit"


def test_crop_plan_interpolates_within_shot_and_jumps_at_cuts():
    plan = CropPlan(src_w=1920, src_h=1080, crop_w=606, shots=[
        Shot(start=0, end=2, mode="crop", times=[0, 1], cx=[400, 600]),
        Shot(start=2, end=4, mode="fit"),
    ])
    assert plan.at(0.5) == ("crop", 500.0)
    assert plan.at(2.5)[0] == "fit"
