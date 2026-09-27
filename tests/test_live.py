import time

import av
import pytest
from media import BOX, FRAMES, is_blurred, is_fully_blurred, make_video, read_frames

from crowdveil.live import relay


def run_relay(tmp_path, detect, **kwargs):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src)
    with av.open(str(src)) as inp:
        written = relay(inp, str(dst), detect, **kwargs)
    return written, dst


def test_relay_blurs_and_copies_audio(tmp_path):
    calls = iter(range(FRAMES))

    def detect(image):
        return [BOX] if next(calls) == 10 else []

    written, dst = run_relay(tmp_path, detect, radius=2, detect_every=1, max_backlog=FRAMES)
    frames = read_frames(dst)
    assert written == len(frames) == FRAMES
    assert [i for i, f in enumerate(frames) if is_blurred(f)] == [8, 9, 10, 11, 12]
    with av.open(str(dst)) as out:
        assert out.streams.audio[0].codec_context.name == "aac"


def test_relay_blurs_whole_frames_when_detection_fails(tmp_path):
    def detect(image):
        raise RuntimeError("boom")

    written, dst = run_relay(tmp_path, detect, max_backlog=FRAMES)
    frames = read_frames(dst)
    assert written == FRAMES
    assert all(is_fully_blurred(f) for f in frames)


def test_relay_blurs_whole_frames_when_behind(tmp_path):
    detected = 0

    def slow_detect(image):
        nonlocal detected
        detected += 1
        time.sleep(0.02)
        return []

    # A file is read far faster than detection runs, so the backlog builds immediately.
    written, dst = run_relay(tmp_path, slow_detect, radius=0, detect_every=1, max_backlog=3)
    frames = read_frames(dst)
    assert written == FRAMES
    assert detected < FRAMES
    assert sum(is_fully_blurred(f) for f in frames) == FRAMES - detected


def test_relay_detects_every_nth_frame_and_covers_the_rest(tmp_path):
    calls = []

    def detect(image):
        calls.append(len(calls))
        return [BOX] if len(calls) == 6 else []  # the 6th call sees frame 10

    written, dst = run_relay(tmp_path, detect, radius=1, detect_every=2, max_backlog=FRAMES)
    frames = read_frames(dst)
    assert len(calls) == FRAMES // 2
    assert [i for i, f in enumerate(frames) if is_blurred(f)] == [9, 10, 11]


def test_relay_rejects_radius_too_small_for_detect_every(tmp_path):
    with pytest.raises(ValueError, match="radius"):
        run_relay(tmp_path, lambda image: [], radius=1, detect_every=3)
