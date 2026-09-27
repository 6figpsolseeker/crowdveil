from fractions import Fraction
from types import SimpleNamespace

import av
import pytest
from media import BOX, FRAMES, HEIGHT, INSIDE, WIDTH, is_blurred, make_video, read_frames

from crowdveil.video import frame_rate, process, redact_file


def test_blurs_detected_frames_and_neighbours(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src)
    calls = iter(range(FRAMES))

    def detect(image):
        return [BOX] if next(calls) == 10 else []

    assert redact_file(src, dst, detect, radius=2) == FRAMES
    frames = read_frames(dst)
    assert len(frames) == FRAMES
    assert [i for i, f in enumerate(frames) if is_blurred(f)] == [8, 9, 10, 11, 12]
    assert frames[0][INSIDE].std() > 100


def test_copies_audio(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src)
    redact_file(src, dst, lambda image: [])
    with av.open(str(src)) as a, av.open(str(dst)) as b:
        assert b.streams.audio[0].codec_context.name == "aac"
        assert b.streams.audio[0].frames == a.streams.audio[0].frames


def test_without_audio(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, audio=False)
    assert redact_file(src, dst, lambda image: []) == FRAMES
    with av.open(str(dst)) as b:
        assert not b.streams.audio


def test_rejects_input_without_video(tmp_path):
    src, dst = tmp_path / "in.m4a", tmp_path / "out.mp4"
    make_video(src, video=False)
    with pytest.raises(ValueError, match="no video stream"):
        redact_file(src, dst, lambda image: [])
    assert not dst.exists()


def test_removes_output_on_failure(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src)

    def detect(image):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        redact_file(src, dst, detect)
    assert not dst.exists()


@pytest.mark.skipif(
    not hasattr(av.video.stream.VideoStream, "set_display_rotation"),
    reason="writing rotated fixtures needs a newer PyAV",
)
def test_applies_display_rotation(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, rotation=-90)  # portrait phone footage: rotate 90 degrees clockwise
    seen = []

    def detect(image):
        seen.append(image.shape[:2])
        return []

    redact_file(src, dst, detect)
    assert set(seen) == {(WIDTH, HEIGHT)}
    frame = read_frames(dst)[0]
    assert frame.shape[:2] == (WIDTH, HEIGHT)
    red = frame[..., 2].astype(int) - frame[..., 1]
    assert red[:16, -16:].mean() > 150  # marker moved from top-left to top-right


def test_forces_keyframes_by_stream_time(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src)
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        process(inp, out, lambda image: [], 0, {"g": "1000"}, keyframe_interval=0.25)
    with av.open(str(dst)) as result:
        keys = [
            i for i, p in enumerate(p for p in result.demux(video=0) if p.size) if p.is_keyframe
        ]
    assert keys == [0, 8, 16, 24]  # 30 fps: first frame at or after 0, 0.25, 0.5, 0.75 s


def test_frame_rate_falls_back_when_unprobed():
    assert frame_rate(SimpleNamespace(average_rate=Fraction(30000, 1001), guessed_rate=None)) == (
        Fraction(30000, 1001)
    )
    assert frame_rate(SimpleNamespace(average_rate=None, guessed_rate=Fraction(60))) == 60
    assert frame_rate(SimpleNamespace(average_rate=None, guessed_rate=Fraction(1000))) == 30
    assert frame_rate(SimpleNamespace(average_rate=None, guessed_rate=None)) == 30
