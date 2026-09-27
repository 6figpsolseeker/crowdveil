import av
import numpy as np
import pytest

from crowdveil.video import redact_file

WIDTH, HEIGHT, FRAMES, RATE = 128, 96, 30, 30
BOX = (48, 32, 32, 32)
INSIDE = (slice(40, 56), slice(56, 72))  # well within BOX


def checkerboard():
    y, x = np.indices((HEIGHT, WIDTH))
    image = (((x // 4) + (y // 4)) % 2 * 255).astype(np.uint8)
    return np.dstack([image] * 3)


def make_video(path, rotation=None, video=True, audio=True):
    with av.open(str(path), "w") as out:
        vstream = out.add_stream("libx264", rate=RATE) if video else None
        astream = out.add_stream("aac", rate=48000, layout="mono") if audio else None

        if vstream:
            vstream.width, vstream.height, vstream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            vstream.options = {"crf": "10"}
            if rotation is not None:
                vstream.set_display_rotation(rotation)
            image = checkerboard()
            image[:16, :16] = (0, 0, 255)  # red marker, top-left
            for i in range(FRAMES):
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = i
                out.mux(vstream.encode(frame))
            out.mux(vstream.encode(None))

        if astream:
            samples = 48000 * FRAMES // RATE
            tone = np.sin(np.arange(samples) * 2 * np.pi * 440 / 48000).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(tone[None, :], format="fltp", layout="mono")
            frame.sample_rate, frame.pts = 48000, 0
            out.mux(astream.encode(frame))
            out.mux(astream.encode(None))


def read_frames(path):
    with av.open(str(path)) as inp:
        return [f.to_ndarray(format="bgr24") for f in inp.decode(video=0)]


def is_blurred(image):
    return image[INSIDE].std() < 30


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
