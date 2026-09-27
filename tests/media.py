"""Synthetic test media."""

import av
import numpy as np

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


def is_fully_blurred(image):
    return is_blurred(image) and image[HEIGHT // 2 :, WIDTH // 2 :].std() < 30
