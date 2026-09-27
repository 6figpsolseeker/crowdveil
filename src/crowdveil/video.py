"""Redact video files: decode, blur detected faces, re-encode, and copy audio through."""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from crowdveil.faces import Box
from crowdveil.redact import Window, blur

Detector = Callable[[np.ndarray], list[Box]]
Frame = tuple[np.ndarray, int | None, Fraction | None]  # image, pts, time base


def redact_file(src: Path, dst: Path, detect: Detector, radius: int = 5) -> int:
    """Write `src` to `dst` with every box returned by `detect` blurred.

    Frames are rotated upright according to the source's display matrix. Audio is copied
    without re-encoding. Returns the number of video frames written. On failure, `dst` is
    removed.
    """
    with av.open(str(src)) as inp:
        if not inp.streams.video:
            raise ValueError(f"no video stream in {src}")
        try:
            with av.open(str(dst), "w") as out:
                return _redact(inp, out, detect, radius)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise


def _redact(
    inp: av.container.InputContainer,
    out: av.container.OutputContainer,
    detect: Detector,
    radius: int,
) -> int:
    vin = inp.streams.video[0]
    vout = out.add_stream("libx264", rate=vin.guessed_rate or vin.average_rate or 30)
    vout.pix_fmt = "yuv420p"
    vout.time_base = vin.time_base
    vout.options = {"crf": "18", "preset": "medium"}

    ain = inp.streams.audio[0] if inp.streams.audio else None
    aout = out.add_stream_from_template(ain) if ain else None

    window: Window[Frame] = Window(radius)
    written = 0
    # The output can't start until the first frame gives its upright size.
    held: list[av.Packet] | None = []

    def write(frame: Frame, boxes: list[Box]) -> None:
        nonlocal written
        image, pts, time_base = frame
        blur(image, boxes)
        result = av.VideoFrame.from_ndarray(image, format="bgr24")
        result.pts = pts
        result.time_base = time_base
        out.mux(vout.encode(result))
        written += 1

    for packet in inp.demux([s for s in (vin, ain) if s]):
        if packet.stream is ain:
            if packet.dts is not None:
                packet.stream = aout
                if held is None:
                    out.mux(packet)
                else:
                    held.append(packet)
            continue
        for decoded in packet.decode():
            image = upright(decoded)
            if held is not None:
                vout.height, vout.width = image.shape[:2]
                out.mux(held)
                held = None
            if ready := window.push((image, decoded.pts, decoded.time_base), detect(image)):
                write(*ready)

    if held is not None:
        raise ValueError("no decodable video frames")
    for ready in window.flush():
        write(*ready)
    out.mux(vout.encode(None))
    return written


def upright(frame: av.VideoFrame) -> np.ndarray:
    image = frame.to_ndarray(format="bgr24")
    turns = round(frame.rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(image, turns)) if turns else image
