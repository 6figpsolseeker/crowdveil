"""Decode video, blur detected faces, re-encode, and copy audio through."""

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from crowdveil.faces import Box
from crowdveil.redact import Window, blur

log = logging.getLogger(__name__)

Detector = Callable[[np.ndarray], list[Box]]
Frame = tuple[np.ndarray, int | None, Fraction | None]  # image, pts, time base
Demuxer = Callable[[list[av.stream.Stream]], Iterable[av.Packet]]

FILE_ENCODER = {"crf": "18", "preset": "medium"}


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
                return process(inp, out, detect, radius, FILE_ENCODER)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise


def process(
    inp: av.container.InputContainer,
    out: av.container.OutputContainer,
    detect: Detector,
    radius: int,
    encoder: dict[str, str],
    demux: Demuxer | None = None,
    keyframe_interval: float | None = None,
) -> int:
    """Blur the first video stream of `inp` into `out`, copying the first audio stream.

    `encoder` holds libx264 options. `demux` replaces `inp.demux` as the packet source.
    `keyframe_interval` forces a keyframe every that many seconds of stream time. Detection
    runs on a worker thread, one frame ahead of encoding. Returns the number of video frames
    written.
    """
    vin = inp.streams.video[0]
    vout = out.add_stream("libx264", rate=frame_rate(vin))
    vout.pix_fmt = "yuv420p"
    vout.time_base = vin.time_base
    vout.options = dict(encoder)

    ain = inp.streams.audio[0] if inp.streams.audio else None
    aout = out.add_stream_from_template(ain) if ain else None

    window: Window[Frame] = Window(radius)
    written = 0
    # The output can't start until the first frame gives its upright size.
    held: list[av.Packet] | None = []
    # The frame whose detection runs while the next frame is decoded.
    pending: tuple[Frame, Future[list[Box]]] | None = None
    next_keyframe: float | None = None

    def write(frame: Frame, boxes: list[Box]) -> None:
        nonlocal written, next_keyframe
        image, pts, time_base = frame
        blur(image, boxes)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)  # much faster than PyAV's conversion
        result = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        result.pts = pts
        result.time_base = time_base
        if keyframe_interval and pts is not None and time_base:
            now = float(pts * time_base)
            if next_keyframe is None or now >= next_keyframe:
                result.pict_type = av.video.frame.PictureType.I
                next_keyframe = now + keyframe_interval
        out.mux(vout.encode(result))
        written += 1

    def settle() -> None:
        if pending and (ready := window.push(pending[0], pending[1].result())):
            write(*ready)

    with ThreadPoolExecutor(max_workers=1) as pool:
        for packet in (demux or inp.demux)([s for s in (vin, ain) if s]):
            if packet.stream is ain:
                if packet.dts is not None:
                    packet.stream = aout
                    if held is None:
                        out.mux(packet)
                    else:
                        held.append(packet)
                continue
            try:
                frames = packet.decode()
            except av.InvalidDataError as e:
                log.warning("skipping undecodable packet: %s", e)
                continue
            for decoded in frames:
                image = upright(decoded)
                if held is not None:
                    vout.height, vout.width = image.shape[:2]
                    out.mux(held)
                    held = None
                elif image.shape[:2] != (vout.height, vout.width):
                    size = (vout.width, vout.height)
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                detection = pool.submit(detect, image)
                settle()
                pending = ((image, decoded.pts, decoded.time_base), detection)
        settle()

    if held is not None:
        raise ValueError("no decodable video frames")
    for ready in window.flush():
        write(*ready)
    out.mux(vout.encode(None))
    return written


def frame_rate(stream: av.video.stream.VideoStream) -> Fraction:
    """The stream's frame rate, or 30 when a live stream hasn't been probed long enough to
    tell (FLV then reports its 1000 Hz timebase)."""
    rate = stream.average_rate or stream.guessed_rate
    return rate if rate and rate <= 240 else Fraction(30)


def upright(frame: av.VideoFrame) -> np.ndarray:
    image = frame.to_ndarray(format="bgr24")
    turns = round(frame.rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(image, turns)) if turns else image
