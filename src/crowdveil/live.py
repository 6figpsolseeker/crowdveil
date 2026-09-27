"""Relay a live stream: pull from an ingest server, blur, and publish."""

import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import NoReturn

import av
import numpy as np

from crowdveil.faces import Box
from crowdveil.video import Detector, process

log = logging.getLogger(__name__)

READ_TIMEOUT = 5.0  # seconds without data before the input is considered gone
KEYFRAME_INTERVAL = 2.0  # seconds; what Twitch and YouTube ask for

# Probe only the start of a live input. The default probes for seconds, delaying the start
# and queueing a backlog. The frame rate may then be unknown, so the encoder budgets bits
# from timestamps (force-cfr=0) rather than from a declared rate.
PROBE = {"analyzeduration": "500000"}


def serve(src: str, dst: str, detect: Detector, **options) -> NoReturn:
    """Relay every stream published at `src` to `dst`, reconnecting between streams.

    `options` are passed to `relay`.
    """
    waiting = False
    while True:
        try:
            inp = open_input(src)
        except (OSError, av.FFmpegError):
            if not waiting:
                log.info("waiting for a stream at %s", src)
                waiting = True
            time.sleep(0.5)
            continue
        waiting = False
        log.info("stream started")
        try:
            with inp:
                frames = relay(inp, dst, detect, **options)
            log.info("stream ended after %d frames", frames)
        except OSError as e:
            # The message alone: the filename part would include the stream key.
            log.error("relay failed: %s", e.strerror or type(e).__name__)
            time.sleep(2)
        except Exception:
            log.exception("relay failed")
            time.sleep(2)


def open_input(src: str) -> av.container.InputContainer:
    options = dict(PROBE)
    if src.startswith("rtsp"):
        options["rtsp_transport"] = "tcp"  # UDP drops packets under load
    return av.open(src, options=options, timeout=(READ_TIMEOUT, READ_TIMEOUT))


def relay(
    inp: av.container.InputContainer,
    dst: str,
    detect: Detector,
    radius: int = 5,
    bitrate: int = 6000,
    detect_every: int = 2,
    max_backlog: int = 15,
) -> int:
    """Blur `inp` and publish it to `dst` until the input ends. `bitrate` is in kbit/s.

    Detection runs on every `detect_every`th frame; `radius` must be at least
    `detect_every - 1` so the frames between are covered. Frames are blurred whole, never
    passed through unblurred, if detection fails or if more than `max_backlog` frames are
    waiting; detection resumes once a third of that remain. Returns the number of video
    frames written.
    """
    if detect_every < 1 or radius < detect_every - 1:
        raise ValueError("radius must be at least detect_every - 1")
    if not inp.streams.video:
        raise ValueError("no video stream in input")
    encoder = {
        "preset": "veryfast",
        "tune": "zerolatency",
        "x264-params": "force-cfr=0",
        "sc_threshold": "0",
        "b": f"{bitrate}k",
        "maxrate": f"{bitrate}k",
        "bufsize": f"{bitrate * 2}k",
    }
    reader = _Reader(inp)
    try:
        with av.open(dst, "w", format="flv" if dst.startswith("rtmp") else None) as out:
            guarded = _failsafe(detect, reader, detect_every, max_backlog)
            return process(
                inp,
                out,
                guarded,
                radius,
                encoder,
                demux=reader,
                keyframe_interval=KEYFRAME_INTERVAL,
            )
    finally:
        reader.stop()


class _Reader:
    """Demuxes on a background thread, so the backlog of unprocessed frames is known."""

    def __init__(self, inp: av.container.InputContainer):
        self._inp = inp
        self._queue: queue.Queue[av.Packet | None] = queue.Queue()
        self._read = 0  # written only by the reader thread
        self._taken = 0  # written only by the consumer
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def backlog(self) -> int:
        return self._read - self._taken

    def __call__(self, streams: list[av.stream.Stream]) -> Iterator[av.Packet]:
        video = streams[0]
        self._thread = threading.Thread(target=self._run, args=(streams, video), daemon=True)
        self._thread.start()
        while (packet := self._queue.get()) is not None:
            if packet.stream is video:
                self._taken += 1
            yield packet
        if self._error:
            raise self._error

    def stop(self) -> None:
        """Stop reading and wait for the thread, so the input can be closed safely."""
        self._stopping.set()
        if self._thread:
            self._thread.join()

    def _run(self, streams: list[av.stream.Stream], video: av.stream.Stream) -> None:
        try:
            for packet in self._inp.demux(streams):
                if self._stopping.is_set():
                    break
                if packet.stream is video:
                    self._read += 1
                self._queue.put(packet)
        except OSError as e:
            log.info("input closed: %s", e.strerror or type(e).__name__)
        except BaseException as e:
            self._error = e
        finally:
            self._queue.put(None)


def _failsafe(detect: Detector, reader: _Reader, every: int, max_backlog: int) -> Detector:
    behind = failing = False
    frame = -1

    def guarded(image: np.ndarray) -> list[Box]:
        nonlocal behind, failing, frame
        frame += 1
        whole = [(0, 0, image.shape[1], image.shape[0])]
        backlog = reader.backlog
        if behind and backlog <= max_backlog // 3:
            log.info("caught up with the stream")
            behind = False
        elif not behind and backlog > max_backlog:
            log.warning("falling behind the stream; blurring whole frames until caught up")
            behind = True
        if behind:
            return whole
        if frame % every:
            return []
        try:
            boxes = detect(image)
        except Exception:
            if not failing:
                log.exception("face detection failed; blurring whole frames")
                failing = True
            return whole
        if failing:
            log.info("face detection recovered")
            failing = False
        return boxes

    return guarded
