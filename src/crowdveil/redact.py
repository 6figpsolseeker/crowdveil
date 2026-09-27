"""Temporal smoothing of detections and in-place blurring of frames."""

from collections import deque
from typing import Generic, TypeVar

import cv2
import numpy as np

from crowdveil.faces import Box

T = TypeVar("T")


class Window(Generic[T]):
    """Delays items by `radius` and pairs each with every box detected within `radius` items
    on either side, so a face stays covered through missed detections and from the first
    frame it appears in."""

    def __init__(self, radius: int):
        if radius < 0:
            raise ValueError("radius must be non-negative")
        self.radius = radius
        self._items: deque[tuple[T, list[Box]]] = deque()
        self._pending = 0

    def push(self, item: T, boxes: list[Box]) -> tuple[T, list[Box]] | None:
        self._items.append((item, boxes))
        self._pending += 1
        return self._emit() if self._pending > self.radius else None

    def flush(self) -> list[tuple[T, list[Box]]]:
        return [self._emit() for _ in range(self._pending)]

    def _emit(self) -> tuple[T, list[Box]]:
        index = len(self._items) - self._pending
        start = max(0, index - self.radius)
        stop = index + self.radius + 1
        boxes = list(
            dict.fromkeys(
                b for i in range(start, min(stop, len(self._items))) for b in self._items[i][1]
            )
        )
        item = self._items[index][0]
        self._pending -= 1
        while len(self._items) - self._pending > self.radius:
            self._items.popleft()
        return item, boxes


def blur(frame: np.ndarray, boxes: list[Box], padding: float = 0.25) -> None:
    """Blur each box in `frame` in place, expanded by `padding` of its size on every side.

    The region is downsampled to a few pixels before being scaled back up, so no fine
    detail survives to be recovered.
    """
    height, width = frame.shape[:2]
    for x, y, w, h in boxes:
        dx, dy = int(w * padding), int(h * padding)
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(width, x + w + dx), min(height, y + h + dy)
        if x1 <= x0 or y1 <= y0:
            continue
        region = frame[y0:y1, x0:x1]
        small = cv2.resize(region, (8, 8), interpolation=cv2.INTER_AREA)
        region[:] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
