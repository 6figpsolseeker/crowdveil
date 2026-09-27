"""Face detection, with an allowlist of known faces that are never blurred."""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from crowdveil.models import FACE_DETECTOR, FACE_RECOGNIZER, fetch

Box = tuple[int, int, int, int]  # x, y, width, height

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cosine-similarity threshold recommended for SFace by the model authors.
MATCH_THRESHOLD = 0.363


class FaceFilter:
    """Finds faces in a frame and returns the ones that do not belong to the crew."""

    def __init__(self, min_score: float = 0.6, match_threshold: float = MATCH_THRESHOLD):
        self._detector = cv2.FaceDetectorYN.create(
            str(fetch(FACE_DETECTOR)), "", (320, 320), min_score
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(fetch(FACE_RECOGNIZER)), "")
        self._match_threshold = match_threshold
        self._crew: list[np.ndarray] = []

    def enroll(self, paths: Iterable[Path]) -> None:
        """Add the largest face in each image to the crew."""
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read image: {path}")
            faces = self._detect(image)
            if len(faces) == 0:
                raise ValueError(f"no face found in {path}")
            largest = max(faces, key=lambda f: f[2] * f[3])
            self._crew.append(self._embed(image, largest))

    def __call__(self, frame: np.ndarray) -> list[Box]:
        return [
            (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
            for f in self._detect(frame)
            if not self._is_crew(frame, f)
        ]

    def _detect(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def _embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        return self._recognizer.feature(self._recognizer.alignCrop(image, face)).copy()

    def _is_crew(self, frame: np.ndarray, face: np.ndarray) -> bool:
        if not self._crew:
            return False
        embedding = self._embed(frame, face)
        return any(
            self._recognizer.match(embedding, known, cv2.FaceRecognizerSF_FR_COSINE)
            >= self._match_threshold
            for known in self._crew
        )


def crew_images(directory: Path) -> list[Path]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no images in {directory}")
    return paths
