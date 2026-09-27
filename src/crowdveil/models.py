"""Download and cache model weights, verified by SHA-256."""

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_ZOO = "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models"


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str


FACE_DETECTOR = Model(
    "face_detection_yunet_2023mar.onnx",
    f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)

FACE_RECOGNIZER = Model(
    "face_recognition_sface_2021dec.onnx",
    f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "crowdveil"


def fetch(model: Model) -> Path:
    """Return the local path to `model`, downloading it on first use."""
    path = cache_dir() / model.filename
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        try:
            with urllib.request.urlopen(model.url) as response:
                digest = hashlib.sha256()
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    tmp.write(chunk)
            if digest.hexdigest() != model.sha256:
                raise RuntimeError(f"checksum mismatch for {model.filename}")
        except BaseException:
            tmp.close()
            os.unlink(tmp.name)
            raise
    os.replace(tmp.name, path)
    return path
