import cv2
import numpy as np
import pytest

from crowdveil.faces import FaceFilter, crew_images
from crowdveil.models import Model, fetch

_SAMPLES = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "bed6800f056e57f6c945a7bd56e324e0fa965e24/samples/data"
)
LENA = Model(
    "lena.jpg",
    f"{_SAMPLES}/lena.jpg",
    "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07",
)
MESSI = Model(
    "messi5.jpg",
    f"{_SAMPLES}/messi5.jpg",
    "1d570e49654e84c7a943918537bd9e5e1ef82920152e147c834006e235be97c9",
)


@pytest.fixture(scope="module")
def faces():
    return FaceFilter()


@pytest.fixture(scope="module")
def scene():
    """Two different people side by side: Lena on the left, Messi on the right."""
    lena = cv2.imread(str(fetch(LENA)))
    messi = cv2.imread(str(fetch(MESSI)))
    messi = cv2.resize(messi, (messi.shape[1] * lena.shape[0] // messi.shape[0], lena.shape[0]))
    return np.hstack([lena, messi]), lena.shape[1]


@pytest.mark.network
def test_detects_every_face_without_crew(faces, scene):
    image, split = scene
    boxes = faces(image)
    assert len(boxes) == 2
    assert sorted(x < split for x, _, _, _ in boxes) == [False, True]


@pytest.mark.network
def test_crew_member_is_not_returned(scene):
    image, split = scene
    faces = FaceFilter()
    faces.enroll([fetch(LENA)])
    boxes = faces(image)
    assert len(boxes) == 1
    assert boxes[0][0] >= split


@pytest.mark.network
def test_enroll_rejects_image_without_face(faces, tmp_path):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.zeros((200, 200, 3), np.uint8))
    with pytest.raises(ValueError, match="no face"):
        faces.enroll([blank])


def test_crew_images_filters_by_suffix(tmp_path):
    (tmp_path / "a.JPG").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "notes.txt").touch()
    assert [p.name for p in crew_images(tmp_path)] == ["a.JPG", "b.png"]


def test_crew_images_rejects_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        crew_images(tmp_path)
