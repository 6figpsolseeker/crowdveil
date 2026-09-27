import cv2
import numpy as np
import pytest

from crowdveil.faces import RECHECK_AFTER, FaceFilter, crew_images, iou
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
def test_max_width_returns_boxes_in_frame_coordinates(faces, scene):
    image, _ = scene
    full = faces(image)
    downscaled = FaceFilter(max_width=image.shape[1] // 2)(image)
    assert len(full) == 2
    for a in full:
        assert any(np.allclose(a, b, atol=0.1 * max(a[2], a[3])) for b in downscaled)


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


@pytest.mark.network
def test_tracked_bystanders_skip_recognition_until_recheck(scene, monkeypatch):
    image, split = scene
    faces = FaceFilter()
    faces.enroll([fetch(LENA)])
    recognised = []
    embed = faces._embed
    monkeypatch.setattr(
        faces, "_embed", lambda img, face: recognised.append(face[0]) or embed(img, face)
    )

    expected = faces(image)
    assert [x >= split for x, _, _, _ in expected] == [True]
    assert len(recognised) == 2

    for _ in range(RECHECK_AFTER):
        recognised.clear()
        assert faces(image) == expected
        assert len(recognised) == 1  # only the crew member is recognised again

    recognised.clear()
    assert faces(image) == expected
    assert len(recognised) == 2


def test_iou():
    a = np.array([0, 0, 10, 10])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([5, 0, 10, 10])) == pytest.approx(50 / 150)
    assert iou(a, np.array([10, 0, 10, 10])) == 0.0
