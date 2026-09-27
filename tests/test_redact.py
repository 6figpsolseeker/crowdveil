import numpy as np
import pytest

from crowdveil.redact import Window, blur


def run(window, detections):
    out = [r for i, boxes in enumerate(detections) if (r := window.push(i, boxes))]
    return out + window.flush()


def test_window_holds_boxes_across_radius():
    box = (1, 2, 3, 4)
    detections = [[], [], [], [box], [], [], [], []]
    result = run(Window(2), detections)
    assert [item for item, _ in result] == list(range(8))
    assert [i for i, boxes in result if boxes] == [1, 2, 3, 4, 5]


def test_window_delays_by_radius():
    window = Window(3)
    assert [window.push(i, []) for i in range(3)] == [None] * 3
    assert window.push(3, []) == (0, [])


def test_window_radius_zero_passes_through():
    window = Window(0)
    assert window.push("a", [(0, 0, 1, 1)]) == ("a", [(0, 0, 1, 1)])
    assert window.flush() == []


def test_window_edges_and_short_input():
    box = (0, 0, 1, 1)
    result = run(Window(5), [[box], [], []])
    assert [item for item, _ in result] == [0, 1, 2]
    assert all(boxes == [box] for _, boxes in result)


def test_window_merges_repeated_boxes():
    box = (0, 0, 1, 1)
    result = run(Window(2), [[box], [box, (5, 5, 1, 1)], [box]])
    assert all(boxes.count(box) == 1 for _, boxes in result)


def test_window_rejects_negative_radius():
    with pytest.raises(ValueError):
        Window(-1)


def test_blur_changes_only_padded_box():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    blur(frame, [(40, 40, 20, 20)], padding=0.25)

    inside = (slice(35, 65), slice(35, 65))
    assert not np.array_equal(frame[inside], original[inside])
    mask = np.ones((100, 100), bool)
    mask[inside] = False
    assert np.array_equal(frame[mask], original[mask])


def test_blur_destroys_detail():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blur(frame, [(0, 0, 64, 64)], padding=0)
    assert np.abs(np.diff(frame.astype(int), axis=1)).mean() < 10


def test_blur_clips_to_frame_and_skips_offscreen():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    original = frame.copy()
    blur(frame, [(-5, -5, 20, 20), (100, 100, 10, 10)], padding=0.25)
    assert not np.array_equal(frame[:20, :20], original[:20, :20])
    assert np.array_equal(frame[20:], original[20:])
    assert np.array_equal(frame[:, 20:], original[:, 20:])
