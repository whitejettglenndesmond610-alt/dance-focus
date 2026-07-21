import pytest

from dance_focus.geometry import Box, build_crop_path, crop_size_for_aspect


def test_vertical_crop_uses_full_source_height():
    assert crop_size_for_aspect(1920, 1080, 9 / 16) == (608, 1080)


def test_horizontal_crop_uses_full_source_width():
    assert crop_size_for_aspect(1080, 1920, 16 / 9) == (1080, 608)


def test_crop_path_tracks_subject_and_stays_in_source():
    boxes = [Box(x, 200, 180, 600) for x in (20, 300, 700, 1200, 1700)]
    path = build_crop_path(boxes, (1920, 1080), (606, 1080), 30, 0.5)

    assert len(path) == len(boxes)
    assert all(0 <= crop.x <= 1920 - crop.width for crop in path)
    assert all(crop.y == 0 for crop in path)
    for box, crop in zip(boxes, path, strict=True):
        assert crop.x <= box.x
        assert crop.x + crop.width >= box.right


def test_crop_path_interpolates_temporarily_lost_tracking():
    boxes = [Box(100, 100, 100, 300), None, None, Box(500, 100, 100, 300)]
    path = build_crop_path(boxes, (1000, 600), (400, 600), 30, 0)

    assert [crop.x for crop in path] == sorted(crop.x for crop in path)


def test_crop_path_requires_at_least_one_box():
    with pytest.raises(ValueError):
        build_crop_path([None, None], (1920, 1080), (606, 1080), 30)
