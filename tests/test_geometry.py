import pytest

from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    FramingSettings,
    Point,
    PoseAnchor,
    TrackSample,
    TrackState,
    TrackingResult,
    build_camera_path,
    build_crop_path,
    crop_size_for_aspect,
)


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


def test_dynamic_camera_path_changes_source_window_but_keeps_output_size():
    heights = [300] * 20 + [650] * 20 + [300] * 20
    boxes = [Box(700, 200, 180, height) for height in heights]
    tracking = TrackingResult(
        samples=tuple(
            TrackSample(box, None, TrackState.TRACKED, 0.9) for box in boxes
        ),
        engine_version="test",
    )

    path = build_camera_path(
        tracking,
        (1920, 1080),
        30,
        FramingSettings(9 / 16, smoothing_seconds=0, auto_zoom=True),
    )

    assert path.output_size == (608, 1080)
    assert len({frame.height for frame in path.frames}) > 1
    assert all(frame.width <= path.output_size[0] for frame in path.frames)
    assert all(frame.height <= path.output_size[1] for frame in path.frames)


def test_balanced_preset_limits_automatic_pan_speed_and_acceleration():
    centers = [450] * 20 + [650] * 20 + [450] * 20
    tracking = TrackingResult(
        samples=tuple(
            TrackSample(
                Box(center - 20, 300, 40, 80),
                PoseAnchor(Point(center, 340), 0.9),
                TrackState.TRACKED,
                0.9,
            )
            for center in centers
        ),
        engine_version="test",
    )

    path = build_camera_path(
        tracking,
        (1280, 720),
        30,
        FramingSettings(1.0, smoothing_seconds=0.25, auto_zoom=False),
    )
    positions = [crop.x + crop.width / 2 for crop in path.frames]
    velocities = [current - previous for previous, current in zip(positions, positions[1:])]
    accelerations = [
        current - previous for previous, current in zip(velocities, velocities[1:])
    ]

    assert max(abs(value) for value in velocities) <= 23
    assert max(abs(value) for value in accelerations) <= 4


def test_camera_keyframes_override_zoom_and_follow_position():
    samples = tuple(
        TrackSample(
            Box(100 + index * 100, 200, 120, 400),
            PoseAnchor(Point(160 + index * 100, 390), 0.9),
            TrackState.TRACKED,
            0.9,
        )
        for index in range(5)
    )
    tracking = TrackingResult(samples=samples, engine_version="test")

    path = build_camera_path(
        tracking,
        (1280, 720),
        30,
        FramingSettings(1.0, smoothing_seconds=0, max_zoom=2),
        [
            CameraKeyframe(
                frame_index=2,
                center=Point(420, 360),
                zoom=2.0,
                follow_strength=0.0,
            )
        ],
    )

    middle = path.frames[2]
    assert middle.width == 360
    assert middle.height == 360
    assert middle.x == 240
    assert middle.y == 220


def test_manual_position_remains_visible_with_full_auto_follow():
    samples = tuple(
        TrackSample(
            Box(300 + index * 20, 300, 80, 120),
            PoseAnchor(Point(340 + index * 20, 360), 0.9),
            TrackState.TRACKED,
            0.9,
        )
        for index in range(5)
    )
    tracking = TrackingResult(samples=samples, engine_version="test")

    path = build_camera_path(
        tracking,
        (1280, 720),
        30,
        FramingSettings(1.0, smoothing_seconds=0, auto_zoom=False, max_zoom=2),
        [
            CameraKeyframe(
                frame_index=2,
                center=Point(450, 360),
                zoom=2.0,
                follow_strength=1.0,
            )
        ],
    )

    middle = path.frames[2]
    assert middle.x + middle.width / 2 == 450
    assert path.frames[4].x > middle.x


def test_camera_path_rejects_short_tracking_excursion():
    centers = [400 + index * 2 for index in range(60)]
    for index, excursion in zip(
        range(27, 34), (0, -100, -200, -300, -200, -100, 0), strict=True
    ):
        centers[index] += excursion
    tracking = TrackingResult(
        samples=tuple(
            TrackSample(
                Box(center - 40, 300, 80, 120),
                PoseAnchor(Point(center, 360), 0.9),
                TrackState.TRACKED,
                0.9,
            )
            for center in centers
        ),
        engine_version="test",
    )

    path = build_camera_path(
        tracking,
        (1280, 720),
        30,
        FramingSettings(1.0, smoothing_seconds=0.1, auto_zoom=False),
    )
    crop_centers = [crop.x + crop.width / 2 for crop in path.frames]
    largest_step = max(
        abs(current - previous)
        for previous, current in zip(crop_centers, crop_centers[1:])
    )

    assert largest_step < 30
