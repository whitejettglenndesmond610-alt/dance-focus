from dance_focus.geometry import Box, Point, PoseAnchor
from dance_focus.pose import (
    PoseCandidate,
    interpolate_pose_anchors,
    match_pose_candidate,
)


def test_pose_matching_rejects_nearby_wrong_person():
    target = Box(100, 100, 100, 300)
    correct = PoseCandidate(
        Box(96, 95, 108, 310),
        0.98,
        PoseAnchor(Point(150, 220), 0.9),
    )
    wrong = PoseCandidate(
        Box(230, 90, 100, 310),
        0.99,
        PoseAnchor(Point(280, 220), 0.95),
    )

    match, quality = match_pose_candidate(target, [wrong, correct])

    assert match == correct
    assert quality > 0.7


def test_pose_anchor_interpolation_follows_per_frame_subject_box():
    boxes = [Box(index * 100, 0, 100, 200) for index in range(3)]
    sampled = {
        0: PoseAnchor(Point(50, 80), 0.9, Box(0, 0, 100, 200)),
        2: PoseAnchor(Point(250, 120), 0.8, Box(200, 0, 100, 200)),
    }

    anchors = interpolate_pose_anchors(boxes, sampled)

    assert anchors[1].point == Point(150, 100)
    assert round(anchors[1].confidence, 2) == 0.85
