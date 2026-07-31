import dance_focus.analysis as analysis_module
from dance_focus.analysis import (
    affected_correction_interval,
    _identity_recovery_is_plausible,
    _interpolate_sampled_boxes,
    _tracking_confidences,
    splice_tracking_result,
    reanalyze_subject_interval,
)
from dance_focus.geometry import Box, TrackSample, TrackState, TrackingResult
from dance_focus.video import VideoInfo


def test_tracking_confidence_flags_large_identity_jump():
    boxes = [
        Box(100, 100, 80, 220),
        Box(104, 100, 80, 220),
        Box(108, 100, 80, 220),
        Box(700, 100, 160, 100),
        Box(704, 100, 160, 100),
    ]

    quality = _tracking_confidences(boxes)

    assert quality[1] > 0.8
    assert quality[3] < 0.42


def test_reid_rejects_impossible_short_jump_but_allows_nearby_candidate():
    previous = Box(400, 300, 80, 140)

    assert _identity_recovery_is_plausible(
        previous, Box(500, 300, 80, 140), 3, 1280, 3
    )
    assert not _identity_recovery_is_plausible(
        previous, Box(80, 300, 80, 140), 3, 1280, 3
    )


def test_pose_candidates_refine_boxes_between_sampled_frames():
    original = [Box(0, 0, 100, 200)] * 5

    refined = _interpolate_sampled_boxes(
        original,
        {0: Box(100, 20, 80, 220), 4: Box(500, 20, 80, 220)},
    )

    assert refined[0] == Box(100, 20, 80, 220)
    assert refined[2] == Box(300, 20, 80, 220)
    assert refined[4] == Box(500, 20, 80, 220)


def test_correction_interval_runs_forward_to_next_prompt():
    old = {0: Box(0, 0, 10, 10), 80: Box(80, 0, 10, 10)}
    new = {**old, 30: Box(30, 0, 10, 10)}

    assert affected_correction_interval(old, new, 30, 120) == (30, 80)


def test_deleting_correction_merges_from_previous_prompt():
    old = {
        0: Box(0, 0, 10, 10),
        30: Box(30, 0, 10, 10),
        80: Box(80, 0, 10, 10),
    }
    new = {0: old[0], 80: old[80]}

    assert affected_correction_interval(old, new, 30, 120) == (0, 80)


def test_splice_tracking_result_preserves_samples_outside_patch():
    previous = TrackingResult(
        tuple(
            TrackSample(Box(index, 0, 10, 10), None, TrackState.TRACKED, 0.8)
            for index in range(5)
        ),
        "old",
    )
    replacement = TrackingResult(
        (
            TrackSample(Box(20, 0, 10, 10), None, TrackState.MANUAL, 1.0),
            TrackSample(Box(30, 0, 10, 10), None, TrackState.TRACKED, 0.9),
        ),
        "new",
    )

    result = splice_tracking_result(previous, replacement, 2, 4)

    assert result.samples[:2] == previous.samples[:2]
    assert result.samples[2:4] == replacement.samples
    assert result.samples[4:] == previous.samples[4:]


def test_interval_reanalysis_only_replaces_affected_samples(monkeypatch, tmp_path):
    video = VideoInfo(tmp_path / "clip.mp4", 640, 360, 30, 5)
    previous = TrackingResult(
        tuple(
            TrackSample(Box(index, 0, 10, 10), None, TrackState.TRACKED, 0.8)
            for index in range(5)
        ),
        "old",
    )
    prompts = {0: Box(0, 0, 10, 10), 2: Box(200, 0, 10, 10)}
    monkeypatch.setattr(
        analysis_module,
        "track_subject_interval",
        lambda *_args, **_kwargs: [prompts[2], Box(210, 0, 10, 10)],
    )

    def fake_analyze(_info, _prompts, **kwargs):
        boxes = kwargs["sam_boxes_override"]
        return TrackingResult(
            tuple(
                TrackSample(box, None, TrackState.TRACKED, 0.9)
                for box in boxes
            ),
            "new",
        )

    monkeypatch.setattr(analysis_module, "analyze_subject", fake_analyze)

    result = reanalyze_subject_interval(video, prompts, previous, 2, 4)

    assert result.samples[:2] == previous.samples[:2]
    assert result.samples[4:] == previous.samples[4:]
    assert result.samples[2].box == prompts[2]
    assert result.samples[3].box == Box(210, 0, 10, 10)
