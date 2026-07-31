from pathlib import Path

import torch

from dance_focus.geometry import Box
from dance_focus.sam2_tracker import _chunk_directory, _mask_box, _track_range


def test_mask_box_uses_positive_sam_logits():
    logits = torch.full((1, 1, 12, 20), -1.0)
    logits[0, 0, 2:9, 4:16] = 1.0

    box, mask = _mask_box(logits)

    assert box == Box(4, 2, 12, 7)
    assert mask.shape == (12, 20)
    assert int(mask.sum()) == 84


def test_chunk_directory_links_global_frames(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for frame_index in range(5):
        (frames / f"{frame_index:06d}.jpg").write_bytes(b"frame")

    chunk = _chunk_directory(frames, 2, 5)

    links = sorted(chunk.glob("*.jpg"))
    assert [link.name for link in links] == ["000000.jpg", "000001.jpg", "000002.jpg"]
    assert all(link.is_symlink() for link in links)
    assert links[0].resolve() == frames / "000002.jpg"


def test_track_range_uses_manual_box_as_hard_interval_boundary(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for frame_index in range(8):
        (frames / f"{frame_index:06d}.jpg").write_bytes(b"frame")

    class FakePredictor:
        def __init__(self):
            self.box_prompts = []
            self.mask_prompts = []

        def init_state(self, video_path, **_kwargs):
            return {"count": len(list(Path(video_path).glob("*.jpg")))}

        def add_new_points_or_box(self, state, frame_idx, obj_id, box):
            self.box_prompts.append((frame_idx, obj_id, tuple(box)))

        def add_new_mask(self, state, frame_idx, obj_id, mask):
            self.mask_prompts.append((frame_idx, obj_id, mask.shape))

        def propagate_in_video(self, state):
            for index in range(state["count"]):
                logits = torch.full((1, 1, 16, 16), -1.0)
                logits[0, 0, 3:9, 4 + index : 8 + index] = 1.0
                yield index, [1], logits

    predictor = FakePredictor()
    manual = Box(20, 30, 40, 50)

    result = _track_range(
        predictor,
        "cpu",
        frames,
        2,
        8,
        manual,
        None,
        None,
        chunk_frames=4,
        overlap_frames=1,
    )

    assert len(result) == 6
    assert result[0] == manual
    assert len(predictor.box_prompts) == 1
    assert len(predictor.mask_prompts) == 1
