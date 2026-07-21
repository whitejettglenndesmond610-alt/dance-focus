from pathlib import Path

import torch

from dance_focus.geometry import Box
from dance_focus.sam2_tracker import _chunk_directory, _mask_box


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
