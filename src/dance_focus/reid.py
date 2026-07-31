from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import cv2
import numpy as np

from dance_focus.geometry import Box
from dance_focus.sam2_tracker import cache_root
from dance_focus.vendor.osnet_ain import osnet_ain_x1_0


REID_MODEL_NAME = "OSNet-AIN x1.0 MSMT17"
REID_FILENAME = "osnet_ain_x1_0_msmt17.pth"
REID_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/"
    "a5c5cc037c24235cda3b21085b93ad77c9616224/"
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
    "b64_fb10_softmax_labsmth_flip_jitter.pth?download=true"
)
REID_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
REID_SIZE = 17_293_009

_MODEL = None
_DEVICE = None
_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_reid_checkpoint(
    progress: Callable[[int], None] | None = None,
) -> Path:
    model_dir = cache_root() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / REID_FILENAME
    marker = checkpoint.with_suffix(".verified")
    if checkpoint.exists() and marker.exists():
        if marker.read_text(encoding="ascii").strip() == REID_SHA256:
            if progress:
                progress(100)
            return checkpoint
    if checkpoint.exists() and _sha256(checkpoint) == REID_SHA256:
        marker.write_text(REID_SHA256, encoding="ascii")
        if progress:
            progress(100)
        return checkpoint

    checkpoint.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    partial = checkpoint.with_suffix(".part")
    partial.unlink(missing_ok=True)
    request = Request(REID_URL, headers={"User-Agent": "Dance-Focus/0.4"})
    downloaded = 0
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", REID_SIZE))
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(min(99, round(downloaded * 100 / max(total, 1))))
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if _sha256(partial) != REID_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("OSNet ReID 模型校验失败")
    partial.replace(checkpoint)
    marker.write_text(REID_SHA256, encoding="ascii")
    if progress:
        progress(100)
    return checkpoint


def _get_model():
    global _MODEL, _DEVICE
    import torch

    with _LOCK:
        if _MODEL is None:
            checkpoint = ensure_reid_checkpoint()
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = osnet_ain_x1_0(num_classes=1)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
            own_state = model.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in own_state and own_state[key].shape == value.shape
            }
            missing, _ = model.load_state_dict(compatible, strict=False)
            missing_features = [key for key in missing if not key.startswith("classifier.")]
            if missing_features:
                raise RuntimeError(
                    f"OSNet 权重不完整，缺少 {len(missing_features)} 个特征层"
                )
            _MODEL = model.eval().requires_grad_(False).to(_DEVICE)
    return _MODEL, _DEVICE


def crop_person(frame, box: Box):
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(round(box.x))))
    top = max(0, min(height - 1, int(round(box.y))))
    right = max(left + 1, min(width, int(round(box.right))))
    bottom = max(top + 1, min(height, int(round(box.bottom))))
    return frame[top:bottom, left:right]


def embed_crops(crops_bgr: Sequence[np.ndarray]):
    import torch
    from torch.nn import functional as F

    if not crops_bgr:
        return torch.empty((0, 512))
    model, device = _get_model()
    tensors = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append((tensor - mean) / std)
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        features = model(batch).float()
    return F.normalize(features, dim=1).cpu()


class IdentityGallery:
    def __init__(self, initial_crop) -> None:
        self.features = embed_crops([initial_crop])

    def similarities(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        features = embed_crops(crops)
        if features.numel() == 0:
            return np.empty(0, dtype=np.float32)
        similarities = features @ self.features.T
        return similarities.max(dim=1).values.numpy()

    def update(self, crop, similarity: float) -> None:
        if similarity < 0.75 or len(self.features) >= 6:
            return
        import torch

        feature = embed_crops([crop])
        self.features = torch.cat((self.features, feature), dim=0)
