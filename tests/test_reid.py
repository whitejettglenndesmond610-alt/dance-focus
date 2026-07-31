import torch

from dance_focus.vendor.osnet_ain import osnet_ain_x1_0


def test_vendored_osnet_produces_512_dimension_embeddings():
    model = osnet_ain_x1_0().eval()

    with torch.inference_mode():
        features = model(torch.zeros((1, 3, 256, 128)))

    assert features.shape == (1, 512)
