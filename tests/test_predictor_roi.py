"""The crop box (roi) must reach the model as exact integers, whatever the inference dtype.

On GPUs with compute capability >= 8 the Predictor runs in bfloat16: casting the roi to bf16 rounds it
(1595 -> 1592, 8288 -> 8320) and shifts the predicted normals with respect to the image and the mask.
"""
import numpy as np
import pytest
import torch

import hubconf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Predictor requires CUDA")


class _CaptureModel(torch.nn.Module):
    """Stand-in for the network: records the batch it receives."""

    def forward(self, data):
        self.data = data
        return data


def test_roi_kept_exact(monkeypatch):
    roi = np.array([2760, 4144, 1595, 2747, 1595, 3141])
    monkeypatch.setattr(hubconf, "load_data", lambda imgs, mask: [{"img": np.zeros((3, 4, 4, 2), np.float32), "roi": roi}])
    model = _CaptureModel()
    predictor = hubconf.Predictor(model)
    predictor.predict(None, None)
    received = model.data["roi"][0].cpu().numpy()
    assert received.tolist() == roi.tolist()
    assert not torch.is_floating_point(model.data["roi"])
    # floating inputs still use the inference dtype
    assert model.data["img"].dtype == predictor.dtype
