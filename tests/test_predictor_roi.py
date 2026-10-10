"""The crop box (roi) must reach the model as exact integers, whatever the inference dtype.

On GPUs with compute capability >= 8 (and on the CPU) the Predictor runs in bfloat16: casting the roi to bf16
rounds it (1595 -> 1592, 8288 -> 8320) and shifts the predicted normals with respect to the image and the mask.
"""
import numpy as np
import pytest
import torch

import hubconf

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no CUDA device"))]


class _CaptureModel(torch.nn.Module):
    """Stand-in for the network: records the batch it receives."""

    def forward(self, data):
        self.data = data
        return data


@pytest.mark.parametrize("device", DEVICES)
def test_roi_kept_exact(monkeypatch, device):
    roi = np.array([2760, 4144, 1595, 2747, 1595, 3141])
    monkeypatch.setattr(hubconf, "load_data", lambda imgs, mask: [{"img": np.zeros((3, 4, 4, 2), np.float32), "roi": roi}])
    model = _CaptureModel()
    predictor = hubconf.Predictor(model, device=device)
    predictor.predict(None, None)
    received = model.data["roi"][0].cpu().numpy()
    assert received.tolist() == roi.tolist()
    assert not torch.is_floating_point(model.data["roi"])
    # floating inputs still use the inference dtype
    assert model.data["img"].dtype == predictor.dtype


def test_cpu_predictor_runs_in_bf16():
    """The network casts intermediate tensors to bf16 (wavelet filters, decoder inputs): float32 weights would
    fail on the CPU ("Input type (torch.FloatTensor) and weight type (CPUBFloat16Type) should be the same")."""
    model = torch.nn.Linear(2, 2)
    predictor = hubconf.Predictor(model, device="cpu")
    assert predictor.dtype == torch.bfloat16
    assert model.weight.dtype == torch.bfloat16
