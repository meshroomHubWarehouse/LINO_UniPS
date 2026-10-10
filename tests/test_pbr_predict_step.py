"""LINO-UniPS PBR network (lino_pbr.pth): predict_step returns the normal and BRDF maps at the image size.

CPU only: the network inference (model_step) is replaced by a stand-in.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from src.models.Net_pbr_module import LiNo_UniPS as LiNo_UniPS_PBR

H, W = 40, 60
# crop box (rows 10:30, cols 15:45) predicted as a 16 x 16 square
ROI = np.array([H, W, 10, 30, 15, 45])
S = 16


def pbrBatch(mask):
    return {"roi": torch.tensor(ROI)[None], "mask_original": torch.tensor(mask, dtype=torch.float32)[None, None]}


def constantPrediction(channels, value):
    return torch.full((1, channels, S, S), float(value))


@pytest.fixture
def stubPbrNetwork():
    """Stand-in for LiNo_UniPS_PBR: model_step returns constant square maps (raw head values in [-1, 1])."""
    normal = torch.zeros(1, 3, S, S)
    normal[:, 2] = 2.0  # not unit: predict_step normalizes
    outputs = (normal, constantPrediction(3, 0.2), constantPrediction(1, -0.5), constantPrediction(1, 3.0))
    return SimpleNamespace(model_step=lambda batch: outputs, output_interpolation=cv2.INTER_CUBIC)


def test_pbr_predict_step_returns_maps_at_image_size(stubPbrNetwork):
    mask = np.zeros((H, W), np.float32)
    mask[12:28, 20:40] = 1.0
    maps = LiNo_UniPS_PBR.predict_step(stubPbrNetwork, pbrBatch(mask))

    assert set(maps) == {"normal", "baseColor", "roughness", "metallic"}
    assert maps["normal"].shape == (H, W, 3) and maps["baseColor"].shape == (H, W, 3)
    assert maps["roughness"].shape == (H, W, 1) and maps["metallic"].shape == (H, W, 1)
    inside = mask > 0
    np.testing.assert_allclose(maps["normal"][inside], [[0.0, 0.0, 1.0]] * inside.sum(), atol=1e-6)
    # raw values in [-1, 1] mapped to [0, 1], out of range values clipped
    np.testing.assert_allclose(maps["baseColor"][inside], 0.6, atol=1e-6)
    np.testing.assert_allclose(maps["roughness"][inside], 0.25, atol=1e-6)
    np.testing.assert_allclose(maps["metallic"][inside], 1.0, atol=1e-6)
    for values in maps.values():
        assert values.dtype == np.float32
        assert not values[~inside].any()


def test_pbr_predict_step_keeps_values_outside_the_crop_zero(stubPbrNetwork):
    mask = np.ones((H, W), np.float32)
    maps = LiNo_UniPS_PBR.predict_step(stubPbrNetwork, pbrBatch(mask))
    crop = np.zeros((H, W), bool)
    crop[10:30, 15:45] = True
    for values in maps.values():
        assert not values[~crop].any()
        assert np.abs(values[crop]).sum(axis=-1).all()
