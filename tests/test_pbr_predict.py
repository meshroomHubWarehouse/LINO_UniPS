"""Meshroom predict API with the PBR model (lino_pbr.pth): normals and albedo from one network pass.

CPU only: the network inference is replaced by stand-ins, the checkpoints by state dicts of untrained networks.
"""
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import hubconf
import meshroom_predict
from src.models.Net_pbr_module import LiNo_UniPS as LiNo_UniPS_PBR

H, W = 40, 60


class StubPredictor:
    """Stand-in for hubconf.Predictor: returns fixed network outputs and records the data options."""

    def __init__(self, output):
        self.model = SimpleNamespace()
        self.output = output

    def predict(self, inputs, mask, **options):
        self.options = options
        return self.output


def poseInputs():
    images = [np.random.default_rng(i).random((H, W, 3), dtype=np.float32) for i in range(3)]
    mask = np.zeros((H, W), bool)
    mask[5:35, 10:50] = True
    return images, mask


def test_predict_with_normal_model_returns_normal_only():
    normal = np.zeros((H, W, 3), np.float32)
    normal[..., 2] = 1.0
    predictor = StubPredictor(normal)
    maps = meshroom_predict.predict(predictor, *poseInputs(), cropMargin=4, maxProcessingSize=1024,
                                    outputInterpolation="area")
    assert set(maps) == {"normal"}
    np.testing.assert_array_equal(maps["normal"], normal)
    assert predictor.options == {"margin": 4, "max_image_resolution": 1024}
    assert predictor.model.output_interpolation == cv2.INTER_AREA


def test_predict_with_pbr_model_returns_albedo_roughness_metallic():
    output = {
        "normal": np.zeros((H, W, 3), np.float32),
        "baseColor": np.full((H, W, 3), 0.3, np.float32),
        "roughness": np.full((H, W, 1), 0.4, np.float32),
        "metallic": np.full((H, W, 1), 0.1, np.float32),
    }
    maps = meshroom_predict.predict(StubPredictor(output), *poseInputs())
    assert set(maps) == {"normal", "albedo", "roughness", "metallic"}
    assert maps["albedo"].shape == (H, W, 3)
    assert maps["roughness"].shape == (H, W) and maps["metallic"].shape == (H, W)
    np.testing.assert_allclose(maps["albedo"], 0.3)
    np.testing.assert_allclose(maps["roughness"], 0.4)
    assert all(values.dtype == np.float32 for values in maps.values())


def test_predict_rejects_maps_of_another_size():
    output = {"normal": np.zeros((H, W, 3), np.float32), "baseColor": np.zeros((H + 1, W, 3), np.float32),
              "roughness": np.zeros((H, W, 1), np.float32), "metallic": np.zeros((H, W, 1), np.float32)}
    with pytest.raises(RuntimeError, match="base color"):
        meshroom_predict.predict(StubPredictor(output), *poseInputs())


# --- loadModel ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoints():
    """State dicts of untrained networks, standing for lino.pth and lino_pbr.pth."""
    torch.manual_seed(0)
    return {name: meshroom_predict.createNetwork(name).state_dict() for name in meshroom_predict.MODELS}


@pytest.fixture
def fakeLoad(monkeypatch, checkpoints):
    """Replace the checkpoint loading: weightsPath "lino.pth" / "lino_pbr.pth"; records the download URL."""
    calls = []

    def load(path, url=None):
        calls.append((path, url))
        return checkpoints["pbr" if path is None or "pbr" in path else "normal"]

    monkeypatch.setattr(hubconf, "_load_state_dict", load)
    return calls


def test_loadModel_pbr(fakeLoad, checkpoints):
    predictor = meshroom_predict.loadModel("lino_pbr.pth", useGpu=False, model="pbr")
    assert isinstance(predictor.model, LiNo_UniPS_PBR) and predictor.model.brdf
    assert predictor.device.type == "cpu" and predictor.dtype == torch.bfloat16
    weight = "regressor.prediction_base.out_layer.weight"
    torch.testing.assert_close(predictor.model.state_dict()[weight].float(),
                               checkpoints["pbr"][weight].to(torch.bfloat16).float())


def test_loadModel_pbr_rejects_a_normal_only_checkpoint(fakeLoad):
    with pytest.raises(RuntimeError, match="not a PBR checkpoint"):
        meshroom_predict.loadModel("lino.pth", useGpu=False, model="pbr")


def test_loadModel_downloads_the_checkpoint_of_the_model(fakeLoad):
    meshroom_predict.loadModel(None, useGpu=False, model="pbr")
    assert fakeLoad[-1] == (None, hubconf.DEFAULT_PBR_MODEL_URL)
    meshroom_predict.loadModel("lino.pth", useGpu=False, model="normal")
    assert fakeLoad[-1] == ("lino.pth", None)


def test_loadModel_warns_when_a_pbr_checkpoint_is_used_as_normal_model(fakeLoad, caplog):
    logger = logging.getLogger("test_pbr_predict")
    with caplog.at_level(logging.WARNING, logger="test_pbr_predict"):
        predictor = meshroom_predict.loadModel("lino_pbr.pth", useGpu=False, logger=logger, model="normal")
    assert not getattr(predictor.model, "brdf", False)
    assert "PBR checkpoint" in caplog.text


def test_loadModel_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="Unknown LINO-UniPS model"):
        meshroom_predict.loadModel("lino.pth", useGpu=False, model="brdf")


def test_default_pbr_checkpoint_url():
    assert hubconf.DEFAULT_PBR_MODEL_URL.endswith("/houyuanchen/lino/resolve/main/lino_pbr.pth")
