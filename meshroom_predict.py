"""Meshroom API of LINO-UniPS: normal map (and PBR maps) of one multi-lighting pose.

Used by the mrLINOUniPS Meshroom plugin, whose common layer (psCommon.py) handles everything else (SfMData,
image selection and loading, masks, outputs). Contract shared by the photometric stereo plugins:

    predictor = loadModel(weightsPath, useGpu, model="normal" | "pbr")
    maps = predict(predictor, images, mask, **options)

- images: list of float32 RGB arrays (H x W x 3), values as stored in the images (typically [0, 1]),
- mask: bool array (H x W), the pixels of the object,
- maps["normal"]: float32 (H x W x 3) unit normals in the OpenGL camera frame (x right, y up, z towards the
  camera), zero outside the mask and where undefined.

With the PBR model (lino_pbr.pth), the same network pass also gives:
- maps["albedo"]: float32 (H x W x 3) base color in [0, 1], zero outside the mask,
- maps["roughness"], maps["metallic"]: float32 (H x W) in [0, 1], zero outside the mask.
"""
import cv2
import numpy as np
import torch

INTERPOLATIONS = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
MODELS = ("normal", "pbr")
# Weights of the PBR heads only: a normal-only checkpoint does not have them
PBR_HEADS = ("regressor.prediction_base.", "regressor.prediction_rough.", "regressor.prediction_metal.")


def createNetwork(model):
    """Untrained network: LINO-UniPS ("normal") or LINO-UniPS PBR ("pbr": normal, base color, roughness and
    metallic heads on the same features)."""
    if model == "normal":
        from src.models.Net_module import LiNo_UniPS
        return LiNo_UniPS(task_name="Real")
    if model == "pbr":
        from src.models.Net_pbr_module import LiNo_UniPS as LiNo_UniPS_PBR
        return LiNo_UniPS_PBR(task_name="Real", brdf=True)
    raise ValueError("Unknown LINO-UniPS model '{}' (expected one of {})".format(model, ", ".join(MODELS)))


def loadModel(weightsPath, useGpu=True, logger=None, model="normal"):
    """Load the LINO-UniPS network (bf16 on CPU and on GPUs of compute capability >= 8, fp16 on older GPUs).

    Args:
        weightsPath: checkpoint (.pth); None to download it (lino.pth or lino_pbr.pth, see hubconf).
        model: "normal" (lino.pth: normals) or "pbr" (lino_pbr.pth: normals and PBR maps).

    Raises if the checkpoint does not provide every weight of the network (e.g. lino.pth with model="pbr").
    """
    from hubconf import DEFAULT_PBR_MODEL_URL, Predictor, _load_state_dict

    network = createNetwork(model)
    stateDict = _load_state_dict(weightsPath, url=DEFAULT_PBR_MODEL_URL if model == "pbr" else None)
    keys = network.load_state_dict(stateDict, strict=False)
    if keys.missing_keys:
        hint = " (not a PBR checkpoint: use lino_pbr.pth)" if model == "pbr" and all(
            k.startswith(PBR_HEADS) for k in keys.missing_keys) else ""
        raise RuntimeError("Invalid LINO-UniPS {} checkpoint '{}': {} missing weights (e.g. {}){}".format(
            model, weightsPath, len(keys.missing_keys), keys.missing_keys[0], hint))
    network.eval()
    device = "cuda" if useGpu and torch.cuda.is_available() else "cpu"
    if logger:
        if useGpu and device == "cpu":
            logger.warning("No GPU available: running LINO-UniPS on the CPU (slow).")
        if model == "normal" and any(k.startswith(PBR_HEADS) for k in keys.unexpected_keys):
            logger.warning("PBR checkpoint used as a normal-only model: normals of the PBR network.")
        if keys.unexpected_keys:
            logger.debug("Unused checkpoint entries: {}".format(keys.unexpected_keys))
    return Predictor(network, device=device)


def checkMap(name, values, shape):
    values = np.asarray(values, np.float32)
    if values.ndim == 3 and len(shape) == 2 and values.shape[2] == 1:
        values = values[:, :, 0]
    if values.shape != shape:
        raise RuntimeError("Unexpected LINO-UniPS {} shape {} (expected {})".format(name, values.shape, shape))
    return values


def predict(predictor, images, mask, cropMargin=8, maxProcessingSize=6000, outputInterpolation="cubic"):
    """Normal map (and, with the PBR model, albedo, roughness and metallic maps) of one pose.

    Args:
        cropMargin: margin (pixels) around the bounding box of the mask; the whole image is used when the box
            is closer than this margin to the image border.
        maxProcessingSize: maximum side of the square network input. The crop is resized to
            max(512, min(maxProcessingSize, floor(crop side / 512) * 512)).
        outputInterpolation: resampling of the prediction back to the crop size ("area", "linear", "cubic").

    Returns:
        dict of float32 maps at the input resolution: "normal", plus "albedo", "roughness" and "metallic" with
        the PBR model (see the module documentation).
    """
    if outputInterpolation not in INTERPOLATIONS:
        raise ValueError("Unknown interpolation '{}'".format(outputInterpolation))
    height, width = mask.shape
    inputs = [(np.asarray(image, np.float32), None) for image in images]
    # DemoData expects a 3-channel mask; values in [0, 1] are used as is
    inputMask = np.repeat(mask.astype(np.float32)[:, :, None], 3, axis=2)
    predictor.model.output_interpolation = INTERPOLATIONS[outputInterpolation]
    with torch.no_grad():
        output = predictor.predict(inputs, inputMask, margin=cropMargin, max_image_resolution=maxProcessingSize)
    # the normal-only network returns the normal map, the PBR network a dict of maps
    if not isinstance(output, dict):
        return {"normal": checkMap("normal", output, (height, width, 3))}
    return {
        "normal": checkMap("normal", output["normal"], (height, width, 3)),
        "albedo": checkMap("base color", output["baseColor"], (height, width, 3)),
        "roughness": checkMap("roughness", output["roughness"], (height, width)),
        "metallic": checkMap("metallic", output["metallic"], (height, width)),
    }
