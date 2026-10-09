"""Meshroom API of LINO-UniPS: normal map of one multi-lighting pose.

Used by the mrLINOUniPS Meshroom plugin, whose common layer (psCommon.py) handles everything else (SfMData,
image selection and loading, masks, outputs). Contract shared by the photometric stereo plugins:

    predictor = loadModel(weightsPath, useGpu)
    maps = predict(predictor, images, mask, **options)

- images: list of float32 RGB arrays (H x W x 3), values as stored in the images (typically [0, 1]),
- mask: bool array (H x W), the pixels of the object,
- maps["normal"]: float32 (H x W x 3) unit normals in the OpenGL camera frame (x right, y up, z towards the
  camera), zero outside the mask and where undefined.
"""
import cv2
import numpy as np
import torch

INTERPOLATIONS = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def loadModel(weightsPath, useGpu=True, logger=None):
    """Load the LINO-UniPS network (bf16/fp16 on GPU, float32 on CPU).

    Raises if the checkpoint does not provide every weight of the network.
    """
    from hubconf import Predictor, _load_state_dict
    from src.models.Net_module import LiNo_UniPS

    model = LiNo_UniPS(task_name="Real")
    keys = model.load_state_dict(_load_state_dict(weightsPath), strict=False)
    if keys.missing_keys:
        raise RuntimeError("Invalid LINO-UniPS checkpoint '{}': {} missing weights (e.g. {})".format(
            weightsPath, len(keys.missing_keys), keys.missing_keys[0]))
    model.eval()
    device = "cuda" if useGpu and torch.cuda.is_available() else "cpu"
    if logger:
        if useGpu and device == "cpu":
            logger.warning("No GPU available: running LINO-UniPS on the CPU (slow).")
        if keys.unexpected_keys:
            logger.debug("Unused checkpoint entries: {}".format(keys.unexpected_keys))
    return Predictor(model, device=device)


def predict(predictor, images, mask, cropMargin=8, maxProcessingSize=6000, outputInterpolation="cubic"):
    """Normal map of one pose.

    Args:
        cropMargin: margin (pixels) around the bounding box of the mask; the whole image is used when the box
            is closer than this margin to the image border.
        maxProcessingSize: maximum side of the square network input. The crop is resized to
            max(512, min(maxProcessingSize, floor(crop side / 512) * 512)).
        outputInterpolation: resampling of the prediction back to the crop size ("area", "linear", "cubic").
    """
    if outputInterpolation not in INTERPOLATIONS:
        raise ValueError("Unknown interpolation '{}'".format(outputInterpolation))
    height, width = mask.shape
    inputs = [(np.asarray(image, np.float32), None) for image in images]
    # DemoData expects a 3-channel mask; values in [0, 1] are used as is
    inputMask = np.repeat(mask.astype(np.float32)[:, :, None], 3, axis=2)
    predictor.model.output_interpolation = INTERPOLATIONS[outputInterpolation]
    with torch.no_grad():
        normal = predictor.predict(inputs, inputMask, margin=cropMargin, max_image_resolution=maxProcessingSize)
    normal = np.asarray(normal, np.float32)
    if normal.shape != (height, width, 3):
        raise RuntimeError("Unexpected LINO-UniPS output shape {} (expected {})".format(normal.shape, (height, width, 3)))
    return {"normal": normal}
