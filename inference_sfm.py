"""
SfM-based inference for LINO_UniPS.

Reads an AliceVision SfMData JSON, runs photometric stereo per pose
using the LINO_UniPS model, and produces an output JSON mapping
poseIds to normal map paths.

The crop/uncrop is handled internally by LINO's DemoData and
predict_step — output normal maps are full-resolution.
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)


def load_sfm(sfm_path):
    """Load SfMData — try pyalicevision first (supports .sfm, .abc, .json),
    fallback to json.load.  Uses sfmDataIO.save() to a temp JSON so ALL
    fields (version, intrinsics, metadata, surveys…) are preserved."""
    try:
        from pyalicevision import sfmData as avSfmData, sfmDataIO
        import tempfile
        data = avSfmData.SfMData()
        if sfmDataIO.load(data, sfm_path, sfmDataIO.ALL):
            logger.info("Loaded SfMData via pyalicevision: %s", sfm_path)
            with tempfile.NamedTemporaryFile(suffix=".sfm", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sfmDataIO.save(data, tmp_path, sfmDataIO.ALL)
                with open(tmp_path, "r") as f:
                    return json.load(f)
            finally:
                os.unlink(tmp_path)
        logger.warning("pyalicevision failed to load %s, falling back to JSON", sfm_path)
    except ImportError:
        logger.info("pyalicevision not available, using JSON loader")
    with open(sfm_path, "r") as f:
        return json.load(f)


load_sfm_json = load_sfm  # backward compat


def group_views_by_pose(sfm_data):
    """Group views by poseId.

    Returns:
        dict mapping poseId (str) -> list of view dicts
    """
    groups = {}
    for view in sfm_data.get("views", []):
        pose_id = str(view.get("poseId", view.get("viewId")))
        groups.setdefault(pose_id, []).append(view)
    return groups


def extract_alpha_mask(views):
    """Extract mask by ANDing all alpha channels from the pose's images.

    Images with all-white alpha are skipped. The result is the intersection
    of all non-trivial alpha masks, keeping only the object area.

    Returns a PIL Image (RGB) or None.
    """
    import numpy as np
    combined = None
    count = 0
    for v in views:
        path = v.get("path", "")
        if not path or not os.path.isfile(path):
            continue
        img = Image.open(path)
        if img.mode not in ("RGBA", "LA", "PA"):
            continue
        alpha = np.array(img.split()[-1])
        # Skip all-white (trivial) alpha channels
        if alpha.min() > 250:
            continue
        mask = (alpha > 0).astype(np.uint8)
        if combined is None:
            combined = mask
        else:
            combined = combined * mask  # logical AND
        count += 1
    if combined is None:
        return None
    logger.info("Extracted alpha mask from %d images (AND)", count)
    combined_255 = (combined * 255).astype(np.uint8)
    mask_pil = Image.fromarray(combined_255)
    return Image.merge("RGB", (mask_pil, mask_pil, mask_pil))


def find_mask_for_pose(pose_id, mask_folder, view_ids=None, views=None):
    """Find a mask image for a given pose.

    Search order: {pose_id}.png, {viewId}.png, mask.png, alpha channel
    Returns a PIL Image or None.
    """
    if mask_folder and os.path.isdir(mask_folder):
        for candidate_id in [pose_id] + (view_ids or []):
            path = os.path.join(mask_folder, f"{candidate_id}.png")
            if os.path.isfile(path):
                return Image.open(path).convert("RGB")

        path = os.path.join(mask_folder, "mask.png")
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")

    # Fallback: extract from alpha channel
    if views:
        return extract_alpha_mask(views)

    return None


def load_images_for_pose(views, nb_img=-1, downscale=1):
    """Load images for one pose group.

    Args:
        views: list of view dicts sharing the same poseId
        nb_img: number of images to use (-1 = all)
        downscale: integer downscale factor

    Returns:
        list of (numpy_array_rgb, None) tuples — the format DemoData expects
    """
    image_paths = []
    for v in views:
        path = v.get("path", "")
        if path and os.path.isfile(path):
            image_paths.append(path)
        else:
            logger.warning(f"Image not found for viewId {v.get('viewId')}: {path}")

    if not image_paths:
        raise RuntimeError(f"No valid images for pose {views[0].get('poseId')}")

    # Select subset if requested
    if nb_img > 0 and nb_img < len(image_paths):
        indices = np.random.choice(len(image_paths), nb_img, replace=False)
        image_paths = [image_paths[i] for i in sorted(indices)]

    imgs_list = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB"))

        if downscale > 1:
            h, w = img.shape[:2]
            new_h, new_w = h // downscale, w // downscale
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        imgs_list.append((img, None))

    return imgs_list


def run_sfm_inference(sfm_path, output_folder, mask_folder=None,
                      mask_output_folder=None,
                      nb_img=-1, downscale=1, use_cuda=True,
                      task_name="Real", weights_path=None, seed=42):
    """Run LINO_UniPS inference on all poses in an SfM JSON file.

    Args:
        sfm_path: path to input SfMData JSON
        output_folder: where to write normal maps and output JSON
        mask_folder: optional folder with masks (named by poseId or viewId)
        nb_img: number of images per pose (-1 = all)
        downscale: integer downscale factor
        use_cuda: use GPU
        task_name: task name for model config (e.g. "Real", "DiLiGenT")
        weights_path: path to local .pth weights file (None = download)
        seed: random seed

    Returns:
        Path to the output JSON file.
    """
    os.makedirs(output_folder, exist_ok=True)
    seed_everything(seed=seed, workers=True)

    # Load SfM data
    sfm_data = load_sfm(sfm_path)
    pose_groups = group_views_by_pose(sfm_data)
    logger.info(f"Loaded {len(sfm_data.get('views', []))} views, "
                f"{len(pose_groups)} poses")

    # Load model
    logger.info("Loading LINO_UniPS model...")
    from hubconf import LINO, lino_unips

    if weights_path and os.path.isfile(weights_path):
        predictor = LINO(local_file_path=weights_path, task_name=task_name)
    else:
        model = lino_unips(pretrained=True, task_name=task_name)
        predictor_cls = type('Predictor', (), {
            '__init__': lambda self, m: setattr(self, 'model', m),
            'predict': lambda self, imgs, mask: self.model(
                self._make_batch(imgs, mask)),
        })
        # Use the LINO Predictor class directly
        predictor = LINO(local_file_path=None, task_name=task_name)

    logger.info("Model loaded")

    # Process each pose
    results = []
    total_start = time.time()

    for pose_id, views in pose_groups.items():
        logger.info(f"=== Pose {pose_id} ({len(views)} views) ===")
        pose_start = time.time()

        try:
            # Load images as list of (np_array, None) for DemoData
            imgs_list = load_images_for_pose(views, nb_img, downscale)

            # Load mask
            view_ids = [str(v["viewId"]) for v in views]
            mask_img = find_mask_for_pose(pose_id, mask_folder, view_ids, views=views)

            # Save extracted mask if output folder is set
            if mask_img is not None and mask_output_folder and not mask_folder:
                os.makedirs(mask_output_folder, exist_ok=True)
                mask_path = os.path.join(mask_output_folder, f"{pose_id}.png")
                mask_img.save(mask_path)
                logger.info("Saved mask to %s", mask_path)

            # Resize mask to match (downscaled) image dimensions
            if mask_img is not None and imgs_list:
                img_h, img_w = imgs_list[0][0].shape[:2]
                mask_w, mask_h = mask_img.size  # PIL: (width, height)
                if mask_h != img_h or mask_w != img_w:
                    logger.info("Resizing mask from %dx%d to %dx%d "
                                "to match images",
                                mask_w, mask_h, img_w, img_h)
                    mask_img = mask_img.resize(
                        (img_w, img_h), Image.NEAREST)

            # Run prediction via the Predictor API
            result = predictor.predict(imgs_list, mask_img)

            # result is the normal map (H, W, 3) in [-1, 1]
            if isinstance(result, torch.Tensor):
                normal = result.cpu().numpy()
            else:
                normal = np.array(result)

            # Ensure shape is (H, W, 3)
            if normal.ndim == 4:
                normal = normal.squeeze(0)
            if normal.shape[0] == 3 and normal.ndim == 3:
                normal = np.transpose(normal, (1, 2, 0))

            # Save as 16-bit PNG
            normal_rgb = (((normal + 1) / 2) * 65535).astype(np.uint16)
            out_path = os.path.join(output_folder, f"{pose_id}.png")
            cv2.imwrite(out_path, normal_rgb[:, :, ::-1])

            pose_time = time.time() - pose_start
            logger.info(f"Pose {pose_id}: {normal.shape}, "
                        f"saved to {out_path} ({pose_time:.1f}s)")

            # Find representative view
            rep_view = next(
                (v for v in views if str(v.get("viewId")) == str(pose_id)),
                views[0])

            results.append({
                "poseId": pose_id,
                "viewId": str(rep_view.get("viewId")),
                "normalMapPath": os.path.abspath(out_path),
                "width": normal.shape[1],
                "height": normal.shape[0],
                "nbImages": len(views),
            })

        except Exception as e:
            logger.error(f"Failed on pose {pose_id}: {e}", exc_info=True)
            continue

    total_time = time.time() - total_start
    logger.info(f"All poses processed in {total_time:.1f}s")

    logger.info(f"Inference complete: {len(results)} poses processed")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LINO_UniPS inference from SfM JSON"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input SfMData JSON file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output folder for normal maps")
    parser.add_argument("--masks", "-m", default=None,
                        help="Folder with mask PNGs (named by poseId/viewId)")
    parser.add_argument("--nb-img", type=int, default=-1,
                        help="Number of images per pose (-1 = all)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Integer downscale factor (1 = no downscale)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use GPU")
    parser.add_argument("--task-name", default="Real",
                        help="Task name for model config")
    parser.add_argument("--weights", default=None,
                        help="Path to local .pth weights file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_sfm_inference(
        sfm_path=args.input,
        output_folder=args.output,
        mask_folder=args.masks,
        nb_img=args.nb_img,
        downscale=args.downscale,
        use_cuda=args.cuda,
        task_name=args.task_name,
        weights_path=args.weights,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
