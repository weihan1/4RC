import argparse
from pathlib import Path

import numpy as np
import torch

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import rgb, _resize_pil_image
from arc.models.arc.arc import Arc
import tqdm
import cv2
import os


DEFAULT_AIM_ROOT = Path(

    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_full"
)
DEFAULT_AIM_RESULTS_ROOT = Path(
    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_results"
)
RGB_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif")
FRAME_OUTPUT_FILES = (
    "pts3d.npy",
    "depth_conf.npy",
    "c2w.npy",
    "K.npy",
    "rgb_processed.png",
)

def select_rgb_frames(rgb_frames, max_rgb_frames: int):
    if max_rgb_frames <= 0:
        raise ValueError(f"max_rgb_frames must be positive, got {max_rgb_frames}")
    if len(rgb_frames) <= max_rgb_frames:
        return rgb_frames

    indices = np.linspace(0, len(rgb_frames) - 1, max_rgb_frames, dtype=int)
    return [rgb_frames[i] for i in indices]

def list_aim_rgb_frames(sequence_dir: Path):
    return sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in RGB_SUFFIXES
        and "_mask" in path.stem
    )

def load_images(folder_or_list, size, square_ok=False, verbose=True, rotate_clockwise_90=False, crop_to_landscape=False, patch_size=16):
    import PIL.Image
    import torchvision.transforms as tvf
    from PIL.ImageOps import exif_transpose

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        heif_support_enabled = True
    except ImportError:
        heif_support_enabled = False
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")


        W1, H1 = img.size
        if size <= 392:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size <= 392:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            # 16 is the patch size and 8 is the 16//2
            halfw, halfh = ((2 * cx) // patch_size) * patch_size//2, ((2 * cy) // patch_size) * patch_size//2
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        # true_shape = [img.size] if height > width else [img.size[::-1]] # if is protrait, the true shape should inverse
        true_shape = [img.size[::-1]] # true shape requires H, W
        imgs.append(img)

    assert imgs, "no images found at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs

def load_sequence_images(sequence_dir: Path, rgb_frames=None):
    if rgb_frames is None:
        rgb_frames = list_aim_rgb_frames(sequence_dir)
    if not rgb_frames:
        raise ValueError(f"No AiM RGB frames found in {sequence_dir}")

    images = load_images(
        [str(path) for path in rgb_frames],
        size=512,
        patch_size=14,
        verbose=True,
        square_ok=True #force square inputs
    )
    return images, rgb_frames

def sequence_outputs_exist(frame_paths, output_dir: Path):
    return all(frame_outputs_exist(output_dir, frame_path) for frame_path in frame_paths)

def run_sequence(
    animal: str,
    sequence_dir: Path,
    results_root: Path,
    max_rgb_frames: int = 100,
    skip_existing: bool = False,
):
    frame_paths = list_aim_rgb_frames(sequence_dir)
    if not frame_paths:
        raise ValueError(f"No AiM RGB frames found in {sequence_dir}")

    frame_paths = select_rgb_frames(frame_paths, max_rgb_frames)

    output_dir = results_root / animal / sequence_dir.name
    if skip_existing and sequence_outputs_exist(frame_paths, output_dir):
        print("sequence exists already, skipping")
        return None

    images, frame_paths = load_sequence_images(sequence_dir, rgb_frames=frame_paths)

def parse_args():
    parser = argparse.ArgumentParser(description="Run 4RC demo on AiM sequences.")
    parser.add_argument(
        "--animal",
        type=str,
        required=True,
        help="AiM animal category, for example 'wolf'.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="AiM sequence name. If omitted, run all sequences for the animal.",
    )
    parser.add_argument(
        "--aim-root",
        type=Path,
        default=DEFAULT_AIM_ROOT,
        help="Root directory of the AiM_full dataset.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_AIM_RESULTS_ROOT,
        help="Root directory where AiM sequence predictions will be saved.",
    )
    parser.add_argument(
        "--max-rgb-frames",
        type=int,
        default=100,
        help="Maximum number of RGB frames to run inference on per sequence.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Do not overwrite existing per-frame 4RC outputs.",
    )
    return parser.parse_args()

def resolve_sequence_dirs(aim_root: Path, animal: str, sequence: str | None):
    animal_dir = aim_root / animal
    if not animal_dir.exists():
        raise FileNotFoundError(f"Animal directory does not exist: {animal_dir}")
    if not animal_dir.is_dir():
        raise NotADirectoryError(f"Animal path is not a directory: {animal_dir}")

    if sequence is not None:
        sequence_dir = animal_dir / sequence
        if not sequence_dir.exists():
            raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
        if not sequence_dir.is_dir():
            raise NotADirectoryError(f"Sequence path is not a directory: {sequence_dir}")
        return [sequence_dir]

    return sorted(path for path in animal_dir.iterdir() if path.is_dir())

def main():
    args = parse_args()
    sequence_dirs = resolve_sequence_dirs(args.aim_root, args.animal, args.sequence)

    for sequence_dir in tqdm.tqdm(sequence_dirs, desc="Sequences"):
        try:
            run_sequence(
                args.animal,
                sequence_dir,
                args.results_root,
                max_rgb_frames=args.max_rgb_frames,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()


    