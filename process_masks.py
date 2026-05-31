import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import rgb
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
MASK_OUTPUT_FILE = "mask_processed.png"

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

def _resize_mask_pil_image(mask, long_edge_size):
    scale = long_edge_size / max(mask.size)
    new_size = tuple(int(round(dim * scale)) for dim in mask.size)
    return mask.resize(new_size, PIL.Image.NEAREST)


def _load_single_channel_mask(path):
    from PIL.ImageOps import exif_transpose

    mask = exif_transpose(PIL.Image.open(path))
    bands = mask.getbands()

    if mask.mode in {"1", "L", "I", "I;16", "P"}:
        return mask.copy()

    if "A" in bands:
        alpha = np.asarray(mask.getchannel("A"))
        if np.any(alpha != alpha.flat[0]):
            return PIL.Image.fromarray(alpha)

    mask_np = np.asarray(mask)
    if mask_np.ndim == 2:
        return PIL.Image.fromarray(mask_np)
    if mask_np.ndim == 3 and mask_np.shape[2] == 1:
        return PIL.Image.fromarray(mask_np[..., 0])
    if mask_np.ndim == 3 and mask_np.shape[2] >= 3:
        if np.array_equal(mask_np[..., 0], mask_np[..., 1]) and np.array_equal(mask_np[..., 0], mask_np[..., 2]):
            return PIL.Image.fromarray(mask_np[..., 0])
        return mask.convert("L")

    raise ValueError(f"Unsupported mask shape for {path}: {mask_np.shape}")


def _crop_like_dust3r(img, size, square_ok=False, patch_size=16):
    W, H = img.size
    cx, cy = W // 2, H // 2

    if size <= 392:
        half = min(cx, cy)
        return img.crop((cx - half, cy - half, cx + half, cy + half))

    halfw = ((2 * cx) // patch_size) * patch_size // 2
    halfh = ((2 * cy) // patch_size) * patch_size // 2
    if not square_ok and W == H:
        halfh = 3 * halfw / 4
    return img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))


def load_masks(folder_or_list, size, square_ok=False, verbose=True, patch_size=16):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        heif_support_enabled = True
    except ImportError:
        heif_support_enabled = False
    """Open masks and apply the same resize/crop geometry as DUSt3R RGB inputs."""
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
        img = _load_single_channel_mask(os.path.join(root, path))

        W1, H1 = img.size
        if size <= 392:
            # resize short side to 224 (then crop)
            img = _resize_mask_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_mask_pil_image(img, size)
        img = _crop_like_dust3r(img, size=size, square_ok=square_ok, patch_size=patch_size)

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
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

    images = load_masks(
        [str(path) for path in rgb_frames],
        size=512,
        patch_size=14,
        verbose=True,
        square_ok=True #force square inputs
    )
    return images, rgb_frames

def frame_timestep(frame_path: Path):
    return frame_path.stem.removesuffix("_mask")


def frame_output_dir(output_dir: Path, frame_path: Path):
    return output_dir / frame_timestep(frame_path) / "4rc"


def frame_outputs_exist(output_dir: Path, frame_path: Path):
    return (frame_output_dir(output_dir, frame_path) / MASK_OUTPUT_FILE).exists()


def sequence_outputs_exist(frame_paths, output_dir: Path):
    return all(frame_outputs_exist(output_dir, frame_path) for frame_path in frame_paths)


def save_processed_masks(masks, frame_paths, output_dir: Path):
    if len(masks) != len(frame_paths):
        raise ValueError(f"Got {len(masks)} masks for {len(frame_paths)} frame paths")

    for mask, frame_path in zip(masks, frame_paths):
        save_dir = frame_output_dir(output_dir, frame_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        mask.save(save_dir / MASK_OUTPUT_FILE)

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
    save_processed_masks(images, frame_paths, output_dir)


def _run_sequence_task(
    animal: str,
    sequence_dir: Path,
    results_root: Path,
    max_rgb_frames: int,
    skip_existing: bool,
):
    try:
        run_sequence(
            animal,
            sequence_dir,
            results_root,
            max_rgb_frames=max_rgb_frames,
            skip_existing=skip_existing,
        )
        return sequence_dir, None
    except Exception as exc:
        return sequence_dir, exc

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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of sequences to process concurrently.",
    )
    return parser.parse_args()

def resolve_sequence_dirs(
    aim_root: Path,
    results_root: Path,
    animal: str,
    sequence: str | None,
):
    animal_dir = aim_root / animal
    if not animal_dir.exists():
        raise FileNotFoundError(f"Animal directory does not exist: {animal_dir}")
    if not animal_dir.is_dir():
        raise NotADirectoryError(f"Animal path is not a directory: {animal_dir}")

    results_animal_dir = results_root / animal
    if not results_animal_dir.exists():
        return []
    if not results_animal_dir.is_dir():
        raise NotADirectoryError(
            f"Animal results path is not a directory: {results_animal_dir}"
        )

    if sequence is not None:
        result_sequence_dir = results_animal_dir / sequence
        if not result_sequence_dir.exists():
            raise FileNotFoundError(
                f"Sequence results directory does not exist: {result_sequence_dir}"
            )
        if not result_sequence_dir.is_dir():
            raise NotADirectoryError(
                f"Sequence results path is not a directory: {result_sequence_dir}"
            )

        sequence_dir = animal_dir / sequence
        if not sequence_dir.exists():
            raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
        if not sequence_dir.is_dir():
            raise NotADirectoryError(f"Sequence path is not a directory: {sequence_dir}")
        return [sequence_dir]

    return sorted(
        animal_dir / path.name
        for path in results_animal_dir.iterdir()
        if path.is_dir() and (animal_dir / path.name).is_dir()
    )

def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")

    sequence_dirs = resolve_sequence_dirs(
        args.aim_root,
        args.results_root,
        args.animal,
        args.sequence,
    )

    if args.workers == 1 or len(sequence_dirs) <= 1:
        for sequence_dir in tqdm.tqdm(sequence_dirs, desc="Sequences"):
            _, error = _run_sequence_task(
                args.animal,
                sequence_dir,
                args.results_root,
                args.max_rgb_frames,
                args.skip_existing,
            )
            if error is not None:
                print(f"{sequence_dir.name}: {error}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_sequence_task,
                args.animal,
                sequence_dir,
                args.results_root,
                args.max_rgb_frames,
                args.skip_existing,
            )
            for sequence_dir in sequence_dirs
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Sequences"):
            sequence_dir, error = future.result()
            if error is not None:
                print(f"{sequence_dir.name}: {error}")


if __name__ == "__main__":
    main()


    
