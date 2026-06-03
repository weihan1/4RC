import argparse
from pathlib import Path

import numpy as np
import torch
import PIL.Image
from PIL.ImageOps import exif_transpose

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import _resize_pil_image, load_images_for_eval, rgb
from arc.models.arc.arc import Arc
import tqdm
import cv2


DEFAULT_AIM_ROOT = Path(
    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_full"
)
DEFAULT_AIM_RESULTS_ROOT = Path(
    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_results"
)
RGB_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif")
MODEL_INPUT_SIZE = 512
MODEL_PATCH_SIZE = 14
MODEL_SQUARE_OK = False
FRAME_OUTPUT_FILES = (
    "pts3d_processed.npy",
    "depth.npy",
    "depth_processed.npy",
    "depth_conf.npy",
    "depth_conf_processed.npy",
    "c2w.npy",
    "K.npy",
    "K_processed.npy",
    "rgb_processed.png",
)

def world_to_camera(pts3d: np.ndarray, w2c: np.ndarray):
    if pts3d.ndim != 3 or pts3d.shape[-1] != 3:
        raise ValueError(f"Expected pts3d with shape [H, W, 3], got {pts3d.shape}")
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected c2w with shape [4, 4], got {w2c.shape}")

    ones = np.ones((*pts3d.shape[:2], 1), dtype=pts3d.dtype)
    pts3d_h = np.concatenate([pts3d, ones], axis=-1)
    cam_points_h = pts3d_h @ w2c.T
    return cam_points_h[..., :3]

def camera_points_to_depth(cam_points: np.ndarray, K: np.ndarray):
    if K.shape != (3, 3):
        raise ValueError(f"Expected K with shape [3, 3], got {K.shape}")

    projected = cam_points @ K.T
    return projected[..., 2]


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


def list_aim_rgb_frames(sequence_dir: Path):
    return sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in RGB_SUFFIXES
        and "_rgb_original" in path.stem
    )

def _scale_k_between_resolutions(
    k: np.ndarray, w_from: int, h_from: int, w_to: int, h_to: int
) -> np.ndarray:
    sx = float(w_to) / float(w_from) if w_from else 1.0
    sy = float(h_to) / float(h_from) if h_from else 1.0
    out = np.asarray(k, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out.astype(np.float32)


def _get_image_preprocess_geometry(
    frame_path: Path,
    size: int = MODEL_INPUT_SIZE,
    patch_size: int = MODEL_PATCH_SIZE,
    square_ok: bool = MODEL_SQUARE_OK,
):
    with PIL.Image.open(frame_path) as pil_img:
        img = exif_transpose(pil_img).convert("RGB")
        orig_width, orig_height = img.size

        if size == 256:
            resized = _resize_pil_image(
                img, round(size * max(orig_width / orig_height, orig_height / orig_width))
            )
        else:
            resized = _resize_pil_image(img, size)

        resized_width, resized_height = resized.size
        cx, cy = resized_width // 2, resized_height // 2
        if size == 256:
            half = min(cx, cy)
            processed_size = (2 * half, 2 * half)
        else:
            halfw = ((2 * cx) // patch_size) * (patch_size // 2)
            halfh = ((2 * cy) // patch_size) * (patch_size // 2)
            if not square_ok and resized_width == resized_height:
                halfh = int(3 * halfw / 4)
            processed_size = (2 * halfw, 2 * halfh)

    processed_width, processed_height = processed_size
    return {
        "orig_width": orig_width,
        "orig_height": orig_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
    }


def _map_processed_intrinsics_to_original_frame(
    k_processed: np.ndarray,
    frame_path: Path | None = None,
    geometry: dict[str, int] | None = None,
) -> np.ndarray:
    if geometry is None:
        if frame_path is None:
            raise ValueError("Either frame_path or geometry must be provided")
        geometry = _get_image_preprocess_geometry(frame_path)

    return _scale_k_between_resolutions(
        np.asarray(k_processed, dtype=np.float64),
        w_from=geometry["processed_width"],
        h_from=geometry["processed_height"],
        w_to=geometry["orig_width"],
        h_to=geometry["orig_height"],
    )

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


def select_rgb_frames(rgb_frames, max_rgb_frames: int):
    if max_rgb_frames <= 0:
        raise ValueError(f"max_rgb_frames must be positive, got {max_rgb_frames}")
    if len(rgb_frames) <= max_rgb_frames:
        return rgb_frames

    indices = np.linspace(0, len(rgb_frames) - 1, max_rgb_frames, dtype=int)
    return [rgb_frames[i] for i in indices]


def load_sequence_images(sequence_dir: Path, rgb_frames=None):
    if rgb_frames is None:
        rgb_frames = list_aim_rgb_frames(sequence_dir)
    if not rgb_frames:
        raise ValueError(f"No AiM RGB frames found in {sequence_dir}")

    images = load_images_for_eval(
        [str(path) for path in rgb_frames],
        size=MODEL_INPUT_SIZE,
        patch_size=MODEL_PATCH_SIZE,
        verbose=True,
        square_ok=MODEL_SQUARE_OK,
        crop=False,
    )
    return images, rgb_frames


def to_numpy_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_batch_dim(array: np.ndarray):
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def get_timestep_from_frame_path(frame_path: Path):
    suffix = "_rgb_original"
    stem = frame_path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected AiM RGB filename: {frame_path.name}")
    return stem[: -len(suffix)]


def get_frame_output_dir(output_dir: Path, frame_path: Path):
    timestep = get_timestep_from_frame_path(frame_path)
    return output_dir / timestep / "4rc"


def frame_outputs_exist(output_dir: Path, frame_path: Path):
    frame_output_dir = get_frame_output_dir(output_dir, frame_path)
    return all((frame_output_dir / name).exists() for name in FRAME_OUTPUT_FILES)


def sequence_outputs_exist(frame_paths, output_dir: Path):
    return all(frame_outputs_exist(output_dir, frame_path) for frame_path in frame_paths)


def to_uint8_rgb_image(view):
    true_shape = squeeze_batch_dim(to_numpy_array(view["true_shape"]))
    img = rgb(squeeze_batch_dim(to_numpy_array(view["img"])), true_shape=true_shape)
    return np.rint(img * 255.0).astype(np.uint8)


def save_sequence_outputs(predictions, frame_paths, output_dir: Path, skip_existing: bool = False):
    preds = predictions["preds"]
    views = predictions["views"]
    if len(preds) != len(frame_paths):
        raise ValueError(
            f"Prediction/frame count mismatch: {len(preds)} predictions vs "
            f"{len(frame_paths)} frames"
        )

    saved_frames = 0
    skipped_frames = 0
    for pred, view, frame_path in zip(preds, views, frame_paths):
        frame_output_dir = get_frame_output_dir(output_dir, frame_path)
        if skip_existing and frame_outputs_exist(output_dir, frame_path):
            print(f">> Skipping existing outputs for {frame_path.name}")
            skipped_frames += 1
            continue

        frame_output_dir.mkdir(parents=True, exist_ok=True)
        geometry = _get_image_preprocess_geometry(frame_path)

        pts3d = squeeze_batch_dim(to_numpy_array(pred["pts"])).astype(np.float32, copy=False)
        depth_conf_processed = squeeze_batch_dim(to_numpy_array(pred["conf"])).astype(
            np.float32, copy=False
        )
        k_processed = squeeze_batch_dim(to_numpy_array(pred["intrinsic"])).astype(
            np.float32, copy=False
        )
        k_original = _map_processed_intrinsics_to_original_frame(
            k_processed, geometry=geometry
        )
        c2w = squeeze_batch_dim(to_numpy_array(pred["extrinsic"])).astype(np.float32, copy=False)
        w2c = np.linalg.inv(c2w).astype(np.float32, copy=False)
        cam_points = world_to_camera(pts3d, w2c)
        # After projection with K, the third coordinate remains the camera-space z depth.
        depth_processed = camera_points_to_depth(cam_points, k_processed).astype(
            np.float32, copy=False
        )
        depth = cv2.resize(
            depth_processed,
            (geometry["orig_width"], geometry["orig_height"]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        depth_conf = cv2.resize(
            depth_conf_processed,
            (geometry["orig_width"], geometry["orig_height"]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        np.save(frame_output_dir / "pts3d_processed.npy", pts3d)
        np.save(frame_output_dir / "depth_conf.npy", depth_conf)
        np.save(frame_output_dir / "depth_conf_processed.npy", depth_conf_processed)
        np.save(frame_output_dir / "depth.npy", depth)
        np.save(frame_output_dir / "depth_processed.npy", depth_processed)
        np.save(frame_output_dir / "c2w.npy", c2w)
        np.save(frame_output_dir / "K_processed.npy", k_processed)
        np.save(frame_output_dir / "K.npy", k_original)
        img = to_uint8_rgb_image(view)
        cv2.imwrite(
            str(frame_output_dir / "rgb_processed.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        saved_frames += 1

    return saved_frames, skipped_frames


def run_sequence(
    model,
    device: str,
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

    with torch.no_grad():
        predictions, _ = inference(
            images,
            model,
            device,
            dtype="bf16-mixed",
            profiling=True,
            verbose=True,
            use_center_as_anchor=False,
        )

    saved_frames, skipped_frames = save_sequence_outputs(
        predictions,
        frame_paths,
        output_dir,
        skip_existing=skip_existing,
    )

    # print(
    #     f">> Inference complete for {sequence_dir.name}: "
    #     f"{len(predictions['preds'])} views"
    # )
    # if skip_existing:
    #     print(f">> Saved {saved_frames} frame outputs and skipped {skipped_frames} existing ones")
    # print(f">> Saved outputs to {output_dir}")
    return predictions


def main():
    args = parse_args()
    sequence_dirs = resolve_sequence_dirs(args.aim_root, args.animal, args.sequence)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Arc.from_pretrained("Luo-Yihang/4RC").to(device)
    model.eval()

    for sequence_dir in tqdm.tqdm(sequence_dirs, desc="Sequences"):
        try:
            run_sequence(
                model,
                device,
                args.animal,
                sequence_dir,
                args.results_root,
                max_rgb_frames=args.max_rgb_frames,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            print(f"Encountered {e}")


if __name__ == "__main__":
    main()
