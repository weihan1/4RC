import argparse
from pathlib import Path

import numpy as np
import torch

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import load_images, rgb
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
FRAME_OUTPUT_FILES = (
    "pts3d.npy",
    "depth_conf.npy",
    "c2w.npy",
    "K.npy",
    "rgb_processed.png",
)


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


def load_sequence_images(sequence_dir: Path):
    rgb_frames = list_aim_rgb_frames(sequence_dir)
    if not rgb_frames:
        raise ValueError(f"No AiM RGB frames found in {sequence_dir}")

    # print(f">> Loading sequence {sequence_dir.name} from {sequence_dir}")
    # print(f"   Found {len(rgb_frames)} RGB frames")
    images = load_images(
        [str(path) for path in rgb_frames],
        size=512,
        patch_size=14,
        verbose=True,
        square_ok=True #force square inputs
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

        np.save(frame_output_dir / "pts3d.npy", squeeze_batch_dim(to_numpy_array(pred["pts"])))
        np.save(frame_output_dir / "depth_conf.npy", squeeze_batch_dim(to_numpy_array(pred["conf"])))
        c2w = torch.linalg.inv(pred["extrinsic"])
        np.save(frame_output_dir / "c2w.npy", squeeze_batch_dim(to_numpy_array(c2w)))
        np.save(frame_output_dir / "K.npy", squeeze_batch_dim(to_numpy_array(pred["intrinsic"])))
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
    skip_existing: bool = False,
):
    frame_paths = list_aim_rgb_frames(sequence_dir)
    if not frame_paths:
        raise ValueError(f"No AiM RGB frames found in {sequence_dir}")

    output_dir = results_root / animal / sequence_dir.name
    if skip_existing and sequence_outputs_exist(frame_paths, output_dir):
        # print(f">> Skipping {sequence_dir.name}: outputs already exist in {output_dir}")
        return None

    images, frame_paths = load_sequence_images(sequence_dir)

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
                skip_existing=args.skip_existing,
            )
        except:
            continue


if __name__ == "__main__":
    main()
