import argparse
from pathlib import Path

import numpy as np
import tqdm


DEFAULT_AIM_RESULTS_ROOT = Path(
    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_results"
)
FRAME_INPUT_FILES = ("pts3d.npy", "c2w.npy", "K.npy")
DEPTH_OUTPUT_FILE = "depth.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate depth.npy from existing 4RC outputs in AiM_results."
    )
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
        help="Sequence name. If omitted, process all sequences for the animal.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_AIM_RESULTS_ROOT,
        help="Root directory containing existing AiM 4RC outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Do not overwrite an existing depth.npy.",
    )
    return parser.parse_args()


def resolve_sequence_dirs(results_root: Path, animal: str, sequence: str | None):
    animal_dir = results_root / animal
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


def list_frame_output_dirs(sequence_dir: Path):
    return sorted(
        frame_dir / "4rc"
        for frame_dir in sequence_dir.iterdir()
        if frame_dir.is_dir() and (frame_dir / "4rc").is_dir()
    )


def frame_inputs_exist(frame_output_dir: Path):
    return all((frame_output_dir / name).exists() for name in FRAME_INPUT_FILES)


def world_to_camera(pts3d: np.ndarray, c2w: np.ndarray):
    if pts3d.ndim != 3 or pts3d.shape[-1] != 3:
        raise ValueError(f"Expected pts3d with shape [H, W, 3], got {pts3d.shape}")
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected c2w with shape [4, 4], got {c2w.shape}")

    w2c = np.linalg.inv(c2w)
    ones = np.ones((*pts3d.shape[:2], 1), dtype=pts3d.dtype)
    pts3d_h = np.concatenate([pts3d, ones], axis=-1)
    cam_points_h = pts3d_h @ w2c.T
    return cam_points_h[..., :3]


def camera_points_to_depth(cam_points: np.ndarray, K: np.ndarray):
    if K.shape != (3, 3):
        raise ValueError(f"Expected K with shape [3, 3], got {K.shape}")

    projected = cam_points @ K.T
    return projected[..., 2]


def generate_depth(frame_output_dir: Path, skip_existing: bool = False):
    depth_path = frame_output_dir / DEPTH_OUTPUT_FILE
    if skip_existing and depth_path.exists():
        return False

    if not frame_inputs_exist(frame_output_dir):
        missing = [name for name in FRAME_INPUT_FILES if not (frame_output_dir / name).exists()]
        raise FileNotFoundError(f"Missing {missing} in {frame_output_dir}")

    pts3d = np.load(frame_output_dir / "pts3d.npy")
    c2w = np.load(frame_output_dir / "c2w.npy")
    K = np.load(frame_output_dir / "K.npy")

    cam_points = world_to_camera(pts3d, c2w)
    # After projection with K, the third coordinate remains the camera-space z depth.
    depth = camera_points_to_depth(cam_points, K).astype(np.float32, copy=False)
    np.save(depth_path, depth)
    return True


def run_sequence(sequence_dir: Path, skip_existing: bool = False):
    frame_output_dirs = list_frame_output_dirs(sequence_dir)
    if not frame_output_dirs:
        raise ValueError(f"No timestep 4rc directories found in {sequence_dir}")

    saved_frames = 0
    skipped_frames = 0
    for frame_output_dir in frame_output_dirs:
        if skip_existing and (frame_output_dir / DEPTH_OUTPUT_FILE).exists():
            skipped_frames += 1
            continue

        wrote_depth = generate_depth(frame_output_dir, skip_existing=skip_existing)
        if wrote_depth:
            saved_frames += 1
        else:
            skipped_frames += 1

    return saved_frames, skipped_frames


def main():
    args = parse_args()
    sequence_dirs = resolve_sequence_dirs(args.results_root, args.animal, args.sequence)

    for sequence_dir in tqdm.tqdm(sequence_dirs, desc="Sequences"):
        try:
            saved_frames, skipped_frames = run_sequence(
                sequence_dir,
                skip_existing=args.skip_existing,
            )
            print(
                f"{sequence_dir.name}: saved {saved_frames} depth files, "
                f"skipped {skipped_frames}"
            )
        except Exception as exc:
            print(f"Failed on {sequence_dir}: {exc}")


if __name__ == "__main__":
    main()
