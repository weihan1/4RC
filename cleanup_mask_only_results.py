import argparse
import shutil
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path(
    "/home/share/public_nas/Dataset/3D_scene/viscam/downloads/AiM/AiM_results"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Delete frame result directories whose 4rc folder contains only "
            "mask_processed.png."
        )
    )
    parser.add_argument(
        "--animal",
        type=str,
        default="boar",
        help="Animal category under the results root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing per-animal AiM results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories that would be removed without deleting them.",
    )
    return parser.parse_args()


def is_mask_only_4rc_dir(frame_dir: Path) -> bool:
    four_rc_dir = frame_dir / "4rc"
    if not four_rc_dir.is_dir():
        return False

    files = sorted(path.name for path in four_rc_dir.iterdir() if path.is_file())
    return files == ["mask_processed.png"]


def remove_empty_parents(start_dir: Path, stop_dir: Path, dry_run: bool) -> None:
    current = start_dir
    while current != stop_dir and current.exists():
        if any(current.iterdir()):
            break
        if dry_run:
            print(f"Would remove empty directory: {current}")
        else:
            current.rmdir()
            print(f"Removed empty directory: {current}")
        current = current.parent


def main():
    args = parse_args()
    animal_dir = args.results_root / args.animal

    if not animal_dir.exists():
        raise FileNotFoundError(f"Animal results directory does not exist: {animal_dir}")
    if not animal_dir.is_dir():
        raise NotADirectoryError(f"Animal results path is not a directory: {animal_dir}")

    removed_frames = 0
    examined_frames = 0

    for sequence_dir in sorted(path for path in animal_dir.iterdir() if path.is_dir()):
        for frame_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
            examined_frames += 1
            if not is_mask_only_4rc_dir(frame_dir):
                continue

            if args.dry_run:
                print(f"Would remove frame directory: {frame_dir}")
            else:
                shutil.rmtree(frame_dir)
                print(f"Removed frame directory: {frame_dir}")

            removed_frames += 1
            remove_empty_parents(sequence_dir, animal_dir, args.dry_run)

    print(
        f"Examined {examined_frames} frame directories under {animal_dir}. "
        f"{'Would remove' if args.dry_run else 'Removed'} {removed_frames}."
    )


if __name__ == "__main__":
    main()
