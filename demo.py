#!/usr/bin/env python3
"""
4RC batch demo for SAV-style exports (same layout as Depth-Anything-3 ``demo.py``).

Layout:
  <sequence_root>/00000/rgb.png
  <sequence_root>/00001/rgb.png
  ...

For each five-digit frame folder, runs 4RC on ``rgb.png``, writes ``4rc_depth.npy`` (camera-frame
positive Z from predicted world points), then writes ``<parent>/<sequence_name>_4rc_depth.mp4``
with depth visualization matching Any4D ``scripts/demo_sav.py``.

Usage:
  python demo.py --sequence /path/to/sav_000/sav_000001
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import load_images
from arc.models.arc.arc import Arc

FRAME_DIR_PATTERN = re.compile(r"^\d{5}$")
RGB_CANDIDATES = ("rgb.png", "rgb.jpg", "image.png", "frame.png")
DEPTH_NPY_NAME = "4rc_depth.npy"


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Same implementation as Any4D ``scripts/demo_sav.py`` (depth colormap + inverse-depth scaling).
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


def list_frame_dirs(sequence_root: Path) -> list[Path]:
    dirs = [
        p
        for p in sequence_root.iterdir()
        if p.is_dir() and FRAME_DIR_PATTERN.match(p.name)
    ]
    return sorted(dirs, key=lambda p: int(p.name))


def find_rgb(frame_dir: Path) -> Path | None:
    for name in RGB_CANDIDATES:
        p = frame_dir / name
        if p.is_file():
            return p
    return None


def global_depth_viz_range_demo_sav(planes: list[np.ndarray]) -> tuple[float, float]:
    """Match ``demo_sav.py`` video block: flatten valid depths, then 2/98 pct on inverse depth."""
    valid_chunks = [d[d > 0] for d in planes if np.any(d > 0)]
    if not valid_chunks:
        return 0.0, 1.0
    all_depths = np.concatenate(valid_chunks)
    global_min = float(np.percentile(1 / all_depths, 2))
    global_max = float(np.percentile(1 / all_depths, 98))
    if global_min == global_max:
        global_min -= 1e-6
        global_max += 1e-6
    return global_min, global_max


def load_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pred_to_depth_cam_z(pred: dict) -> np.ndarray:
    """
    Recover per-pixel camera-frame Z from predicted world points and stored c2w extrinsic.
    Postprocessed 4RC preds expose ``pts`` (world) and ``extrinsic`` (camera-to-world, 4x4).
    """
    pts = pred["pts"]
    ext = pred["extrinsic"]
    if pts.dim() == 4:
        pts = pts[0]
    if ext.dim() == 3:
        ext = ext[0]

    pts_np = pts.detach().cpu().numpy().astype(np.float64)
    c2w = ext.detach().cpu().numpy().astype(np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)])
    if c2w.shape != (4, 4):
        raise ValueError(f"Unexpected extrinsic shape {c2w.shape}, expected 4x4 or 3x4")

    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    flat = pts_np.reshape(-1, 3)
    pc = (flat @ R.T) + t
    z = pc[:, 2].reshape(pts_np.shape[0], pts_np.shape[1]).astype(np.float32)
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    return z


def build_video_frame(
    rgb_path: Path | None,
    depth: np.ndarray,
    global_min: float,
    global_max: float,
    side_by_side: bool,
) -> np.ndarray:
    depth_vis = visualize_depth(depth, depth_min=global_min, depth_max=global_max)
    if not side_by_side or rgb_path is None:
        return depth_vis
    rgb = load_rgb_u8(rgb_path)
    h, w = depth_vis.shape[:2]
    rgb_r = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([rgb_r, depth_vis], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sequence",
        type=Path,
        default=Path("/home/share/public_nas/Dataset/3D_scene/sam4d-test-data/output/weihan/sav/sav_000/sav_000001"),
        help="Path to one sequence folder (contains 00000, 00001, ... subfolders).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="Luo-Yihang/4RC",
        help="Hugging Face hub id or local checkpoint for Arc.from_pretrained",
    )
    p.add_argument("--size", type=int, default=512, help="Long-edge image size for load_images (multiple of patch_size)")
    p.add_argument("--fps", type=float, default=10.0, help="FPS for the output depth video")
    p.add_argument(
        "--side-by-side",
        action="store_true",
        help="Each video frame is RGB | depth colormap (RGB resized to depth resolution).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Skip inference when {DEPTH_NPY_NAME} already exists (still included in the video).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sequence_root = args.sequence.expanduser().resolve()
    if not sequence_root.is_dir():
        print(f"Not a directory: {sequence_root}", file=sys.stderr)
        return 1

    frame_dirs = list_frame_dirs(sequence_root)
    if not frame_dirs:
        print(f"No five-digit frame folders under {sequence_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available; running on CPU (slow).", file=sys.stderr)

    dtype = "bf16-mixed" if device.type == "cuda" else "32"

    print(f"Loading 4RC model {args.checkpoint!r} on {device} …")
    model = Arc.from_pretrained(args.checkpoint).to(device)
    model.eval()

    depths_ordered: list[np.ndarray] = []
    rgb_paths_ordered: list[Path | None] = []

    for frame_dir in frame_dirs:
        rgb_path = find_rgb(frame_dir)
        if rgb_path is None:
            print(f"Skip {frame_dir.name}: no rgb image ({', '.join(RGB_CANDIDATES)})", file=sys.stderr)
            continue

        out_npy = frame_dir / DEPTH_NPY_NAME
        if args.skip_existing and out_npy.is_file():
            depth = np.load(out_npy)
            if depth.ndim != 2:
                print(f"Skip {frame_dir.name}: unexpected {DEPTH_NPY_NAME} shape {depth.shape}", file=sys.stderr)
                continue
        else:
            imgs = load_images([str(rgb_path)], size=args.size, verbose=False, patch_size=Arc.PATCH_SIZE)
            for img in imgs:
                img["track_query_idx"] = torch.tensor([0])

            with torch.no_grad():
                output_dict = inference(
                    imgs,
                    model,
                    device,
                    dtype=dtype,
                    verbose=False,
                    profiling=False,
                    use_center_as_anchor=False,
                )

            pred = output_dict["preds"][0]
            depth = pred_to_depth_cam_z(pred)
            np.save(out_npy, depth)
            print(f"{frame_dir.name}: wrote {out_npy}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

        depths_ordered.append(depth)
        rgb_paths_ordered.append(rgb_path if args.side_by_side else None)

    if not depths_ordered:
        print("No frames processed.", file=sys.stderr)
        return 1

    global_min, global_max = global_depth_viz_range_demo_sav(depths_ordered)
    frames = [
        build_video_frame(rgb_paths_ordered[i], depths_ordered[i], global_min, global_max, args.side_by_side)
        for i in range(len(depths_ordered))
    ]

    parent = sequence_root.parent
    video_path = sequence_root / f"4rc_depth.mp4"
    os.makedirs(parent, exist_ok=True)
    iio.imwrite(str(video_path), np.stack(frames), fps=args.fps)
    print(f"Wrote {video_path} ({len(frames)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
