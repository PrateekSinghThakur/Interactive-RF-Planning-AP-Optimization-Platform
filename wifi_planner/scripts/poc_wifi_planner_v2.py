#!/usr/bin/env python3
"""WiFi Planner V2 proof-of-concept experiment.

Usage:
  python scripts/poc_wifi_planner_v2.py uploads/floorplan.jpeg --out poc_outputs

This validates the architecture:
  Building footprint -> coverage grid -> wall obstacles -> propagation -> heatmap
instead of wall detections defining where coverage exists.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def clean_binary(img: np.ndarray, min_area: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def building_mask_from_clean(clean: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    inv = cv2.bitwise_not(closed)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = clean.shape[:2]
    best_area = 0
    building_mask = np.zeros_like(clean)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        touches_border = bool(np.any(xs == 0) or np.any(xs == w - 1) or np.any(ys == 0) or np.any(ys == h - 1))
        if touches_border:
            continue
        if area > best_area:
            best_area = area
            building_mask = np.zeros_like(inv)
            building_mask[labels == i] = 255
    return building_mask


def extract_walls(clean: np.ndarray):
    lines = cv2.HoughLinesP(clean, rho=1, theta=np.pi / 180, threshold=100, minLineLength=150, maxLineGap=20)
    walls = []
    if lines is None:
        return walls
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        walls.append(((x1, y1), (x2, y2)))
    return walls


def wall_loss(ap, point, walls, loss_per_wall=15):
    loss = 0
    for wall in walls:
        if intersect(ap, point, wall[0], wall[1]):
            loss += loss_per_wall
    return loss


def rssi(ap, point, walls, tx_power=20):
    distance = np.linalg.norm(np.array(ap, dtype=float) - np.array(point, dtype=float))
    distance_loss = 20 * np.log10(distance + 1)
    return tx_power - distance_loss - wall_loss(ap, point, walls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("poc_outputs"))
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    clean = clean_binary(img)
    building_mask = building_mask_from_clean(clean)
    walls = extract_walls(clean)

    h, w = building_mask.shape
    ys, xs = np.where(building_mask > 0)
    if len(xs) == 0:
        ap = (w // 2, h // 2)
    else:
        ap = (int(np.median(xs)), int(np.median(ys)))

    heat = np.full((h, w), np.nan, dtype=np.float32)
    grid_points = []
    for y in range(0, h, args.step):
        for x in range(0, w, args.step):
            if building_mask[y, x] > 0:
                grid_points.append((x, y))
                heat[y, x] = rssi(ap, (x, y), walls)

    # Dilate sparse grid samples for visualization only.
    heat_vis = np.nan_to_num(heat, nan=-120.0)
    sparse = np.zeros((h, w), dtype=np.float32)
    for x, y in grid_points:
        cv2.circle(sparse, (x, y), max(4, args.step // 2), float(heat_vis[y, x]), -1)

    norm = np.clip((sparse + 95) / 60 * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[sparse == 0] = 0
    overlay = img.copy()
    mask = sparse != 0
    overlay[mask] = cv2.addWeighted(img, 0.48, color, 0.52, 0)[mask]
    cv2.circle(overlay, ap, 10, (255, 0, 0), -1)

    cv2.imwrite(str(args.out / "01_clean_structural.png"), clean)
    cv2.imwrite(str(args.out / "02_building_mask.png"), building_mask)
    cv2.imwrite(str(args.out / "03_heatmap_overlay.png"), overlay)

    print(f"image={args.image}")
    print(f"building_pixels={int(np.count_nonzero(building_mask))}")
    print(f"grid_points={len(grid_points)}")
    print(f"walls={len(walls)}")
    print(f"ap={ap}")
    print(f"outputs={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
