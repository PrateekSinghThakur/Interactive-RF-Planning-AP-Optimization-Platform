"""Simple floorplan detector for WiFi Planner V2.

This detector deliberately avoids full building understanding. For the Wi-Fi POC we
only need two reliable layers:

1. building_footprints: where Wi-Fi coverage/grid points may exist
2. walls: attenuation obstacles crossed by propagation rays/grid rasterization

Rooms, doors, windows and semantic labels are optional and left empty unless a
future module supplies them. The app layer can still let users add/edit them.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

__all__ = ["detect"]


THRESHOLD_VALUE = 200
MIN_COMPONENT_AREA = 150
DOOR_GAP_CLOSE_KERNEL = 11
HOUGH_THRESHOLD = 100
HOUGH_MIN_LINE_LENGTH = 150
HOUGH_MAX_LINE_GAP = 20
DEMO_WALL_ATTENUATION_DB = 8


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image")
    return img


def _resize_for_speed(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    return binary


def _remove_small_components(binary: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255
    return clean


def _close_gaps(clean: np.ndarray, kernel_size: int = DOOR_GAP_CLOSE_KERNEL) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)


def _polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _largest_non_border_open_component(closed: np.ndarray) -> tuple[np.ndarray, int | None]:
    inv = cv2.bitwise_not(closed)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = labels.shape
    best_area = 0
    best_label: int | None = None
    building_mask = np.zeros_like(inv)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        touches_border = bool(
            np.any(xs == 0)
            or np.any(xs == w - 1)
            or np.any(ys == 0)
            or np.any(ys == h - 1)
        )
        if touches_border:
            continue
        if area > best_area:
            best_area = area
            best_label = i
            building_mask[:] = 0
            building_mask[labels == i] = 255

    return building_mask, best_label


def _external_envelope_fallback(clean: np.ndarray) -> np.ndarray:
    h, w = clean.shape[:2]
    long_side = max(h, w)
    fallback = clean.copy()

    # Remove exported page/title frame from the envelope operation.
    margin = max(20, int(long_side * 0.02))
    fallback[:margin, :] = 0
    fallback[-margin:, :] = 0
    fallback[:, :margin] = 0
    fallback[:, -margin:] = 0

    kernel_size = max(15, int(long_side * 0.018))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    blob = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel, iterations=2)
    blob = cv2.dilate(blob, kernel, iterations=1)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def _mask_to_footprints(mask: np.ndarray) -> list[dict[str, Any]]:
    h, w = mask.shape[:2]
    image_area = h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, list[list[float]]]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.005:
            continue
        epsilon = max(5.0, 0.006 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        poly = [[round(float(p[0][0]), 3), round(float(p[0][1]), 3)] for p in approx]
        if _polygon_area(poly) <= 0:
            continue
        candidates.append((area, poly))

    candidates.sort(key=lambda item: item[0], reverse=True)
    footprints: list[dict[str, Any]] = []
    for idx, (area, poly) in enumerate(candidates[:4], 1):
        footprints.append(
            {
                "id": f"fp_det_{idx}",
                "boundary_polygon_m": poly,
                "confidence": round(float(max(0.35, min(0.92, 0.45 + area / max(1, image_area) * 1.35))), 3),
                "user_edited": False,
            }
        )
    return footprints


def _detect_building_footprints(clean: np.ndarray, closed: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    building_mask, label = _largest_non_border_open_component(closed)
    ratio = float(np.count_nonzero(building_mask) / max(1, building_mask.size))
    method = "largest_non_border_open_component"

    # If the interior component is too small, use a simple external envelope.
    if ratio < 0.02:
        building_mask = _external_envelope_fallback(clean)
        method = "external_envelope_fallback"

    footprints = _mask_to_footprints(building_mask)
    return footprints, {
        "method": method,
        "component_label": label,
        "footprint_count": len(footprints),
        "footprint_pixel_ratio": ratio,
    }


def _line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180


def _dedupe_walls(raw_lines: list[tuple[int, int, int, int]], max_count: int = 120) -> list[tuple[int, int, int, int]]:
    raw_lines.sort(key=lambda line: math.hypot(line[2] - line[0], line[3] - line[1]), reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line
        length = math.hypot(x2 - x1, y2 - y1)
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        angle = _line_angle(x1, y1, x2, y2)
        duplicate = False
        for sx1, sy1, sx2, sy2 in selected:
            slen = math.hypot(sx2 - sx1, sy2 - sy1)
            smid = np.array([(sx1 + sx2) / 2, (sy1 + sy2) / 2])
            sangle = _line_angle(sx1, sy1, sx2, sy2)
            if min(abs(angle - sangle), abs(abs(angle - sangle) - 180)) > 5:
                continue
            if np.linalg.norm(mid - smid) < 10 and abs(length - slen) / max(length, slen, 1) < 0.30:
                duplicate = True
                break
        if not duplicate:
            selected.append(line)
        if len(selected) >= max_count:
            break
    return selected


def _detect_walls(clean: np.ndarray) -> list[dict[str, Any]]:
    lines = cv2.HoughLinesP(
        clean,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return []

    raw_lines = [tuple(int(v) for v in line[0]) for line in lines]
    selected = _dedupe_walls(raw_lines)
    h, w = clean.shape[:2]
    long_side = max(h, w)

    walls: list[dict[str, Any]] = []
    for idx, (x1, y1, x2, y2) in enumerate(selected, 1):
        length = math.hypot(x2 - x1, y2 - y1)
        confidence = max(0.45, min(0.92, 0.50 + length / max(long_side, 1) * 0.50))
        walls.append(
            {
                "id": f"w_det_{idx}",
                "start_m": [float(x1), float(y1)],
                "end_m": [float(x2), float(y2)],
                "thickness_m": 3.0,
                "material": "unknown",
                "attenuation_db": DEMO_WALL_ATTENUATION_DB,
                "confidence": round(float(confidence), 3),
                "user_edited": False,
            }
        )
    return walls


def _image_tier(binary: np.ndarray, clean: np.ndarray) -> str:
    ink_ratio = float(np.count_nonzero(binary) / max(1, binary.size))
    retained_ratio = float(np.count_nonzero(clean) / max(1, binary.size))
    if min(binary.shape[:2]) < 500:
        return "C"
    if ink_ratio > 0.35 or retained_ratio < 0.005:
        return "B"
    return "A"


def detect(image_bytes: bytes, mode: str = "simple") -> dict[str, Any]:
    """Detect only what Wi-Fi propagation needs: footprint + wall obstacles."""
    img = _resize_for_speed(_decode_image(image_bytes))
    binary = _binarize(img)
    clean = _remove_small_components(binary)
    closed = _close_gaps(clean)

    footprints, footprint_metrics = _detect_building_footprints(clean, closed)
    walls = _detect_walls(clean)
    tier = _image_tier(binary, clean)

    ink_ratio = float(np.count_nonzero(binary) / max(1, binary.size))
    clean_ratio = float(np.count_nonzero(clean) / max(1, clean.size))
    route_to_trace = not footprints or tier == "C"

    return {
        "schema_version": "0.1.0",
        "detection": {
            "engine": "simple_footprint_wall_detector",
            "mode": "simple",
            "tier": tier,
            "route_to_trace_mode": route_to_trace,
            "review_confidence_threshold": 0.55,
            "metrics": {
                "ink_ratio": ink_ratio,
                "clean_ratio": clean_ratio,
                "wall_count": len(walls),
                "footprint": footprint_metrics,
                "note": "Simplified detector: building footprint defines grid; walls are attenuation obstacles only.",
            },
            "image_size_px": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "note": None,
        },
        "walls": walls,
        "rooms": [],
        "doors": [],
        "windows": [],
        "building_footprints": footprints,
    }
