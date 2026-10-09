"""Stage 3: one pixel-to-rink homography per frame (3a, every frame) and its smoothing within play segments (3b).

Fit: RANSAC from the frame's keypoints to the template, in the feet -> pixel
direction so the RANSAC and validity thresholds are in pixels; the stored H
is the inverse (pixels -> feet) with h22 = 1. Valid: enough inliers, not
collinear, median reprojection error under the limit, taken over all the frame's
points (not just the RANSAC inliers), so a frame where fewer than half the
points agree is invalid. Then, within a
segment, fitted transforms are smoothed over a short window and short gaps
are interpolated; longer gaps stay invalid, and frames outside any segment get no row.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np
import polars as pl

from tracker.rink import template_points
from tracker.schemas import H_COLUMNS, from_rows, validate


@dataclass(frozen=True)
class RinkTransformSettings:
    min_inliers: int = 4
    max_reproj_px: float = 10.0
    ransac_px: float = 5.0
    smooth_window: int = 5
    interpolate_gap_s: float = 0.5

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, values: dict) -> "RinkTransformSettings":
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown rink transform settings {unknown}; known: {sorted(known)}")
        casts = {"int": int, "float": float}
        return cls(**{key: casts[known[key]](value) for key, value in values.items()})


DEFAULT_RINK_TRANSFORM = RinkTransformSettings()


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.c_[pts, np.ones(len(pts))] @ H.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def invert(H: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(H)
    return inverse / inverse[2, 2]


def _spread(points: np.ndarray) -> bool:
    """False when the points are (nearly) collinear: a homography needs four points in general position."""
    centred = points - points.mean(axis=0)
    return np.linalg.matrix_rank(centred, tol=1e-6 * max(1.0, np.abs(centred).max())) == 2


def fit_frame(pixels: np.ndarray, ids: np.ndarray, settings: RinkTransformSettings) -> tuple[np.ndarray | None, int, float | None]:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(ids, dtype=int)
    if len(ids) < max(4, settings.min_inliers) or not _spread(pixels):
        return None, 0, None
    feet = template_points()[ids]
    try:
        H_ft2px, mask = cv2.findHomography(feet, pixels, cv2.RANSAC, settings.ransac_px)
    except cv2.error:
        return None, 0, None
    if H_ft2px is None or mask is None or not np.isfinite(H_ft2px).all() or abs(np.linalg.det(H_ft2px)) < 1e-12:
        return None, 0, None
    inliers = mask.ravel().astype(bool)
    n_inliers = int(inliers.sum())
    if n_inliers < settings.min_inliers or not _spread(pixels[inliers]) or not _spread(feet[inliers]):
        return None, n_inliers, None
    error = float(np.median(np.linalg.norm(project(H_ft2px, feet) - pixels, axis=1)))
    if not np.isfinite(error) or error >= settings.max_reproj_px:
        return None, n_inliers, error
    return invert(H_ft2px), n_inliers, error


def _row(clip_id: str, frame: int, H: np.ndarray | None, n_inliers: int, error: float | None, status: str) -> dict:
    values = {h: None for h in H_COLUMNS} if H is None else dict(zip(H_COLUMNS, [float(v) for v in H.ravel()]))
    return {"clip_id": clip_id, "frame": int(frame), **values, "n_inliers": int(n_inliers),
            "reproj_error_px": None if error is None else float(error), "status": status}


def fit_all(keypoints: pl.DataFrame, clip_id: str, n_frames: int,
            settings: RinkTransformSettings = DEFAULT_RINK_TRANSFORM) -> pl.DataFrame:
    """Stage 3a: the raw fit of every frame of the clip, `fitted` or `invalid`. No smoothing, no interpolation."""
    keypoints = validate(keypoints, "keypoints")
    points: dict[int, pl.DataFrame] = {}
    if keypoints.height:
        for key, part in keypoints.partition_by("frame", as_dict=True).items():
            points[int(key[0] if isinstance(key, tuple) else key)] = part
    rows = []
    for frame in range(n_frames):
        part = points.get(frame)
        if part is None:
            H, n_inliers, error = None, 0, None
        else:
            H, n_inliers, error = fit_frame(part.select("x_px", "y_px").to_numpy(), part["keypoint_id"].to_numpy(), settings)
        rows.append(_row(clip_id, frame, H, n_inliers, error, "fitted" if H is not None else "invalid"))
    return validate(from_rows("rink_fits", rows), "rink_fits")


def smooth_transforms(fits: pl.DataFrame, frames: pl.DataFrame, fps: float,
                      settings: RinkTransformSettings = DEFAULT_RINK_TRANSFORM) -> pl.DataFrame:
    """Stage 3b: within each play segment, average fitted frames over a short window and interpolate short gaps."""
    fits = validate(fits, "rink_fits")
    frames = validate(frames, "frames")
    play = frames.filter(pl.col("is_play")).sort("frame")
    if play.height == 0:
        return validate(from_rows("rink_transforms", []), "rink_transforms")
    clip_id = str(play["clip_id"][0])
    raw: dict[int, tuple[np.ndarray | None, int, float | None]] = {}
    for row in fits.iter_rows(named=True):
        H = np.array([row[h] for h in H_COLUMNS], dtype=float).reshape(3, 3) if row["status"] == "fitted" else None
        raw[int(row["frame"])] = (H, int(row["n_inliers"]), row["reproj_error_px"])
    missing = [int(f) for f in play["frame"].to_list() if int(f) not in raw]
    if missing:
        raise ValueError(f"rink_fits has no row for play frames {missing[:5]}")
    half = max(0, settings.smooth_window // 2)
    max_gap = round(settings.interpolate_gap_s * fps)
    rows: dict[int, dict] = {}
    for segment_id, segment in play.partition_by("segment_id", as_dict=True).items():
        frame_list = sorted(int(f) for f in segment["frame"].to_list())
        fitted = [f for f in frame_list if raw[f][0] is not None]
        smoothed: dict[int, np.ndarray] = {}
        for i, f in enumerate(fitted):
            window = [raw[g][0] for g in fitted[max(0, i - half): i + half + 1] if abs(g - f) <= half]
            smoothed[f] = np.mean(window, axis=0)
            smoothed[f] = smoothed[f] / smoothed[f][2, 2]
        for f in frame_list:
            H, n_inliers, error = raw[f]
            if H is not None:
                rows[f] = _row(clip_id, f, smoothed[f], n_inliers, error, "fitted")
            else:
                rows[f] = _row(clip_id, f, None, n_inliers, error, "invalid")
        for a, b in zip(fitted, fitted[1:]):
            gap = b - a - 1
            if 0 < gap < max_gap:
                for f in range(a + 1, b):
                    weight = (f - a) / (b - a)
                    H = smoothed[a] + (smoothed[b] - smoothed[a]) * weight
                    rows[f] = _row(clip_id, f, H / H[2, 2], 0, None, "interpolated")
    table = from_rows("rink_transforms", [rows[f] for f in sorted(rows)])
    return validate(table, "rink_transforms")


def fit_transforms(keypoints: pl.DataFrame, frames: pl.DataFrame, fps: float,
                   settings: RinkTransformSettings = DEFAULT_RINK_TRANSFORM) -> pl.DataFrame:
    """Fit every frame, then smooth within the play segments of `frames`. Kept for callers that have both tables."""
    frames = validate(frames, "frames")
    if frames.height == 0:
        return validate(from_rows("rink_transforms", []), "rink_transforms")
    clip_id = str(frames["clip_id"][0])
    n_frames = int(frames["frame"].max()) + 1
    return smooth_transforms(fit_all(keypoints, clip_id, n_frames, settings), frames, fps, settings)


def transforms_by_frame(table: pl.DataFrame) -> dict[int, tuple[np.ndarray, str]]:
    table = validate(table, "rink_transforms").filter(pl.col("status") != "invalid")
    out = {}
    for row in table.iter_rows(named=True):
        out[int(row["frame"])] = (np.array([row[h] for h in H_COLUMNS], dtype=float).reshape(3, 3), row["status"])
    return out
