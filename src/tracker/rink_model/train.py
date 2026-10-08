"""Trains the rink keypoint model and records everything needed to reproduce the run.

Validation is a seeded slice of the same training clips, never the held-out
clips. Weights are chosen by PCK at 10 px on that slice. The stop line (one
epoch too slow) is enforced here so a long local run never starts by accident.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from tracker.rink_model.data import INPUT_HW, LabelledFrames, Sample, collate, image_tensor
from tracker.rink_model.model import N_KEYPOINTS, build_model, decode, save_weights
from tracker.video import Clip

PACKAGES = ["torch", "torchvision", "polars", "opencv-python", "numpy"]


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 60
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_share: float = 0.2
    seed: int = 20261007
    pck_px: float = 10.0
    max_epoch_minutes: float = 10.0
    vis_threshold: float = 0.5

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, values: dict) -> "TrainSettings":
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown training settings {unknown}; known: {sorted(known)}")
        casts = {"int": int, "float": float}
        return cls(**{key: casts[known[key]](value) for key, value in values.items()})


class TrainingTooSlow(RuntimeError):
    """The first epoch exceeded the stop line; the run record says so."""


def pck(predictions: dict[tuple[str, int], np.ndarray], samples: list[Sample], px: float) -> dict:
    hits, n_points, errors = 0, 0, []
    for sample in samples:
        pred = predictions.get((sample.clip_id, sample.frame))
        for k in range(N_KEYPOINTS):
            if sample.points[k, 2] <= 0:
                continue
            n_points += 1
            if pred is None or pred[k, 2] <= 0:
                continue
            error = math.hypot(pred[k, 0] - sample.points[k, 0], pred[k, 1] - sample.points[k, 1])
            errors.append(error)
            hits += error <= px
    return {"pck": hits / n_points if n_points else 0.0,
            "mean_px": float(np.mean(errors)) if errors else None,
            "n_points": n_points, "n_predicted": len(errors)}


POS_WEIGHT = 50.0


def heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy per cell, weighted up where the target is positive so the peaks are not drowned by zeros."""
    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_cell * (1.0 + POS_WEIGHT * targets)).mean()


def visibility(predictions: dict[tuple[str, int], np.ndarray], samples: list[Sample], threshold: float) -> dict:
    hit = n_visible = false = n_invisible = 0
    for sample in samples:
        pred = predictions.get((sample.clip_id, sample.frame))
        if pred is None:
            n_visible += int((sample.points[:, 2] > 0).sum()); n_invisible += int((sample.points[:, 2] <= 0).sum())
            continue
        visible = sample.points[:, 2] > 0
        confident = pred[:, 2] >= threshold
        hit += int((visible & confident).sum()); n_visible += int(visible.sum())
        false += int((~visible & confident).sum()); n_invisible += int((~visible).sum())
    return {"vis_recall": hit / n_visible if n_visible else 0.0,
            "false_per_frame": false / len(samples) if samples else 0.0,
            "n_visible": n_visible, "n_invisible": n_invisible}


@torch.no_grad()
def predict_samples(model, samples, sequences, device, input_hw=INPUT_HW) -> dict[tuple[str, int], np.ndarray]:
    model.eval()
    clips: dict[str, Clip] = {}
    out: dict[tuple[str, int], np.ndarray] = {}
    for sample in samples:
        if sample.clip_id not in clips:
            clips[sample.clip_id] = Clip(sequences[sample.clip_id])
        image = clips[sample.clip_id].read_frame(sample.frame)
        logits = model(image_tensor(image, input_hw)[None].to(device))[0]
        out[(sample.clip_id, sample.frame)] = decode(logits, image.shape[1], image.shape[0])
    return out


def _versions() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def train(train_samples: list[Sample], val_samples: list[Sample], sequences: dict[str, Path], out_dir,
          settings: TrainSettings, device, pretrained: bool = True, input_hw=INPUT_HW, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    model = build_model(pretrained).to(device)
    loader = torch.utils.data.DataLoader(
        LabelledFrames(train_samples, sequences, augment=True, seed=settings.seed, input_hw=input_hw),
        batch_size=settings.batch_size, shuffle=True, collate_fn=collate,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, settings.epochs))
    meta = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "settings": settings.as_dict(),
            "pretrained": pretrained, "input_hw": list(input_hw)}
    record = {"settings": settings.as_dict(), "device": str(device), "pretrained": pretrained, "input_hw": list(input_hw),
              "packages": _versions(), "n_train": len(train_samples), "n_val": len(val_samples),
              "train_frames": [[s.clip_id, s.frame] for s in train_samples],
              "val_frames": [[s.clip_id, s.frame] for s in val_samples],
              "history": [], "best_epoch": None, "best_pck": None, "stopped": None}

    def write_record() -> None:
        (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    for epoch in range(1, settings.epochs + 1):
        model.train()
        started = time.time()
        total = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            loss = heatmap_loss(model(images), targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        scheduler.step()
        minutes = (time.time() - started) / 60
        entry = {"epoch": epoch, "loss": total / max(1, len(loader)), "minutes": round(minutes, 2)}
        if val_samples:
            predictions = predict_samples(model, val_samples, sequences, device, input_hw)
            entry.update(pck(predictions, val_samples, settings.pck_px))
            entry.update(visibility(predictions, val_samples, settings.vis_threshold))
        record["history"].append(entry)
        log(json.dumps(entry))
        save_weights(model, out_dir / "last.pt", {**meta, "epoch": epoch})
        improved = not val_samples or record["best_pck"] is None or entry["pck"] > record["best_pck"]
        if improved:
            record["best_epoch"], record["best_pck"] = epoch, entry.get("pck")
            save_weights(model, out_dir / "best.pt", {**meta, "epoch": epoch, "pck": entry.get("pck")})
        if epoch == 1 and minutes > settings.max_epoch_minutes:
            record["stopped"] = (f"first epoch took {minutes:.1f} minutes, above the {settings.max_epoch_minutes:g}-minute "
                                 "stop line; ask about the pre-approved GPU rental before continuing")
            write_record()
            raise TrainingTooSlow(record["stopped"])
        write_record()
    return record


def check_frames(detector, samples: list[Sample], sequences: dict[str, Path], transform_settings=None) -> dict:
    """The rink scorer on labelled frames, from this model's detections: PCK plus transform validity and distance."""
    from tracker.evaluate.rink import FT_TO_M
    from tracker.rink import template_points
    from tracker.stages.rink_transform import DEFAULT_RINK_TRANSFORM, fit_frame, project

    settings = transform_settings or DEFAULT_RINK_TRANSFORM
    template = template_points()
    clips: dict[str, Clip] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    distances: list[float] = []
    n_valid = 0
    for sample in samples:
        if sample.clip_id not in clips:
            clips[sample.clip_id] = Clip(sequences[sample.clip_id])
        points = detector.detect(clips[sample.clip_id].read_frame(sample.frame))
        predictions[(sample.clip_id, sample.frame)] = points
        found = points[:, 2] > 0
        H, _, _ = fit_frame(points[found, :2], np.flatnonzero(found), settings)
        if H is None:
            continue
        n_valid += 1
        visible = sample.points[:, 2] > 0
        if visible.any():
            feet = project(H, sample.points[visible, :2])
            distances.extend((np.linalg.norm(feet - template[visible], axis=1) * FT_TO_M).tolist())
    result = pck(predictions, samples, 10.0)
    result.update({"n_frames": len(samples), "n_valid": n_valid,
                   "valid_share": n_valid / len(samples) if samples else None,
                   "median_m": float(np.median(distances)) if distances else None, "distances_m": distances})
    return result
