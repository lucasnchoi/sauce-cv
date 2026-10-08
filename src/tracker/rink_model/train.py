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

from tracker.rink_model.data import LabelledFrames, Sample, collate, image_tensor
from tracker.rink_model.model import N_KEYPOINTS, build_model, save_weights
from tracker.video import Clip

PACKAGES = ["torch", "torchvision", "polars", "opencv-python", "numpy"]


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 30
    batch_size: int = 2
    lr: float = 0.0025
    momentum: float = 0.9
    weight_decay: float = 1e-4
    val_share: float = 0.2
    seed: int = 20261007
    pck_px: float = 10.0
    max_epoch_minutes: float = 10.0

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


@torch.no_grad()
def predict_samples(model, samples: list[Sample], sequences: dict[str, Path], device) -> dict[tuple[str, int], np.ndarray]:
    model.eval()
    clips: dict[str, Clip] = {}
    out: dict[tuple[str, int], np.ndarray] = {}
    for sample in samples:
        if sample.clip_id not in clips:
            clips[sample.clip_id] = Clip(sequences[sample.clip_id])
        image = clips[sample.clip_id].read_frame(sample.frame)
        result = model([image_tensor(image).to(device)])[0]
        points = np.zeros((N_KEYPOINTS, 3), dtype=float)
        if len(result["scores"]):
            best = int(torch.argmax(result["scores"]))
            points[:, :2] = result["keypoints"][best, :, :2].cpu().numpy()
            points[:, 2] = torch.sigmoid(result["keypoints_scores"][best]).cpu().numpy()
        out[(sample.clip_id, sample.frame)] = points
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
          settings: TrainSettings, device, pretrained: bool = True, size: dict | None = None, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    size = dict(size or {})
    model = build_model(pretrained, **size).to(device)
    loader = torch.utils.data.DataLoader(
        LabelledFrames(train_samples, sequences, augment=True, seed=settings.seed),
        batch_size=settings.batch_size, shuffle=True, collate_fn=collate,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=settings.lr, momentum=settings.momentum, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, settings.epochs))
    meta = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "settings": settings.as_dict(),
            "pretrained": pretrained, "size": size}
    record = {"settings": settings.as_dict(), "device": str(device), "pretrained": pretrained, "size": size,
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
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        scheduler.step()
        minutes = (time.time() - started) / 60
        entry = {"epoch": epoch, "loss": total / max(1, len(loader)), "minutes": round(minutes, 2)}
        if val_samples:
            entry.update(pck(predict_samples(model, val_samples, sequences, device), val_samples, settings.pck_px))
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
