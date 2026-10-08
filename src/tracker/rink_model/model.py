"""The rink keypoint model: torchvision Keypoint R-CNN (BSD-3) with a 33-keypoint head.

COCO weights are loaded with their 17-keypoint person head, which is then
replaced; the class head stays two classes (background, rink). The inference
wrapper is stage 1's detector. Nothing outside rink_model imports torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

from tracker.rink import KEYPOINTS
from tracker.rink_model.data import image_tensor

N_KEYPOINTS = len(KEYPOINTS)
NATIVE_SIZE = {"min_size": 720, "max_size": 1280}     # the clips' own size: no resize
KEYPOINT_HEAD_CHANNELS = 512


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(pretrained: bool = True, **size) -> torch.nn.Module:
    sizes = {**NATIVE_SIZE, **size}
    if pretrained:
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, **sizes)
    else:
        model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2,
                                          num_keypoints=N_KEYPOINTS, **sizes)
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(KEYPOINT_HEAD_CHANNELS, N_KEYPOINTS)
    return model


def save_weights(model: torch.nn.Module, path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path, device: str | torch.device | None = None, **size) -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rink keypoint weights not found at {path}")
    saved = torch.load(path, map_location="cpu")
    model = build_model(pretrained=False, **size)
    model.load_state_dict(saved["state_dict"])
    target = device if isinstance(device, torch.device) else pick_device(device)
    return model.to(target).eval(), dict(saved.get("meta", {}))


class RinkKeypointDetector:
    """Stage 1's model. conf is the sigmoid of the keypoint heatmap score of the best rink detection."""

    name = "keypointrcnn-resnet50-fpn-rink33"

    def __init__(self, weights_path, threshold: float = 0.3, device: str | None = None, **size):
        self.weights_path = str(weights_path)
        self.threshold = threshold
        self.device = pick_device(device)
        self._model, self.meta = load_model(weights_path, self.device, **size)

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        points = np.zeros((N_KEYPOINTS, 3), dtype=float)
        out = self._model([image_tensor(image_bgr).to(self.device)])[0]
        if len(out["scores"]) == 0:
            return points
        best = int(torch.argmax(out["scores"]))
        xy = out["keypoints"][best, :, :2].cpu().numpy()
        conf = torch.sigmoid(out["keypoints_scores"][best]).cpu().numpy()
        keep = conf >= self.threshold
        points[keep, :2] = xy[keep]
        points[keep, 2] = conf[keep]
        return points
