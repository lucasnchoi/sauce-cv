"""The rink keypoint model: a heatmap regressor whose peak value is a visibility score.

ResNet-50 features (stride 32) feed three stride-2 deconvolutions and a 1x1
convolution to one heatmap per keypoint at stride 4 (the SimpleBaseline layout,
Xiao et al. 2018). Trained with a zero target for invisible keypoints, so the
model learns absence as well as position. Nothing outside rink_model imports torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tracker.rink import KEYPOINTS
from tracker.rink_model.data import INPUT_HW, image_tensor

N_KEYPOINTS = len(KEYPOINTS)
DECONV_CHANNELS = 256


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class RinkHeatmapNet(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = torch.nn.Sequential(*list(backbone.children())[:-2])     # up to layer4: (B, 2048, H/32, W/32)
        layers: list[torch.nn.Module] = []
        channels = 2048
        for _ in range(3):
            layers += [torch.nn.ConvTranspose2d(channels, DECONV_CHANNELS, 4, stride=2, padding=1, bias=False),
                       torch.nn.BatchNorm2d(DECONV_CHANNELS), torch.nn.ReLU(inplace=True)]
            channels = DECONV_CHANNELS
        self.head = torch.nn.Sequential(*layers, torch.nn.Conv2d(DECONV_CHANNELS, N_KEYPOINTS, kernel_size=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


def build_model(pretrained: bool = True) -> RinkHeatmapNet:
    return RinkHeatmapNet(pretrained=pretrained)


def save_weights(model: torch.nn.Module, path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path, device: str | torch.device | None = None) -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rink keypoint weights not found at {path}")
    saved = torch.load(path, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(saved["state_dict"])
    target = device if isinstance(device, torch.device) else pick_device(device)
    return model.to(target).eval(), dict(saved.get("meta", {}))


def decode(logits: torch.Tensor, width: int, height: int) -> np.ndarray:
    """(33, 3) of x_px, y_px, conf from one image's heatmap logits (33, rows, cols)."""
    prob = torch.sigmoid(logits.detach().float()).cpu().numpy()
    rows, cols = prob.shape[1:]
    points = np.zeros((N_KEYPOINTS, 3), dtype=float)
    for k in range(N_KEYPOINTS):
        r, c = np.unravel_index(int(prob[k].argmax()), prob[k].shape)
        dx = dy = 0.0
        if 0 < c < cols - 1:
            dx = 0.25 * np.sign(prob[k, r, c + 1] - prob[k, r, c - 1])
        if 0 < r < rows - 1:
            dy = 0.25 * np.sign(prob[k, r + 1, c] - prob[k, r - 1, c])
        points[k] = ((c + 0.5 + dx) / cols * width, (r + 0.5 + dy) / rows * height, float(prob[k, r, c]))
    return points


class RinkKeypointDetector:
    """Stage 1's model. conf is the heatmap's peak value: near 1 for a visible keypoint, near 0 for an absent one."""

    name = "resnet50-heatmap-rink33"

    def __init__(self, weights_path, threshold: float = 0.3, device: str | None = None, input_hw=INPUT_HW):
        self.weights_path = str(weights_path)
        self.threshold = threshold
        self.input_hw = tuple(input_hw)
        self.device = pick_device(device)
        self._model, self.meta = load_model(weights_path, self.device)

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        logits = self._model(image_tensor(image_bgr, self.input_hw)[None].to(self.device))[0]
        points = decode(logits, width, height)
        points[points[:, 2] < self.threshold] = 0.0
        return points
