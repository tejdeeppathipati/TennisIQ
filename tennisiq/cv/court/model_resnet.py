"""
ResNet50-based court keypoint regression model.

Takes a single RGB image and directly regresses 14 keypoint (x, y) coordinates
(28 output values) instead of producing heatmaps.

Architecture: torchvision ResNet50 with final FC layer replaced (2048 → 28).
Weights: keypoints_model.pth
"""
import torch
import torchvision.models as models


def build_court_resnet(num_keypoints: int = 14) -> torch.nn.Module:
    """Build a ResNet50 with modified FC for keypoint regression."""
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_keypoints * 2)
    return model
