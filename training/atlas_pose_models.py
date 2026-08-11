from __future__ import annotations

import timm
import torch
from torch import nn


MODEL_SPECS = {
    "xception": ("legacy_xception.tf_in1k", nn.ReLU),
    "efficientnetv2": ("tf_efficientnetv2_s.in21k_ft_in1k", nn.SiLU),
    "convnext": ("convnext_tiny.fb_in22k_ft_in1k", nn.GELU),
}


class AtlasPoseRegressor(nn.Module):
    def __init__(self, architecture: str, pretrained: bool = True):
        super().__init__()
        model_name, activation = MODEL_SPECS[architecture]
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            activation(),
            nn.Dropout(0.20),
            nn.Linear(256, 256),
            activation(),
            nn.Dropout(0.10),
            nn.Linear(256, 3),
        )
        self.orientation_head = nn.Linear(self.backbone.num_features, 1)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone((image - self.input_mean) / self.input_std)
        pose = self.head(features)
        orientation_logit = self.orientation_head(features).squeeze(1)
        sign = torch.where(orientation_logit > 0.0, -1.0, 1.0)[:, None]
        return torch.cat((pose[:, :1], pose[:, 1:] * sign), dim=1)

    def forward_with_orientation(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone((image - self.input_mean) / self.input_std)
        return self.head(features), self.orientation_head(features).squeeze(1)


class PhysicalPoseOutput(nn.Module):
    def __init__(self, model: AtlasPoseRegressor, target_center: torch.Tensor, target_scale: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("target_center", target_center)
        self.register_buffer("target_scale", target_scale)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pose, orientation_logit = self.model.forward_with_orientation(image)
        orientation_sign = torch.where(orientation_logit > 0.0, -1.0, 1.0)[:, None]
        pose = torch.cat((pose[:, :1], pose[:, 1:] * orientation_sign), dim=1)
        return pose * self.target_scale + self.target_center, orientation_logit


def set_backbone_trainable(model: AtlasPoseRegressor, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
