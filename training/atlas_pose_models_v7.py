from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


BACKBONES = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "maxvit_tiny": "maxvit_tiny_rw_224.sw_in1k",
    "xception": "legacy_xception.tf_in1k",
}

AP_MIN_UM = -4500.0
AP_MAX_UM = 500.0
AP_STEP_UM = 25.0
AP_BIN_COUNT = 201
TILT_MIN_DEG = -35.0
TILT_MAX_DEG = 35.0
TILT_STEP_DEG = 1.0
TILT_BIN_COUNT = 71
ORIENTATION_LOSS_WEIGHT = 0.35
FINAL_PHYSICAL_TILT_LOSS_WEIGHT = 0.15
BINNED_AUXILIARY_LOSS_WEIGHT = 0.10
OUV_AUXILIARY_LOSS_WEIGHT = 1.00
PHYSICAL_POSE_LOSS_SCALE = (60.0, 2.0, 2.0)

VOXEL_UM = 25.0
BREGMA_AP_INDEX = 216.0
QUICKNII_ML_AP_DV_SHAPE = (456.0, 528.0, 320.0)
ATLAS_CENTER_ML_DV = (227.5, 159.5)
QUICKNII_OUV_CENTER = (0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0)
QUICKNII_OUV_SCALE = (456.0, 256.0, 320.0, 456.0, 320.0, 320.0, 456.0, 224.0, 320.0)
DIRECT_POSE_CENTER = (-2000.0, 0.0, 0.0)
DIRECT_POSE_SCALE = (2500.0, 20.0, 20.0)


def ap_bin_centers(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(AP_MIN_UM, AP_MAX_UM, AP_BIN_COUNT, dtype=dtype)


def ap_bin_edges(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(
        AP_MIN_UM - AP_STEP_UM / 2.0,
        AP_MAX_UM + AP_STEP_UM / 2.0,
        AP_BIN_COUNT + 1,
        dtype=dtype,
    )


def tilt_bin_centers(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(TILT_MIN_DEG, TILT_MAX_DEG, TILT_BIN_COUNT, dtype=dtype)


def tilt_bin_edges(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(
        TILT_MIN_DEG - TILT_STEP_DEG / 2.0,
        TILT_MAX_DEG + TILT_STEP_DEG / 2.0,
        TILT_BIN_COUNT + 1,
        dtype=dtype,
    )


def encode_binned_target(
    target: torch.Tensor,
    centers: torch.Tensor,
    step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.round((target - centers[0]) / step).long().clamp(0, centers.numel() - 1)
    residual = ((target - centers[index]) / (step / 2.0)).clamp(-1.0, 1.0)
    return index, residual


def decode_binned_prediction(
    logits: torch.Tensor,
    residual_logits: torch.Tensor,
    centers: torch.Tensor,
    step: float,
) -> torch.Tensor:
    corrected_centers = centers + torch.tanh(residual_logits) * (step / 2.0)
    return (torch.softmax(logits, dim=-1) * corrected_centers).sum(dim=-1)


def binned_axis_loss(
    logits: torch.Tensor,
    residual_logits: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    step: float,
) -> torch.Tensor:
    target_index, target_residual = encode_binned_target(target, centers, step)
    selected_residual = torch.tanh(residual_logits.gather(1, target_index[:, None]).squeeze(1))
    return (
        F.cross_entropy(logits, target_index) / math.log(logits.shape[-1])
        + F.smooth_l1_loss(selected_residual, target_residual, beta=0.25)
    )


def pose_to_quicknii_ouv(pose: torch.Tensor) -> torch.Tensor:
    ap_um, tilt_lr_deg, tilt_dv_deg = pose.unbind(dim=-1)
    slope_lr = torch.tan(tilt_lr_deg * (math.pi / 180.0))
    slope_dv = torch.tan(tilt_dv_deg * (math.pi / 180.0))
    ap_index = BREGMA_AP_INDEX - ap_um / VOXEL_UM
    origin_ap = (
        ap_index
        + slope_lr * (QUICKNII_ML_AP_DV_SHAPE[0] - ATLAS_CENTER_ML_DV[0])
        - slope_dv * ATLAS_CENTER_ML_DV[1]
    )
    zeros = torch.zeros_like(ap_um)
    return torch.stack(
        (
            zeros,
            QUICKNII_ML_AP_DV_SHAPE[1] - origin_ap,
            zeros + QUICKNII_ML_AP_DV_SHAPE[2],
            zeros + QUICKNII_ML_AP_DV_SHAPE[0],
            QUICKNII_ML_AP_DV_SHAPE[0] * slope_lr,
            zeros,
            zeros,
            -QUICKNII_ML_AP_DV_SHAPE[2] * slope_dv,
            zeros - QUICKNII_ML_AP_DV_SHAPE[2],
        ),
        dim=-1,
    )


def quicknii_ouv_to_pose(ouv: torch.Tensor) -> torch.Tensor:
    origin = ouv[..., :3]
    normal = torch.cross(ouv[..., 3:6], ouv[..., 6:9], dim=-1)
    normal = torch.where((normal[..., 1] < 0.0)[..., None], -normal, normal)
    denominator = normal[..., 1].clamp_min(1e-8)
    ap_per_ml = -normal[..., 0] / denominator
    ap_per_dv = -normal[..., 2] / denominator
    origin_ml = QUICKNII_ML_AP_DV_SHAPE[0] - origin[..., 0]
    origin_ap = QUICKNII_ML_AP_DV_SHAPE[1] - origin[..., 1]
    origin_dv = QUICKNII_ML_AP_DV_SHAPE[2] - origin[..., 2]
    ap_index = (
        origin_ap
        + ap_per_ml * (ATLAS_CENTER_ML_DV[0] - origin_ml)
        + ap_per_dv * (ATLAS_CENTER_ML_DV[1] - origin_dv)
    )
    return torch.stack(
        (
            (BREGMA_AP_INDEX - ap_index) * VOXEL_UM,
            torch.atan(ap_per_ml) * (180.0 / math.pi),
            torch.atan(ap_per_dv) * (180.0 / math.pi),
        ),
        dim=-1,
    )


class SpatialEncoder(nn.Module):
    def __init__(self, architecture: str = "convnext_tiny", pretrained: bool = False):
        super().__init__()
        if architecture not in BACKBONES:
            raise ValueError(f"Unknown AtlasPose backbone: {architecture}")
        self.backbone = timm.create_model(
            BACKBONES[architecture],
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        self.num_features = self.backbone.feature_info.channels()[-1]
        self.input_adapter = (
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
            if architecture == "maxvit_tiny"
            else nn.Identity()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_adapter(image))[-1]


class SpatialPyramidPoseFeatures(nn.Module):
    def __init__(self, feature_count: int, reduced_feature_count: int = 128):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(feature_count, reduced_feature_count, 1, bias=False),
            nn.GroupNorm(8, reduced_feature_count),
            nn.GELU(),
        )
        pyramid_feature_count = reduced_feature_count * (1 + 4 + 16)
        self.project = nn.Sequential(
            nn.LayerNorm(pyramid_feature_count),
            nn.Linear(pyramid_feature_count, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(feature_map)
        level_1 = reduced.mean(dim=(-2, -1))
        level_2 = F.interpolate(reduced, size=(2, 2), mode="bilinear", align_corners=False).flatten(1)
        level_4 = F.interpolate(reduced, size=(4, 4), mode="bilinear", align_corners=False).flatten(1)
        return self.project(torch.cat((level_1, level_2, level_4), dim=1))


class BinnedPoseHead(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.ap = nn.Linear(feature_count, AP_BIN_COUNT * 2)
        self.lr = nn.Linear(feature_count, TILT_BIN_COUNT * 2)
        self.dv = nn.Linear(feature_count, TILT_BIN_COUNT * 2)
        self.register_buffer("ap_centers", ap_bin_centers())
        self.register_buffer("tilt_centers", tilt_bin_centers())

    @staticmethod
    def _split(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output.chunk(2, dim=-1)

    def components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        ap_logits, ap_residuals = self._split(self.ap(features))
        lr_logits, lr_residuals = self._split(self.lr(features))
        dv_logits, dv_residuals = self._split(self.dv(features))
        pose = torch.stack(
            (
                decode_binned_prediction(ap_logits, ap_residuals, self.ap_centers, AP_STEP_UM),
                decode_binned_prediction(lr_logits, lr_residuals, self.tilt_centers, TILT_STEP_DEG),
                decode_binned_prediction(dv_logits, dv_residuals, self.tilt_centers, TILT_STEP_DEG),
            ),
            dim=-1,
        )
        return {
            "pose": pose,
            "ap_logits": ap_logits,
            "ap_residuals": ap_residuals,
            "lr_logits": lr_logits,
            "lr_residuals": lr_residuals,
            "dv_logits": dv_logits,
            "dv_residuals": dv_residuals,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ap_logits, ap_residuals = self._split(self.ap(features))
        lr_logits, lr_residuals = self._split(self.lr(features))
        dv_logits, dv_residuals = self._split(self.dv(features))
        return torch.stack(
            (
                decode_binned_prediction(ap_logits, ap_residuals, self.ap_centers, AP_STEP_UM),
                decode_binned_prediction(lr_logits, lr_residuals, self.tilt_centers, TILT_STEP_DEG),
                decode_binned_prediction(dv_logits, dv_residuals, self.tilt_centers, TILT_STEP_DEG),
            ),
            dim=-1,
        )


class OUVPoseHead(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.normalized_ouv = nn.Linear(feature_count, 9)
        nn.init.zeros_(self.normalized_ouv.bias)
        self.register_buffer("ouv_center", torch.tensor(QUICKNII_OUV_CENTER))
        self.register_buffer("ouv_scale", torch.tensor(QUICKNII_OUV_SCALE))

    def predict_ouv(self, features: torch.Tensor) -> torch.Tensor:
        return self.normalized_ouv(features) * self.ouv_scale + self.ouv_center

    def components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        ouv = self.predict_ouv(features)
        return {"pose": quicknii_ouv_to_pose(ouv), "ouv": ouv}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return quicknii_ouv_to_pose(self.predict_ouv(features))


class DirectPoseHead(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.normalized_pose = nn.Linear(feature_count, 3)
        self.register_buffer("pose_center", torch.tensor(DIRECT_POSE_CENTER))
        self.register_buffer("pose_scale", torch.tensor(DIRECT_POSE_SCALE))

    def components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized_pose = self.normalized_pose(features)
        return {
            "pose": normalized_pose * self.pose_scale + self.pose_center,
            "normalized_pose": normalized_pose,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.normalized_pose(features) * self.pose_scale + self.pose_center


class CoarseAnatomyHead(nn.Module):
    def __init__(self, feature_count: int, class_count: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_count, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, class_count, 1),
        )

    def forward(self, feature_map: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            self.decoder(feature_map),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class AtlasPoseV7(nn.Module):
    def __init__(
        self,
        architecture: str = "convnext_tiny",
        pretrained: bool = False,
        pose_representation: str = "binned",
        anatomy_class_count: int = 9,
    ):
        super().__init__()
        self.encoder = SpatialEncoder(architecture, pretrained)
        self.feature_head = SpatialPyramidPoseFeatures(self.encoder.num_features)
        if pose_representation == "binned":
            self.pose_head = BinnedPoseHead(512)
        elif pose_representation == "direct":
            self.pose_head = DirectPoseHead(512)
        elif pose_representation == "ouv":
            self.pose_head = OUVPoseHead(512)
        else:
            raise ValueError(f"Unknown AtlasPose pose representation: {pose_representation}")
        self.orientation_head = nn.Linear(512, 1)
        self.anatomy_head = CoarseAnatomyHead(self.encoder.num_features, anatomy_class_count)
        self.register_buffer("input_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("input_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encoder((image - self.input_mean) / self.input_std)
        return feature_map, self.feature_head(feature_map)

    @staticmethod
    def physical_pose(
        image_frame_pose: torch.Tensor,
        orientation_inverted_logit: torch.Tensor,
    ) -> torch.Tensor:
        orientation_sign = torch.where(
            orientation_inverted_logit > 0.0,
            -torch.ones_like(orientation_inverted_logit),
            torch.ones_like(orientation_inverted_logit),
        )
        return torch.cat(
            (
                image_frame_pose[:, :1],
                image_frame_pose[:, 1:] * orientation_sign[:, None],
            ),
            dim=1,
        )

    def forward_with_orientation(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, pooled_features = self.encode(image)
        image_frame_pose = self.pose_head(pooled_features)
        orientation_inverted_logit = self.orientation_head(pooled_features).squeeze(1)
        return self.physical_pose(image_frame_pose, orientation_inverted_logit), orientation_inverted_logit

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pose, _ = self.forward_with_orientation(image)
        return pose

    def training_outputs(
        self,
        image: torch.Tensor,
        include_anatomy: bool = True,
    ) -> dict[str, torch.Tensor]:
        feature_map, pooled_features = self.encode(image)
        outputs = self.pose_head.components(pooled_features)
        image_frame_pose = outputs["pose"]
        orientation_inverted_logit = self.orientation_head(pooled_features).squeeze(1)
        outputs["image_frame_pose"] = image_frame_pose
        outputs["pose"] = self.physical_pose(image_frame_pose, orientation_inverted_logit)
        outputs["orientation_inverted_logit"] = orientation_inverted_logit
        outputs["pooled_features"] = pooled_features
        if include_anatomy:
            outputs["anatomy_logits"] = self.anatomy_head(feature_map, image.shape[-2:])
        return outputs


class AtlasPoseV7Export(nn.Module):
    def __init__(self, model: AtlasPoseV7):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_orientation(image)


def binned_pose_loss(outputs: dict[str, torch.Tensor], target_pose: torch.Tensor) -> torch.Tensor:
    ap, lr, dv = target_pose.unbind(dim=-1)
    return torch.stack(
        (
            binned_axis_loss(
                outputs["ap_logits"],
                outputs["ap_residuals"],
                ap,
                ap_bin_centers(ap.dtype).to(ap.device),
                AP_STEP_UM,
            ),
            binned_axis_loss(
                outputs["lr_logits"],
                outputs["lr_residuals"],
                lr,
                tilt_bin_centers(lr.dtype).to(lr.device),
                TILT_STEP_DEG,
            ),
            binned_axis_loss(
                outputs["dv_logits"],
                outputs["dv_residuals"],
                dv,
                tilt_bin_centers(dv.dtype).to(dv.device),
                TILT_STEP_DEG,
            ),
        )
    ).mean()


def image_frame_pose_target(
    physical_pose: torch.Tensor,
    orientation_inverted_target: torch.Tensor,
) -> torch.Tensor:
    orientation_sign = torch.where(
        orientation_inverted_target > 0.5,
        -torch.ones_like(orientation_inverted_target),
        torch.ones_like(orientation_inverted_target),
    )
    return torch.cat(
        (
            physical_pose[:, :1],
            physical_pose[:, 1:] * orientation_sign[:, None],
        ),
        dim=1,
    )


def physical_pose_loss(image_frame_prediction: torch.Tensor, image_frame_target: torch.Tensor) -> torch.Tensor:
    error = (image_frame_prediction - image_frame_target) / image_frame_target.new_tensor(
        PHYSICAL_POSE_LOSS_SCALE
    )
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0)


def soft_physical_tilt(
    image_frame_pose: torch.Tensor,
    orientation_inverted_logit: torch.Tensor,
) -> torch.Tensor:
    """Differentiable expected physical tilt used only during training."""
    orientation_sign = 1.0 - 2.0 * torch.sigmoid(orientation_inverted_logit)
    return image_frame_pose[:, 1:] * orientation_sign[:, None]


def physical_tilt_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target) / target.new_tensor(PHYSICAL_POSE_LOSS_SCALE[1:])
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0)


def atlas_pose_v7_loss(
    outputs: dict[str, torch.Tensor],
    physical_pose_target: torch.Tensor,
    orientation_inverted_target: torch.Tensor,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    frame_target = image_frame_pose_target(physical_pose_target, orientation_inverted_target)
    frame_pose = physical_pose_loss(outputs["image_frame_pose"], frame_target)
    final_tilt = physical_tilt_loss(
        soft_physical_tilt(
            outputs["image_frame_pose"],
            outputs["orientation_inverted_logit"],
        ),
        physical_pose_target[:, 1:],
    )
    if "ap_logits" in outputs:
        auxiliary = binned_pose_loss(outputs, frame_target)
        auxiliary_weight = BINNED_AUXILIARY_LOSS_WEIGHT
    elif "ouv" in outputs:
        auxiliary = ouv_pose_loss(outputs["ouv"], frame_target)
        auxiliary_weight = OUV_AUXILIARY_LOSS_WEIGHT
    else:
        auxiliary = frame_pose.new_zeros(())
        auxiliary_weight = 0.0
    orientation_loss = F.binary_cross_entropy_with_logits(
        outputs["orientation_inverted_logit"],
        orientation_inverted_target,
    )
    weighted_auxiliary = auxiliary_weight * auxiliary
    weighted_orientation = ORIENTATION_LOSS_WEIGHT * orientation_loss
    weighted_final_tilt = FINAL_PHYSICAL_TILT_LOSS_WEIGHT * final_tilt
    total = frame_pose + weighted_final_tilt + weighted_auxiliary + weighted_orientation
    if return_components:
        return total, {
            "image_frame_pose": frame_pose,
            "soft_physical_tilt": final_tilt,
            "weighted_soft_physical_tilt": weighted_final_tilt,
            "representation_auxiliary": auxiliary,
            "weighted_representation_auxiliary": weighted_auxiliary,
            "orientation": orientation_loss,
            "weighted_orientation": weighted_orientation,
        }
    return total


def ouv_pose_loss(ouv_prediction: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    center = ouv_prediction.new_tensor(QUICKNII_OUV_CENTER)
    scale = ouv_prediction.new_tensor(QUICKNII_OUV_SCALE)
    target_ouv = pose_to_quicknii_ouv(target_pose)
    return F.smooth_l1_loss((ouv_prediction - center) / scale, (target_ouv - center) / scale, beta=0.05)
