"""Raw pixel-centre marks sampled on retained observed native CCF surfaces."""

import torch
import torch.nn.functional as F


def raw_marks_to_ribbon_ccf(
    raw_points_xy, raw_to_model_xy, observed_centre_ccf, *,
    raw_shape_h_w, mark_available=None,
):
    """Map B,P,2 raw xy marks to B,K,R,P,3 physical CCF AP/DV/ML micrometres.

    raw_to_model_xy is B,3,3: the recorded homogeneous affine on column vectors
    [x,y,1], with last row [0,0,1]. raw_shape_h_w is H,W or B,2. The surface is
    B,K,R,H,W,3 from final_observed_centre_ccf_ap_dv_ml_um. Its reflection and
    deformation are ALREADY applied; do not apply another flip, warp or O/U/V.
    Every retained cell/representation remains separate; no probability is used.

    GUI ProbeTrace.slice_points use unrotated downsampled raw_display pixel
    centres, not original-resolution pixels. The affine must map that exact
    raster into the actual prepared model input, including its resize/crop and
    any chosen display orientation. display_scale alone is not this transform.

    The surface values already encode the physical x/W,y/H chart. Sampling its
    pixel centres instead uses 2*(xy+.5)/[W,H]-1 with align_corners=False.
    Only the tabulated centre hull 0..W-1,0..H-1 is interpolated: no extrapolation
    into its half-pixel fringe, no border clamping. Invalid/unavailable marks or
    nonfinite sampled CCF values return NaN and valid_mask=False. This is raster
    validity, not a tissue mask, atlas-label validity or localization confidence.
    """
    surface = observed_centre_ccf
    batch, cells, representations, height, width, _ = surface.shape
    points = torch.as_tensor(raw_points_xy, device=surface.device, dtype=surface.dtype)
    affine = torch.as_tensor(raw_to_model_xy, device=surface.device, dtype=surface.dtype)
    raw_size = torch.as_tensor(raw_shape_h_w, device=surface.device, dtype=surface.dtype).reshape(-1, 2).flip(-1)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    model_xy = (homogeneous @ affine.transpose(-1, -2))[..., :2]
    valid = (
        torch.isfinite(points).all(-1)
        & torch.isfinite(affine).all((-2, -1))[:, None]
        & torch.isfinite(model_xy).all(-1)
        & (points >= 0).all(-1) & (points <= raw_size[:, None] - 1).all(-1)
        & (model_xy >= 0).all(-1)
        & (model_xy <= model_xy.new_tensor((width - 1, height - 1))).all(-1)
    )
    if mark_available is not None:
        valid = valid & torch.as_tensor(mark_available, device=surface.device, dtype=torch.bool)
    grid = 2 * (model_xy + .5) / model_xy.new_tensor((width, height)) - 1
    grid = torch.where(valid[..., None], grid, 0)
    count = points.shape[1]
    grid = grid[:, None, None, :, None].expand(-1, cells, representations, -1, -1, -1)
    sampled = F.grid_sample(
        surface.permute(0, 1, 2, 5, 3, 4).reshape(-1, 3, height, width),
        grid.reshape(-1, count, 1, 2), mode="bilinear", padding_mode="zeros",
        align_corners=False,
    ).squeeze(-1).transpose(1, 2).reshape(batch, cells, representations, count, 3)
    valid = valid[:, None, None] & torch.isfinite(sampled).all(-1)
    return {
        "points_ap_dv_ml_um": sampled.masked_fill(~valid[..., None], torch.nan),
        "model_points_xy_px": model_xy,
        "valid_mask": valid,
    }
