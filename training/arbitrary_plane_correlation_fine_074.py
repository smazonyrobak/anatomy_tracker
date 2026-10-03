"""Local complete-pose refinement with the trained whole-plane correlation head."""
import math

import torch
import torch.nn.functional as F


def refine_one(head, query_features, mask_logits, atlas_features, atlas_support,
               base_angle, base_scale, base_tx, base_ty, normal_offsets_um,
               return_scores=False):
    """Return (score, dz_index, roll_rad, scale, tx, ty) for one coarse pose.

    ``atlas_features`` and ``atlas_support`` contain the five rendered normal
    offsets in ``normal_offsets_um`` order. Translations are atlas-chart units.
    """
    device, dtype = query_features.device, query_features.dtype
    angles = torch.as_tensor(base_angle, device=device, dtype=dtype) + torch.arange(-3, 4, device=device, dtype=dtype) * (math.pi / 36)
    scales = torch.as_tensor(base_scale, device=device, dtype=dtype) * torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2], device=device, dtype=dtype)
    roll = angles.repeat_interleave(5)
    scale = scales.repeat(7)
    theta = torch.zeros(35, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 4 * roll.cos() / scale
    theta[:, 0, 1] = 4 * roll.sin() / scale
    theta[:, 1, 0] = -4 * roll.sin() / scale
    theta[:, 1, 1] = 4 * roll.cos() / scale
    grid = F.affine_grid(theta, (35, 1, 64, 64), align_corners=False)
    query = F.grid_sample(query_features.expand(35, -1, -1, -1), grid, align_corners=False)
    mask = F.grid_sample(mask_logits.sigmoid().expand(35, -1, -1, -1), grid, align_corners=False).float()
    query = F.normalize(query.float(), dim=1) * mask
    atlas = F.normalize(head.atlas_projection(atlas_features).float(), dim=1)
    support = atlas_support.float()

    query_fft = torch.fft.rfft2(query, s=(128, 128))
    atlas_fft = torch.fft.rfft2(atlas * support, s=(128, 128))
    support_fft = torch.fft.rfft2(support, s=(128, 128))
    mask_fft = torch.fft.rfft2(mask, s=(128, 128))
    numerator = torch.fft.irfft2(torch.einsum('ochw,nchw->nohw', query_fft.conj(), atlas_fft), s=(128, 128))
    overlap = torch.fft.irfft2(mask_fft.conj()[None] * support_fft[:, None], s=(128, 128)).squeeze(2)

    tx = torch.as_tensor(base_tx, device=device, dtype=dtype) + torch.arange(-6, 7, device=device, dtype=dtype) * 0.025
    ty = torch.as_tensor(base_ty, device=device, dtype=dtype) + torch.arange(-6, 7, device=device, dtype=dtype) * 0.025
    xpixel, ypixel = tx.float() * 16, ty.float() * 16
    xfloor, yfloor = xpixel.floor(), ypixel.floor()
    x0, y0 = xfloor.long().remainder(128), yfloor.long().remainder(128)
    x1, y1 = (x0 + 1).remainder(128), (y0 + 1).remainder(128)
    wx = (xpixel - xfloor)[None, None, None, None, :]
    wy = (ypixel - yfloor)[None, None, None, :, None]
    maps = torch.stack((numerator, overlap))
    values = (1 - wy) * ((1 - wx) * maps[..., y0[:, None], x0[None, :]] + wx * maps[..., y0[:, None], x1[None, :]]) + wy * ((1 - wx) * maps[..., y1[:, None], x0[None, :]] + wx * maps[..., y1[:, None], x1[None, :]])
    numerator, overlap = values[0], values[1].clamp_min(0)
    mass = mask.sum((-2, -1)).view(1, 35, 1, 1)
    fraction = (overlap / (mass + 1e-4)).clamp(1e-4, 1)
    scores = (numerator / (overlap + 1e-4) + 0.3 * fraction.log()) * head.logit_scale.exp()
    scores = scores.reshape(len(normal_offsets_um), 7, 5, 13, 13)
    best = scores.argmax().item()
    dz, a, s, iy, ix = (int(v) for v in torch.unravel_index(torch.tensor(best), scores.shape))
    result = (scores.flatten()[best].item(), dz, (angles[a].item() % (2 * math.pi)),
              scales[s].item(), tx[ix].item(), ty[iy].item())
    return (result, scores) if return_scores else result
