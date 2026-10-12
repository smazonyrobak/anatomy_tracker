"""Exact synthetic physical-point supervision for the oriented-slab ranker."""

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components


def point_loss_300(ranker, sample, true_bank, negatives, support_only=False):
    image = F.avg_pool2d(sample['inputs'], 4)
    source = F.normalize(ranker['source'](image), dim=1)[0]
    support = true_bank[:, :, 1:2].clamp(0, 1)
    intensity = true_bank[:, :, :1] / support.clamp_min(1e-4)
    if support_only:
        intensity = torch.zeros_like(intensity)
    atlas = F.normalize(ranker['atlas'](torch.cat((intensity, support), 2)[0]), dim=1)
    atlas = atlas.permute(1, 0, 2, 3)[None]
    support = support[0].permute(1, 0, 2, 3)[None]
    yy, xx = torch.meshgrid(torch.arange(3, 48, 6, device=image.device),
                            torch.arange(3, 48, 6, device=image.device), indexing='ij')
    yy, xx = yy.flatten(), xx.flatten()
    query = source[:, yy, xx].T
    intact = sample['valid_mask'] & ~sample['fragment_displaced_mask']
    valid = (intact[0, 4 * yy + 2, 4 * xx + 2]
             & (F.avg_pool2d(intact.float()[:, None], 4)[0, 0, yy, xx] > .5))
    world = sample['centre'][0, 4 * yy + 2, 4 * xx + 2]
    centre, frame, basis = full_frame_state_to_components(sample['state'])
    local = (world - centre[0]) @ frame[0]
    chart = torch.linalg.solve(basis[0], local[:, :2].T).T + .5
    x = torch.where(sample['reflection'][0].bool(), 191 / 192 - chart[:, 0], chart[:, 0])
    position = torch.stack(((x * 192 - 1.5) / 4,
        (chart[:, 1] * 192 - 1.5) / 4, local[:, 2] / 250 + 2), -1)
    trial = torch.cat((position[:, None], position[:, None] + negatives), 1)
    grid = (trial / trial.new_tensor((47., 47., 4.))) * 2 - 1
    grid = grid.reshape(1, 1, 1, -1, 3)
    key = F.grid_sample(atlas, grid, align_corners=True).reshape(32, 64, -1).permute(1, 2, 0)
    available = F.grid_sample(support, grid, align_corners=True).reshape(64, -1)
    logits = 16 * (query[:, None] * F.normalize(key, dim=-1)).sum(-1)
    logits = torch.cat((logits[:, :1],
        logits[:, 1:].masked_fill(available[:, 1:] <= .5, -1e4)), 1)
    usable = valid & (available[:, 0] > .9) & (available[:, 1:] > .5).sum(-1).ge(4)
    if bool(usable.any()):
        return F.cross_entropy(logits[usable],
            torch.zeros(int(usable.sum()), device=image.device, dtype=torch.long)), int(usable.sum())
    return logits.sum() * 0, 0
