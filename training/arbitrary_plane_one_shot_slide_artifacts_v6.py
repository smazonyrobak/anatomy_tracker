"""Independent arbitrary-plane sections with a localized stretch/compression tail."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import (
    sample_streaming_synthetic_v7_64_appearance_v3,
    sample_streaming_synthetic_v7_appearance_v3,
)


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_one_shot_slide_artifacts_v6(context, subject_indices, seed, side=192):
    sampler = (sample_streaming_synthetic_v7_64_appearance_v3
               if 'displacement_cpu' in context['bases'][0]
               else sample_streaming_synthetic_v7_appearance_v3)
    source = sampler(context, subject_indices, seed, side)
    device = source['inputs'].device
    axis = torch.linspace(-1, 1, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grids, parameters = [], []
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 30]))
        active = bool(rng.random() < .8)
        cx, cy = rng.uniform(-.45, .45, 2)
        sx, sy = rng.uniform(.13, .32, 2)
        angle = float(rng.uniform(-np.pi, np.pi))
        radial = float(rng.uniform(-.65, .65)) if active else 0.
        transverse = float(np.clip(radial * rng.uniform(.35, 1.2), -.65, .65))
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        envelope = torch.exp(-.5 * ((u / sx).square() + (v / sy).square()))
        du, dv = radial * u * envelope, transverse * v * envelope
        grids.append(torch.stack((xx + du * np.cos(angle) - dv * np.sin(angle),
                                  yy + du * np.sin(angle) + dv * np.cos(angle)), -1))
        parameters.append({'active': active, 'centre_xy': [float(cx), float(cy)],
            'radius_xy': [float(sx), float(sy)], 'angle_rad': angle,
            'radial_strain': radial, 'transverse_strain': transverse})
    grid = torch.stack(grids)
    fields = torch.cat((source['inputs'][:, :1], source['centre'].permute(0, 3, 1, 2),
        source['visible'][:, None], source['support'][:, None],
        source['masks'][:, None].float()), 1)
    deformed = F.grid_sample(fields, grid, padding_mode='zeros', align_corners=True)
    source = {**source, 'inputs': source['inputs'].clone(),
        'centre': deformed[:, 1:4].permute(0, 2, 3, 1),
        'visible': deformed[:, 4], 'support': deformed[:, 5],
        'masks': deformed[:, 6] > .5}
    source['inputs'][:, 0] = deformed[:, 0]
    sample = sample_one_shot_slide_artifacts_v4(
        context, subject_indices, seed, side=side, source_sample=source)
    composed = F.grid_sample(grid.permute(0, 3, 1, 2),
        sample['observed_to_source_grid'], padding_mode='zeros',
        align_corners=True).permute(0, 2, 3, 1)
    for row, record in enumerate(sample['provenance']):
        record['one_shot_slide_artifacts_v6'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 30,
            'source_plane_reused_for_appearance_pair': False,
            'localized_strain': parameters[row],
            'basis': 'independent Gaussian-localized tissue stretch/compression before v3 damage and v4 exposure; not calibrated to real deformation prevalence'}
    return {**sample, 'observed_to_source_grid': composed,
            'localized_strain_grid': grid}
