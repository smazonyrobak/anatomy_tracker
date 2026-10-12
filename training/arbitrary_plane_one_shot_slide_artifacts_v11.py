"""Independent annotation-coherent tissue contrast before v3 damage and v9 exterior."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_RAW_SHA256_V6
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import (
    sample_streaming_synthetic_v7_64_appearance_v3,
)


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXTERIOR_STYLES = ('exact_black', 'near_black', 'gray')
EXTERIOR_PROBS = (.45, .30, .25)


def sample_one_shot_slide_artifacts_v11(context, annotation, subject_indices, seed, side=192):
    source = sample_streaming_synthetic_v7_64_appearance_v3(context, subject_indices, seed, side)
    source = {**source, 'inputs': source['inputs'].clone()}
    device = source['inputs'].device
    annotation = annotation.to(device)
    shape = torch.tensor(annotation.shape, device=device)
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 37]))
        index = (source['centre'][row] / 25 - .5).round().long()
        inside = ((index >= 0) & (index < shape)).all(-1)
        index = torch.minimum(torch.maximum(index, torch.zeros_like(index)), shape - 1)
        labels = annotation[index[..., 0], index[..., 1], index[..., 2]]
        tissue = (source['visible'][row] > .05) & inside & (labels > 0)
        unique, inverse = torch.unique(labels[tissue], sorted=True, return_inverse=True)
        region_codes = rng.normal(0, .65, len(unique)).astype(np.float32)
        region_field = torch.zeros((side, side), device=device)
        region_field[tissue] = torch.from_numpy(region_codes).to(device)[inverse]
        coverage = F.avg_pool2d(tissue[None, None].float(), 5, 1, 2)[0, 0]
        region_field = (F.avg_pool2d(region_field[None, None], 5, 1, 2)[0, 0]
                        / coverage.clamp_min(.1))
        contrast_strength = float(rng.uniform(.25, .8))
        texture_strength = float(rng.uniform(.012, .075))
        texture_seed = int(rng.integers(0, 2**63))
        white = torch.randn((1, 1, side, side),
                            generator=torch.Generator().manual_seed(texture_seed)).to(device)
        correlated = F.avg_pool2d(F.pad(white, (1, 1, 1, 1), mode='reflect'), 3, 1)[0, 0]
        texture = .25 * white[0, 0] + .75 * correlated / correlated.std().clamp_min(.1)
        image = source['inputs'][row, 0]
        styled = (image * (contrast_strength * region_field).exp().clamp(.3, 3)
                  + texture_strength * texture * image.clamp_min(.05).sqrt()).clamp(0, 1)
        source['inputs'][row, 0] = torch.where(tissue, styled, image)
        source['provenance'][row]['one_shot_slide_artifacts_v11'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 37,
            'annotation_raw_sha256': ANNOTATION_RAW_SHA256_V6,
            'region_code_count': len(unique),
            'region_code_sha256': hashlib.sha256(region_codes.tobytes()).hexdigest(),
            'contrast_strength': contrast_strength, 'texture_strength': texture_strength,
            'texture_seed': texture_seed,
            'source_tissue_label_fraction': float(tissue.float().sum() /
                                                  (source['visible'][row] > .05).sum().clamp_min(1)),
            'sampling': 'new region codes and correlated texture for this one physical plane only',
            'central_label_approximation': 'nearest CCF annotation at canonical slab centre, not nine-tap label PSF',
        }

    undamaged_image = source['inputs'][:, 0].clone()
    sample = sample_one_shot_slide_artifacts_v7(context, subject_indices, seed, side,
                                                 source_sample=source)
    inputs = sample['inputs'].clone()
    axis = torch.linspace(-1, 1, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    for row, subject_index in enumerate(subject_indices):
        brush = bool(inputs[row, 2, 0, 0])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 34]))
        style = 'exact_black_brush' if brush else str(rng.choice(EXTERIOR_STYLES, p=EXTERIOR_PROBS))
        level, slope_y, slope_x, noise_seed = 0., 0., 0., None
        if brush:
            inputs[row, 0] *= sample['brush_mask'][row]
        else:
            if style == 'exact_black':
                background = torch.zeros((side, side), device=device, dtype=inputs.dtype)
            else:
                level = float(rng.uniform(.003, .018) if style == 'near_black' else rng.uniform(.03, .35))
                slope_y, slope_x = rng.uniform(-.3, .3, 2).tolist()
                noise_seed = int(rng.integers(0, 2**63))
                generator = torch.Generator().manual_seed(noise_seed)
                coarse = torch.randn((1, 1, 8, 8), generator=generator).to(device)
                fine = torch.randn((side, side), generator=generator).to(device)
                texture = F.interpolate(coarse, (side, side), mode='bicubic', align_corners=False)[0, 0]
                background = (level * (1 + slope_y * yy + slope_x * xx + .15 * texture)
                              + fine * (.0005 if style == 'near_black' else .002 + .025 * level)).clamp(0, 1)
            inputs[row, 0] = torch.where(sample['visible'][row] <= .05, background, inputs[row, 0])
        sample['provenance'][row]['one_shot_slide_artifacts_v11']['exterior'] = {
            'seed_branch': 34, 'style': style, 'level': level,
            'slope_yx': [slope_y, slope_x], 'noise_seed': noise_seed,
            'brush_exact_zero_outside': brush,
        }
    return {**sample, 'inputs': inputs, 'undamaged_image': undamaged_image}
