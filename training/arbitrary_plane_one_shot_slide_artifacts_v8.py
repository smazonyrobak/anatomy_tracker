"""V7 sections with independently sampled black, near-black, or gray exteriors."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXTERIOR_STYLES = ('exact_black', 'near_black', 'gray')
EXTERIOR_PROBS = (.45, .30, .25)


def sample_one_shot_slide_artifacts_v8(context, subject_indices, seed, side=192):
    sample = sample_one_shot_slide_artifacts_v7(context, subject_indices, seed, side)
    inputs = sample['inputs'].clone()
    axis = torch.linspace(-1, 1, side, device=inputs.device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 33]))
        style = str(rng.choice(EXTERIOR_STYLES, p=EXTERIOR_PROBS))
        level, slope_y, slope_x, noise_seed = 0., 0., 0., None
        if style == 'exact_black':
            background = torch.zeros((side, side), device=inputs.device, dtype=inputs.dtype)
        else:
            level = float(rng.uniform(.003, .018) if style == 'near_black' else rng.uniform(.03, .35))
            slope_y, slope_x = rng.uniform(-.3, .3, 2).tolist()
            noise_seed = int(rng.integers(0, 2**63))
            generator = torch.Generator().manual_seed(noise_seed)
            coarse = torch.randn((1, 1, 8, 8), generator=generator).to(inputs.device)
            fine = torch.randn((side, side), generator=generator).to(inputs.device)
            texture = F.interpolate(coarse, (side, side), mode='bicubic', align_corners=False)[0, 0]
            background = (level * (1 + slope_y * yy + slope_x * xx + .15 * texture)
                          + fine * (.0005 if style == 'near_black' else .002 + .025 * level)).clamp(0, 1)
        exterior = sample['visible'][row] <= .05
        inputs[row, 0] = torch.where(exterior, background, inputs[row, 0])
        record = sample['provenance'][row]
        record['one_shot_slide_artifacts_v8'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 33,
            'source_mode': record['mode'], 'exterior_style': style,
            'exterior_probabilities': dict(zip(EXTERIOR_STYLES, EXTERIOR_PROBS)),
            'background_level': level, 'background_slope_yx': [slope_y, slope_x],
            'background_noise_seed': noise_seed,
            'brush_outline_available': bool(inputs[row, 2, 0, 0]),
            'background_replacement': 'only source-visible <= 0.05; CCF targets and valid mask unchanged',
            'sampling': 'one independently drawn v7 physical plane per row; exterior independent of source mode; no appearance pairing',
            'calibration': 'broad TRAIN-only mixture, not measured laboratory background or artifact prevalence',
        }
    return {**sample, 'inputs': inputs}
