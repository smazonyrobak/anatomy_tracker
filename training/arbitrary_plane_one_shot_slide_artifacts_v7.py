"""Independent v6 sections with fluorescence-compatible brightness and backgrounds."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v6 import sample_one_shot_slide_artifacts_v6

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_one_shot_slide_artifacts_v7(context, subject_indices, seed, side=256):
    sample = sample_one_shot_slide_artifacts_v6(context, subject_indices, seed, side)
    inputs = sample['inputs'].clone()
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 32]))
        target_p99 = float(rng.uniform(.95, 1.05) if rng.random() < .65 else
                           rng.uniform(.45, .95) if rng.random() < .75 else
                           rng.uniform(.15, .45))
        contrast_gamma = float(rng.uniform(1.25, 2.1))
        image = inputs[row, 0]
        image = ((image / torch.quantile(image.flatten(), .99).clamp_min(.03))
                 .clamp(0, 1).pow(contrast_gamma) * target_p99).clamp(0, 1)
        mode = sample['provenance'][row]['mode']
        background_level = 0.
        background_seed = None
        if mode == 'raw':
            kind = rng.choice(3, p=(.10, .75, .15))
            if kind:
                background_level = float(rng.uniform(.01, .16) if kind == 1
                                         else rng.uniform(.16, .55))
                background_seed = int(rng.integers(0, 2**63))
                generator = torch.Generator().manual_seed(background_seed)
                coarse = torch.randn((1, 1, 8, 8), generator=generator).to(image.device)
                fine = torch.randn((side, side), generator=generator).to(image.device)
                texture = F.interpolate(coarse, (side, side), mode='bicubic',
                                        align_corners=False)[0, 0]
                background = (background_level * (1 + .18 * texture) +
                              fine * (.002 + .025 * background_level)).clamp(0, 1)
                image = torch.where(sample['visible'][row] > .05, image, background)
            else:
                image = torch.where(sample['visible'][row] > .05, image, 0.)
        elif mode == 'exact_black':
            image = torch.where(sample['visible'][row] > .05, image, 0.)
        inputs[row, 0] = image
        sample['provenance'][row]['one_shot_slide_artifacts_v7'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 32,
            'appearance_source': context.get('appearance_source', 'Allen average_template_25'),
            'target_whole_image_p99_before_background': target_p99,
            'contrast_gamma': contrast_gamma,
            'raw_background_level': background_level, 'background_seed': background_seed,
            'basis': 'bounded fluorescence TRAIN appearance audit 195; independent per physical draw',
        }
    return {**sample, 'inputs': inputs}
