"""Independent arbitrary-plane v4 sections with modest correlated tissue grain."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import (
    sample_streaming_synthetic_v7_appearance_v3,
    sample_streaming_synthetic_v7_64_appearance_v3,
)


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_one_shot_slide_artifacts_v5(context, subject_indices, seed, side=192):
    sampler = (sample_streaming_synthetic_v7_64_appearance_v3 if 'displacement_cpu' in context['bases'][0]
               else sample_streaming_synthetic_v7_appearance_v3)
    source = sampler(context, subject_indices, seed, side)
    inputs = source['inputs'].clone()
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 31]))
        sigma = float(rng.uniform(.55, 1.25))
        amplitude = float(rng.uniform(.008, .055))
        noise_seed = int(rng.integers(0, 2**63))
        noise = torch.randn((1, 1, side, side), generator=torch.Generator().manual_seed(noise_seed)).to(inputs.device)
        axis = torch.arange(-3, 4, device=inputs.device, dtype=inputs.dtype)
        kernel = torch.exp(-axis.square() / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel[:, None] * kernel[None]
        grain = F.conv2d(noise, kernel[None, None], padding=3)
        grain = grain - F.avg_pool2d(grain, 9, 1, 4)
        tissue = source['visible'][row].clamp(0, 1)
        scale = grain.square().mean().sqrt()
        inputs[row, 0] = (inputs[row, 0] + amplitude * grain[0, 0] / scale * tissue).clamp(0, 1)
        source['provenance'][row]['one_shot_tissue_grain_v5'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 31, 'noise_seed': noise_seed,
            'sigma_px': sigma, 'pre_exposure_rms': amplitude,
            'basis': 'spatially correlated high-pass grain; range motivated by 119 TRAIN-only tissue texture gap',
        }
    return sample_one_shot_slide_artifacts_v4(
        context, subject_indices, seed, side, source_sample={**source, 'inputs': inputs})
