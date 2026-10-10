"""Independent v3 planes with acquired-brightfield-compatible exposure variation."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_one_shot_slide_artifacts_v4(context, subject_indices, seed, side=192, source_sample=None):
    sample = sample_one_shot_slide_artifacts_v3(context, subject_indices, seed, side,
                                                source_sample=source_sample)
    inputs = sample['inputs'].clone()
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 29]))
        low = bool(rng.random() < .55)
        exposure = float(np.exp(rng.uniform(np.log(.07 if low else .30),
                                            np.log(.27 if low else 1.20))))
        noise_std = float(rng.uniform(.002, .008) if low else rng.uniform(0, .003))
        noise_seed = int(rng.integers(0, 2**63))
        noise = torch.randn((side, side), generator=torch.Generator().manual_seed(noise_seed),
                            dtype=inputs.dtype).to(inputs.device)
        tissue = sample['visible'][row] > .05
        inputs[row, 0] = (inputs[row, 0] * exposure + noise * noise_std * tissue).clamp(0, 1)
        sample['provenance'][row]['one_shot_slide_artifacts_v4'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 29,
            'low_exposure': low, 'exposure': exposure,
            'tissue_read_noise_std': noise_std, 'noise_seed': noise_seed,
            'basis': 'TRAIN-donor acquired sagittal-ish image scale; broad high-exposure tail retained',
        }
    return {**sample, 'inputs': inputs}
