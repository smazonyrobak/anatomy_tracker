"""Independent v9 planes with optional label-preserving TRAIN-image Fourier style."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_one_shot_slide_artifacts_v10(context, subject_indices, seed, style_banks, side=192):
    sample = sample_one_shot_slide_artifacts_v9(context, subject_indices, seed, side)
    original = sample['inputs'][:, 0].clone()
    yy, xx = torch.meshgrid(torch.arange(side, device=original.device),
                            torch.arange(side, device=original.device), indexing='ij')
    low = ((xx - side // 2).abs() <= 4) & ((yy - side // 2).abs() <= 4)
    low[side // 2, side // 2] = False
    for row, subject_index in enumerate(subject_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 31]))
        use_style = bool(rng.random() < .5)
        domain = ('coronal', 'sagittal')[int(rng.integers(2))] if use_style else None
        index = int(rng.integers(len(style_banks[domain]['records']))) if use_style else None
        alpha = float(rng.uniform(.4, .8)) if use_style else 0.
        if use_style:
            source = torch.fft.fftshift(torch.fft.fft2(original[row]))
            target = torch.fft.fftshift(torch.fft.fft2(style_banks[domain]['images'][index]))
            amplitude = torch.where(low, source.abs().lerp(target.abs(), alpha), source.abs())
            styled = torch.fft.ifft2(torch.fft.ifftshift(amplitude *
                                                         torch.exp(1j * torch.angle(source)))).real
            sample['inputs'][row, 0] = (original[row] + (styled - original[row]) *
                                        sample['visible'][row].clamp(0, 1)).clamp(0, 1)
        sample['provenance'][row]['one_shot_slide_artifacts_v10'] = {
            'code_sha256': CODE_SHA256, 'seed_branch': 31,
            'fourier_style_applied': use_style, 'style_domain': domain,
            'style_index': index,
            'style_source': style_banks[domain]['records'][index] if use_style else None,
            'amplitude_blend': alpha,
            'frequency_window': 'central 9x9 excluding DC of 192x192 FFT',
            'phase_and_dc': 'unchanged source',
            'tissue_blend': 'source visible support; exact-zero exterior preserved',
            'sampling': 'one independently generated physical plane per row; no appearance pairing',
            'weak_alignments_used': False}
    return sample
