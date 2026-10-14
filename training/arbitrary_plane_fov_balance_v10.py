"""Independent v9 physical sections with a narrower, provenance-recorded field of view."""
import hashlib
from pathlib import Path

import numpy as np

from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FOV_SCALE = .72


def load_fov_balance_v10(device='cuda'):
    context = load_streaming_synthetic_v7_64(device)
    for subject in context['subjects']:
        bounds = np.asarray(subject['bounds_um'])
        middle = bounds.mean(0)
        subject['bounds_um'] = (middle + FOV_SCALE * (bounds - middle)).tolist()
    context['nuisance'][:, 2:4] *= FOV_SCALE
    context['provenance']['fov_balance_v10'] = {
        'code_sha256': CODE_SHA256, 'fov_scale': FOV_SCALE,
        'geometry': 'scale conservative subject box and TRAIN acquisition span lengths about their original centres',
        'unchanged': 'independent full-sphere normals, through-support-point and box-offset branches, roll, PSF, virtual deformation, appearance, damage and optional brush',
        'scope': 'TRAIN-only appearance/framing experiment; not a measured physical-cut prevalence'}
    return context


def sample_fov_balance_v10(context, subject_indices, seed, side=192):
    sample = sample_postartifact_gauge_v1(context, subject_indices, seed, side)
    for record in sample['provenance']:
        record['physical_section_id'] += '-fov72'
        record['fov_balance_v10'] = context['provenance']['fov_balance_v10']
    return sample
