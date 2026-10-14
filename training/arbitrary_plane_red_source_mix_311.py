"""TRAIN-only anatomical source variation for independent full-sphere virtual cuts."""
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10

ROOT = Path('I:/AnatomyTracker/data/allen_stpt_red_310_001')
TRAIN_SOURCES = (
    (156198187, 607436902, 112942577, '19d18b558a44cbe6fbdc72e0d53eeb6d72da9b74f497a8a16ffa3e7b7ef5b6a3'),
    (168615344, 607628055, 160252067, '0e5c93594f05b31b0c7f9e2b2898fc790b9d665c722362ed3a79a729397d054a'),
    (307742524, 315602749, 287634601, '17d9ac4f2a92a60ec186e2f7817d3387e4c0c0d5a155018853dbdec75b74e3c6'),
)


def load_red_source_mix_311(device='cuda'):
    context = load_fov_balance_v10(device)
    brain = context['atlas'][1].cpu().numpy() > 0
    intensities = [context['atlas'][0].clone()]
    identities = [{'kind': 'pinned_allen_average', 'donor_id': None,
                   'template_sha256': context['provenance']['atlas_binding']['template_sha256']}]
    for experiment, file_id, donor, expected_sha in TRAIN_SOURCES:
        archive = ROOT / f'{experiment}_{file_id}.zip'
        with archive.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_sha
        with ZipFile(archive) as source:
            header = source.read('resampled_red.mhd').decode()
            assert 'DimSize = 528 320 456' in header and 'ElementSpacing = 25 25 25' in header
            assert 'Offset = 0 0 0' in header and 'ElementType = MET_USHORT' in header
            red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
                (456, 320, 528)).transpose(2, 1, 0)
        low, high = np.quantile(red[brain], (.01, .99))
        normalized = np.clip((red.astype(np.float32) - low) / (high - low), 0, 1)
        normalized[~brain] = 0
        intensities.append(torch.from_numpy(normalized).to(device))
        identities.append({'kind': 'allen_individual_stpt_red', 'experiment_id': experiment,
                           'well_known_file_id': file_id, 'donor_id': donor,
                           'archive_sha256': expected_sha, 'brain_q01_q99': [float(low), float(high)],
                           'split': 'TRAIN', 'registration_truth': 'approximate atlas registration, not physical oblique or biological warp truth'})
    context['source_intensities'] = intensities
    context['source_identities'] = identities
    context['provenance']['red_source_mix_311'] = {
        'sources': identities, 'source_choice': 'one uniform source per independently drawn batch',
        'held_out_donors': [182849231, 517337553, 642454692],
        'role': 'TRAIN-only source variation; no exact donor biological warp truth'}
    return context


def sample_red_source_mix_311(context, subject_indices, seed, side=192):
    source_index = int(np.random.default_rng(np.random.SeedSequence([seed, 311])).integers(
        len(context['source_intensities'])))
    context['atlas'][0].copy_(context['source_intensities'][source_index])
    sample = sample_fov_balance_v10(context, subject_indices, seed, side)
    for record in sample['provenance']:
        record['image_source'] = context['source_identities'][source_index]
    return sample
