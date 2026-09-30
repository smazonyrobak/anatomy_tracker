"""Donor-uniform, memory-mapped TRAIN images with weak Allen affine targets.

Call only after the reserved-image acquisition has exited and frozen completed.json.
No development, calibration, final-test, or benchmark image is read here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import frame_to_physical_ouv, physical_ouv_to_frame

ROOT = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_reserved_real_train():
    completed_path = ROOT / 'completed.json'
    manifest = json.loads(completed_path.read_text())
    assert manifest['training_role'] == 'train_only'
    assert manifest['development_calibration_final_or_benchmark_downloads'] == 0
    assert manifest['training_donors'] == len(manifest['donor_receipts'])
    assert manifest['shape_per_image'] == [1, 192, 192] and manifest['dtype'] == 'float16'
    donors, section_ids, bindings = [], set(), {str(completed_path): sha(completed_path)}
    for animal_id, receipt_sha in sorted(manifest['donor_receipts'].items()):
        directory = ROOT / f'donor_{animal_id}'
        receipt_path = directory / 'completed.json'
        assert sha(receipt_path) == receipt_sha
        receipt = json.loads(receipt_path.read_text())
        assert str(receipt['animal_id']) == animal_id
        assert receipt['source_sha256'] == manifest['source_sha256']
        for name, expected in receipt['output_sha256'].items():
            path = directory / name
            assert sha(path) == expected
            bindings[str(path)] = expected
        bindings[str(receipt_path)] = receipt_sha
        records = [json.loads(line) for line in (directory / 'records.jsonl').open()]
        assert len(records) == receipt['image_count'] == len(receipt['section_ids'])
        assert [row['section_id'] for row in records] == receipt['section_ids']
        assert all(row['training_split'] == 'train' and str(row['animal_id']) == animal_id
                   for row in records)
        assert not section_ids.intersection(receipt['section_ids'])
        section_ids.update(receipt['section_ids'])
        with np.load(directory / 'geometry.npz') as geometry:
            state = geometry['state'].astype(np.float32)
            thickness = geometry['thickness_um'].astype(np.float32)
        assert state.shape == (len(records), 12) and thickness.shape == (len(records),)
        images = np.load(directory / 'images.npy', mmap_mode='r')
        assert images.shape == (len(records), 1, 192, 192) and images.dtype == np.float16
        del images
        donors.append({'animal_id': animal_id, 'images': directory / 'images.npy',
                       'state': state, 'thickness_um': thickness,
                       'identities': [{key: row[key] for key in
                            ('animal_id', 'specimen_id', 'experiment_id', 'section_id')}
                            for row in records]})
    assert len(section_ids) == manifest['training_images']
    return {'donors': donors, 'bindings': bindings, 'training_images': len(section_ids),
            'manifest_sha256': bindings[str(completed_path)], 'label_role': 'weak Allen affine'}


def sample_reserved_real_train(context, donor_index, row_indices, device='cuda'):
    donor = context['donors'][int(donor_index)]
    rows = np.asarray(row_indices, dtype=np.int64)
    image = np.load(donor['images'], mmap_mode='r')[rows].astype(np.float32)
    inputs = torch.zeros(len(rows), 5, 192, 192, device=device)
    inputs[:, :1] = torch.from_numpy(image).to(device)
    state = torch.from_numpy(donor['state'][rows].copy()).to(device)
    normal = full_frame_state_to_components(state)[1][..., :, 2]
    reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
    ouv = frame_to_physical_ouv(*full_frame_state_to_components(state)).reshape(-1, 3, 3).double()
    ouv[reflection, 0] += 191 / 192 * ouv[reflection, 1]
    ouv[reflection, 1] *= -1
    state = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()
    return {'inputs': inputs, 'state': state, 'reflection': reflection.long(),
            'thickness_um': torch.from_numpy(donor['thickness_um'][rows].copy()).to(device),
            'identities': [donor['identities'][i] for i in rows], 'label_role': 'weak Allen affine'}
