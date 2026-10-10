"""Correct-plane zero-offset mapping null for the frozen 143 DEV sections."""

import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components


root = Path('I:/AnatomyTracker')
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
rows = [json.loads(line) for line in
        (root / 'runs/coherent_anatomy_field_143_dev_eval/rows.jsonl').open()]
axis = (torch.arange(64) + .5) / 64
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
for cohort, panel in panels.items():
    counts = {'valid': 0, 'rigid_le_500_um': 0, 'field_le_500_um': 0}
    errors = []
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    selected = {row['section_id']: row for row in rows if row['cohort'] == cohort
                and row['step'] == 10000 and row['arm'] == 'full' and row['role'] == 'exact'}
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            state = torch.from_numpy(arrays['target_state'][None].copy())
            reflection = int(arrays['reflection'])
            truth = torch.from_numpy(arrays['target_centre_um'][2::4, 2::4].copy())
            valid = torch.from_numpy(arrays['valid_mask'][2::4, 2::4].copy()).bool()
        centre, frame, basis = full_frame_state_to_components(state)
        edge = frame[:, :, :2] @ basis
        chart = torch.stack((255 / 256 - xx if reflection else xx, yy), -1)
        rigid = centre[0] + torch.einsum('ij,hwj->hwi', edge[0], chart - .5)
        distance = (rigid - truth).norm(dim=-1)
        counts['valid'] += int(valid.sum())
        counts['rigid_le_500_um'] += int(((distance <= 500) & valid).sum())
        counts['field_le_500_um'] += selected[record['section_id']]['scored_le_500_um_all_valid']
        errors.append(float(distance[valid].mean() / 1000))
    print(json.dumps({'cohort': cohort, 'sections': len(errors), **counts,
        'rigid_le_500_um_percent': 100 * counts['rigid_le_500_um'] / counts['valid'],
        'field_le_500_um_percent': 100 * counts['field_le_500_um'] / counts['valid'],
        'rigid_mean_mm_section_equal': float(np.mean(errors))}))
