"""Frozen C4 orientation-family readout on acquired TRAIN donors; no training."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090

RUN = ROOT / 'runs/c4_joint_321_002'
OUT = ROOT / 'runs/c4_acquired_family_325_001'
PROTOCOL = ROOT / 'agent_worktrees/joint_v6_integration/docs/publication/C4_ACQUIRED_FAMILY_325_PROTOCOL_20261015.md'
DATASETS = (
    ('coronal', ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
     ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy', 'animal_id',
     'model_pixel_to_ap_dv_ml_um'),
    ('sagittal', ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
     ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
     'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not OUT.exists()
checkpoint = torch.load(RUN / 'joint_12000.pt', map_location='cpu', weights_only=True)
model = C4JointSliceModel(side=192).eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
config = {'checkpoint_sha256': sha(RUN / 'joint_12000.pt'),
    'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PROTOCOL),
    'selection': 'lower median by section number and section ID, one per donor',
    'input': 'raw intensity plus four zero auxiliary channels',
    'weak_affine_role': 'diagnostic after inference only; not physical pose truth',
    'datasets': {}}
rows = []
for domain, records_path, image_path, donor_key, affine_key in DATASETS:
    records = [json.loads(line) for line in records_path.open()]
    records = [record for record in records if record.get('training_split',
               record.get('split')) == 'train']
    donors = sorted({record[donor_key] for record in records})
    assert len(donors) == (58 if domain == 'coronal' else 40)
    images = np.load(image_path, mmap_mode='r')
    config['datasets'][domain] = {'records_sha256': sha(records_path),
        'images_sha256': sha(image_path), 'donors': len(donors)}
    for donor in donors:
        group = sorted((record for record in records if record[donor_key] == donor),
                       key=lambda record: (record['section_number'], record['section_id']))
        record = group[(len(group) - 1) // 2]
        image = torch.from_numpy(np.array(images[record['array_row_index'], 0],
                                          dtype=np.float32, copy=True))[None, None]
        if domain == 'sagittal':
            image = F.interpolate(image, (192, 192), mode='area')
        network_input = torch.cat((image, torch.zeros(1, 4, 192, 192)), 1)
        with torch.inference_mode():
            prediction = model.predict(network_input)
            scores = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)[0]
            order = scores.argsort(descending=True)
            states = prediction['state'][0]
            normals = full_frame_state_to_components(states)[1][:, :, 2]
            first_branch = int(order[0])
            first_mode = first_branch // 2
            first_normal = normals[first_mode]
            observed_midpoint = torch.full((1, 1, 2), .5)
            canonical_midpoint = model.canonical_chart(observed_midpoint,
                torch.tensor([first_mode // model.normal_modes]), 192)
            midpoint = rigid_points_090(states[first_mode:first_mode + 1],
                torch.tensor([first_branch % 2]), canonical_midpoint, (192, 192))[0, 0]
        matrix = np.asarray(record[affine_key], dtype=np.float64)
        offset = 1 / 6 if domain == 'sagittal' else 0.
        scale = 256 if domain == 'sagittal' else 192
        origin = matrix[:, 2] + offset * (matrix[:, 0] + matrix[:, 1])
        edges = np.stack((scale * matrix[:, 0], scale * matrix[:, 1]), -1)
        weak_centre = origin + .5 * edges.sum(-1)
        weak_normal = np.cross(edges[:, 0], edges[:, 1])
        weak_normal /= np.linalg.norm(weak_normal)
        weak_axis = int(np.abs(weak_normal).argmax())
        families = normals.abs().argmax(-1)
        selected_axis = int(families[first_mode])
        top16 = [int(branch // 2) for branch in order[:16]]
        rows.append({'domain': domain, 'donor_id': donor,
            'animal_id': record.get('animal_id', record.get('donor_id')),
            'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
            'section_id': record['section_id'], 'array_row_index': record['array_row_index'],
            'source_image_sha256': record.get('source_image_sha256', record.get('image_sha256')),
            'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
            'selected_axis': ('AP', 'DV', 'ML')[selected_axis],
            'first_axis_correct': selected_axis == weak_axis,
            'top16_contains_weak_axis': any(int(families[mode]) == weak_axis for mode in top16),
            'full_bank_contains_weak_axis': bool((families == weak_axis).any()),
            'first_weak_normal_angle_deg': float(torch.rad2deg(torch.acos(
                (first_normal * torch.as_tensor(weak_normal, dtype=first_normal.dtype))
                .sum().abs().clamp(0, 1)))),
            'first_weak_midpoint_gap_mm': float(np.linalg.norm(midpoint.numpy() - weak_centre) / 1000)})

summary = {'cases': len(rows), 'domains': {}}
for domain in ('coronal', 'sagittal'):
    group = [row for row in rows if row['domain'] == domain]
    summary['domains'][domain] = {'donors': len(group),
        'first_axis_correct': sum(row['first_axis_correct'] for row in group),
        'top16_contains_weak_axis': sum(row['top16_contains_weak_axis'] for row in group),
        'full_bank_contains_weak_axis': sum(row['full_bank_contains_weak_axis'] for row in group),
        'median_weak_normal_angle_deg': float(np.median([row['first_weak_normal_angle_deg']
                                                        for row in group])),
        'median_weak_midpoint_gap_mm': float(np.median([row['first_weak_midpoint_gap_mm']
                                                       for row in group])),
        'selected_axes': {axis: sum(row['selected_axis'] == axis for row in group)
                          for axis in ('AP', 'DV', 'ML')}}
coronal, sagittal = summary['domains']['coronal'], summary['domains']['sagittal']
summary['guard_clauses'] = {'coronal_first_family': coronal['first_axis_correct'] >= 55,
    'coronal_normal': coronal['median_weak_normal_angle_deg'] <= 15,
    'coronal_top16': coronal['top16_contains_weak_axis'] >= 50,
    'sagittal_first_family': sagittal['first_axis_correct'] >= 30,
    'sagittal_normal': sagittal['median_weak_normal_angle_deg'] <= 30,
    'sagittal_top16': sagittal['top16_contains_weak_axis'] >= 35}
summary['guard_pass'] = all(summary['guard_clauses'].values())
summary['model_promoted'] = False
summary['calibrated'] = False
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
(OUT / 'completed.json').write_text(json.dumps({'config_sha256': sha(OUT / 'config.json'),
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'n': len(rows), 'guard_pass': summary['guard_pass']}, indent=2) + '\n')
print(json.dumps(summary, allow_nan=False), flush=True)
