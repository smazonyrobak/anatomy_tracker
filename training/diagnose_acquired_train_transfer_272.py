"""Frozen pose model on donor-distinct acquired TRAIN sections; weak affines are diagnostic only."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


CORONAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
SAGITTAL = ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
PARENT = ROOT / 'runs/pose_capture_270_v9_continuation_001'
OUT = ROOT / 'runs/acquired_train_transfer_272_001'
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/ACQUIRED_TRAIN_TRANSFER_272_PROTOCOL_20261012.md'
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parent = json.loads((PARENT / 'completed.json').read_text())
assert parent['candidate'] == 270 and parent['selected_step'] == 8000
checkpoint_path = PARENT / 'pose_step_08000.pt'
with checkpoint_path.open('rb') as stream:
    checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_hash == parent['checkpoint_sha256']['8000']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'])
inputs = (
    ('coronal', CORONAL / 'records.jsonl', CORONAL / 'images.npy', 'animal_id',
     'model_pixel_to_ap_dv_ml_um'),
    ('sagittal', SAGITTAL / 'geometry.jsonl', SAGITTAL / 'model_input.npy', 'donor_id',
     'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
)
config = {'candidate': 272, 'parent_checkpoint_sha256': checkpoint_hash,
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'one_median_section_per_donor': True, 'input': 'raw unbrushed intensity, zero auxiliary channels',
    'metadata_role': 'weak Allen affine; diagnostic only, not blinded physical truth',
    'datasets': {}}
OUT.mkdir(parents=True, exist_ok=False)
cases = []
with torch.inference_mode(), (OUT / 'cases.jsonl').open('w') as output:
    for domain, records_path, images_path, donor_key, affine_key in inputs:
        with records_path.open() as stream:
            records = [json.loads(line) for line in stream]
        if domain == 'coronal':
            records = [record for record in records if record['training_split'] == 'train']
        else:
            assert all(record['split'] == 'train' for record in records)
        donors = sorted({record[donor_key] for record in records})
        assert len(donors) == (58 if domain == 'coronal' else 40)
        image_array = np.load(images_path, mmap_mode='r')
        with records_path.open('rb') as stream:
            records_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        with images_path.open('rb') as stream:
            images_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        config['datasets'][domain] = {'records_sha256': records_hash,
            'images_sha256': images_hash, 'donors': len(donors),
            'selected_section': 'lower median by section_number then section_id'}
        for donor in donors:
            cohort = sorted((record for record in records if record[donor_key] == donor),
                            key=lambda record: (record['section_number'], record['section_id']))
            record = cohort[(len(cohort) - 1) // 2]
            image = torch.from_numpy(np.array(image_array[record['array_row_index'], 0],
                                               dtype=np.float32, copy=True))[None, None].cuda()
            if domain == 'sagittal':
                image = F.interpolate(image, (192, 192), mode='area')
            network_input = torch.cat((image, torch.zeros(1, 4, 192, 192, device='cuda')), 1)
            prediction = model.predict(network_input)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            order = prior.argsort(descending=True)
            centres, frames, basis = full_frame_state_to_components(prediction['state'][0])
            normals = frames[:, :, 2]
            matrix = np.asarray(record[affine_key], dtype=np.float64)
            offset = (1 / 6 if domain == 'sagittal' else 0.)
            scale = 256 if domain == 'sagittal' else 192
            origin = matrix[:, 2] + offset * (matrix[:, 0] + matrix[:, 1])
            edges = np.stack((scale * matrix[:, 0], scale * matrix[:, 1]), -1)
            weak_centre = origin + .5 * edges.sum(-1)
            weak_normal = np.cross(edges[:, 0], edges[:, 1])
            weak_normal /= np.linalg.norm(weak_normal)
            weak_axis = int(np.abs(weak_normal).argmax())
            mode_order = (order // 2).tolist()
            first = mode_order[0]
            selected_normal = normals[first]
            angle = float(torch.rad2deg(torch.acos(
                (selected_normal * torch.as_tensor(weak_normal, device='cuda')).sum().abs().clamp(0, 1))))
            selected_axis = int(selected_normal.abs().argmax())
            weak_minor = float(np.linalg.svd(edges, compute_uv=False)[-1] / 1000)
            selected_minor = float(torch.linalg.svdvals(basis[first]).min() / 1000)
            result = {'domain': domain, 'donor_id': donor,
                'animal_id': record.get('animal_id', record.get('donor_id')),
                'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                'section_id': record['section_id'], 'array_row_index': record['array_row_index'],
                'source_image_sha256': record.get('source_image_sha256', record.get('image_sha256')),
                'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                'selected_axis': ('AP', 'DV', 'ML')[selected_axis],
                'first_axis_correct': selected_axis == weak_axis,
                'top16_contains_weak_axis': any(int(normals[mode].abs().argmax()) == weak_axis
                                                 for mode in mode_order[:16]),
                'first_weak_normal_angle_deg': angle,
                'first_weak_centre_gap_mm': float((centres[first].cpu().numpy() - weak_centre).dot(
                    centres[first].cpu().numpy() - weak_centre)**.5 / 1000),
                'first_minor_fov_mm': selected_minor, 'weak_minor_fov_mm': weak_minor,
                'image_mean': float(image.mean()), 'image_nonzero_fraction': float((image > .03).float().mean())}
            cases.append(result)
            output.write(json.dumps(result, allow_nan=False) + '\n')
        output.flush()
        print(json.dumps({'event': 'domain_complete', 'domain': domain, 'donors': len(donors)}), flush=True)
summary = {'cases': len(cases), 'domains': {}}
for domain in ('coronal', 'sagittal'):
    rows = [row for row in cases if row['domain'] == domain]
    summary['domains'][domain] = {'donors': len(rows),
        'weak_axes': {axis: sum(row['weak_axis'] == axis for row in rows) for axis in ('AP', 'DV', 'ML')},
        'selected_axes': {axis: sum(row['selected_axis'] == axis for row in rows) for axis in ('AP', 'DV', 'ML')},
        'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
        'top16_contains_weak_axis': sum(row['top16_contains_weak_axis'] for row in rows),
        'median_weak_normal_angle_deg': float(np.median([row['first_weak_normal_angle_deg'] for row in rows])),
        'median_weak_centre_gap_mm': float(np.median([row['first_weak_centre_gap_mm'] for row in rows])),
        'median_minor_fov_ratio': float(np.median([row['first_minor_fov_mm'] / row['weak_minor_fov_mm']
                                                   for row in rows]))}
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'candidate': 272,
    'cases_sha256': hashlib.sha256((OUT / 'cases.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'weak_labels_for_training_or_selection': False, 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', **summary}), flush=True)
