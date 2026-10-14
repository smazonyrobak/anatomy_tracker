"""Post-joint-run, fixed TRAIN-only acquired pose diagnostic; weak affines are not truth."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint

pose_run = root / 'runs/joint_spatial_feedback_305_pose_002'
checkpoint = pose_run / 'pose_step_26000.pt'
out = root / 'runs/joint_spatial_feedback_305_acquired_train_001'
datasets = (
    ('coronal', root / 'data/joint_v7_allen_fullcanvas_192_001',
     'records.jsonl', 'images.npy', 'animal_id', 'model_pixel_to_ap_dv_ml_um'),
    ('sagittal', root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'geometry.jsonl', 'model_input.npy', 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
)
assert Path(__file__).resolve().drive.upper() == checkpoint.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
receipt = json.loads((pose_run / 'completed.json').read_text(encoding='utf8'))
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert receipt['summary']['selected_step'] == 26000 and receipt['output_sha256'][checkpoint.name] == checkpoint_sha
model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/JOINT_SPATIAL_305_ACQUIRED_TRAIN_DIAGNOSTIC_PROTOCOL_20261014.md'
config = {'checkpoint_sha256': checkpoint_sha,
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'scope': 'TRAIN donors only; zero-shot image prior; weak affines diagnostic only',
          'selection': 'one lower-median section per donor; no checkpoint or hyperparameter selection',
          'datasets': {}}
cases = []
with torch.inference_mode(), (out / 'cases.jsonl').open('w', encoding='utf8') as output:
    for domain, directory, records_name, images_name, donor_key, affine_key in datasets:
        records_path, images_path = directory / records_name, directory / images_name
        records = [json.loads(line) for line in records_path.open(encoding='utf8')]
        if domain == 'coronal':
            records = [row for row in records if row['training_split'] == 'train']
        else:
            assert all(row['split'] == 'train' for row in records)
        donors = sorted({row[donor_key] for row in records})
        assert len(donors) == (58 if domain == 'coronal' else 40)
        images = np.load(images_path, mmap_mode='r')
        config['datasets'][domain] = {
            'records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
            'images_sha256': hashlib.sha256(images_path.read_bytes()).hexdigest(),
            'donors': len(donors)}
        for donor in donors:
            cohort = sorted((row for row in records if row[donor_key] == donor),
                            key=lambda row: (row['section_number'], row['section_id']))
            row = cohort[(len(cohort) - 1) // 2]
            image = torch.from_numpy(np.array(images[row['array_row_index'], 0],
                                               dtype=np.float32, copy=True))[None, None].cuda()
            if domain == 'sagittal':
                image = F.interpolate(image, (192, 192), mode='area')
            network_input = torch.cat((image, torch.zeros(1, 4, 192, 192, device='cuda')), 1)
            prediction = model.predict(network_input)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            branches = prior.argsort(descending=True)
            centres, frames, basis = full_frame_state_to_components(prediction['state'][0])
            normals = frames[:, :, 2]
            affine = np.asarray(row[affine_key], dtype=np.float64)
            offset = 1 / 6 if domain == 'sagittal' else 0.
            scale = 256 if domain == 'sagittal' else 192
            origin = affine[:, 2] + offset * (affine[:, 0] + affine[:, 1])
            edges = np.stack((scale * affine[:, 0], scale * affine[:, 1]), -1)
            weak_centre = origin + .5 * edges.sum(-1)
            weak_normal = np.cross(edges[:, 0], edges[:, 1])
            weak_normal /= np.linalg.norm(weak_normal)
            first_mode = int(branches[0] // 2)
            selected_normal = normals[first_mode]
            weak_normal_gpu = torch.as_tensor(weak_normal, device='cuda', dtype=selected_normal.dtype)
            angle = float(torch.rad2deg(torch.acos(
                (selected_normal * weak_normal_gpu).sum().abs().clamp(0, 1))))
            weak_axis = int(np.abs(weak_normal).argmax())
            selected_axis = int(selected_normal.abs().argmax())
            case = {'domain': domain, 'donor_id': donor,
                    'animal_id': row.get('animal_id', row.get('donor_id')),
                    'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                    'section_id': row['section_id'], 'array_row_index': row['array_row_index'],
                    'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256')),
                    'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                    'selected_axis': ('AP', 'DV', 'ML')[selected_axis],
                    'first_axis_correct': selected_axis == weak_axis,
                    'top16_contains_weak_axis': any(int(normals[int(branch // 2)].abs().argmax())
                                                    == weak_axis for branch in branches[:16]),
                    'first_weak_normal_angle_deg': angle,
                    'first_weak_centre_gap_mm': float(np.linalg.norm(
                        centres[first_mode].cpu().numpy() - weak_centre) / 1000),
                    'first_minor_fov_mm': float(torch.linalg.svdvals(basis[first_mode]).min() / 1000),
                    'weak_minor_fov_mm': float(np.linalg.svd(edges, compute_uv=False)[-1] / 1000)}
            cases.append(case)
            output.write(json.dumps(case, allow_nan=False) + '\n')
summary = {'cases': len(cases), 'domains': {}}
for domain, *_ in datasets:
    rows = [row for row in cases if row['domain'] == domain]
    summary['domains'][domain] = {
        'donors': len(rows), 'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
        'top16_contains_weak_axis': sum(row['top16_contains_weak_axis'] for row in rows),
        'median_weak_normal_angle_deg': float(np.median(
            [row['first_weak_normal_angle_deg'] for row in rows])),
        'median_weak_centre_gap_mm': float(np.median(
            [row['first_weak_centre_gap_mm'] for row in rows]))}
(out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(out / 'completed.json').write_text(json.dumps({
    'cases_sha256': hashlib.sha256((out / 'cases.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'weak_labels_for_selection': False, 'calibrated': False,
    'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps(summary))
