"""Fixed TRAIN-only acquired pose retention for the two 305 fluorescence arms."""
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

source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/JOINT_SPATIAL_305_FLUORESCENCE_RETENTION_TRAIN_PROTOCOL_20261014.md'
run = root / 'runs/joint_spatial_305_fluorescence_adaptation_001'
prior_cases_path = root / 'runs/joint_spatial_feedback_305_acquired_train_001/cases.jsonl'
out = root / 'runs/joint_spatial_305_fluorescence_retention_train_001'
datasets = (
    ('coronal', root / 'data/joint_v7_allen_fullcanvas_192_001',
     'records.jsonl', 'images.npy', 'animal_id', 'model_pixel_to_ap_dv_ml_um'),
    ('sagittal', root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'geometry.jsonl', 'model_input.npy', 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
)
assert source.drive.upper() == run.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

with (run / 'completed.json').open('rb') as stream:
    run_receipt_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
run_receipt = json.loads((run / 'completed.json').read_text(encoding='utf8'))
assert run_receipt['summary']['step_600_gate_pass']
with prior_cases_path.open('rb') as stream:
    prior_cases_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
prior_cases = [json.loads(line) for line in prior_cases_path.open(encoding='utf8')]
assert len(prior_cases) == 98

config = {'scope': 'fixed TRAIN-only direct-image-prior retention; weak Allen affines diagnostic only',
          'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'adaptation_completion_sha256': run_receipt_sha,
          'prior_305_cases_sha256': prior_cases_sha,
          'checkpoints': {}, 'datasets': {},
          'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
for arm in ('control', 'adapt'):
    checkpoint = run / arm / 'pose_step_00600.pt'
    arm_receipt_path = run / arm / 'completed.json'
    with arm_receipt_path.open('rb') as stream:
        arm_receipt_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert run_receipt['output_sha256'][f'{arm}/completed.json'] == arm_receipt_sha
    arm_receipt = json.loads(arm_receipt_path.read_text(encoding='utf8'))
    assert arm_receipt['arm'] == arm and arm_receipt['updates'] == 600
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert arm_receipt['output_sha256'][checkpoint.name] == checkpoint_sha
    config['checkpoints'][arm] = {'path': str(checkpoint),
        'sha256': checkpoint_sha, 'arm_completion_sha256': arm_receipt_sha}
prepared = []
images_by_domain = {}
for domain, directory, records_name, images_name, donor_key, affine_key in datasets:
    records_path, images_path = directory / records_name, directory / images_name
    if domain == 'coronal':
        with records_path.open('rb', buffering=0) as stream:
            train_lines = [stream.readline() for _ in range(1280)]
        records = [json.loads(line) for line in train_lines]
        assert all(row['training_split'] == 'train' and row['array_row_index'] < 1280
                   for row in records)
    else:
        train_lines = records_path.read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in train_lines]
        assert all(row['split'] == 'train' for row in records)
    donors = sorted({row[donor_key] for row in records})
    assert len(donors) == (58 if domain == 'coronal' else 40)
    images_by_domain[domain] = np.load(images_path, mmap_mode='r')
    config['datasets'][domain] = {'train_records_sha256': hashlib.sha256(
        b''.join(train_lines)).hexdigest(), 'train_record_rows': len(records),
        'donors': len(donors), 'selected_model_input_sha256': {}}
    for donor in donors:
        cohort = sorted((row for row in records if row[donor_key] == donor),
                        key=lambda row: (row['section_number'], row['section_id']))
        row = cohort[(len(cohort) - 1) // 2]
        earlier = prior_cases[len(prepared)]
        image_sha = row.get('source_image_sha256', row.get('image_sha256'))
        assert (domain, donor, row['section_id'], row['array_row_index'], image_sha) == (
            earlier['domain'], earlier['donor_id'], earlier['section_id'],
            earlier['array_row_index'], earlier['source_image_sha256'])
        prepared.append((domain, row, affine_key, earlier['weak_axis']))
assert len(prepared) == 98

out.mkdir(parents=True, exist_ok=False)
cases = []
with (out / 'cases.jsonl').open('w', encoding='utf8') as output, torch.inference_mode():
    for arm in ('control', 'adapt'):
        checkpoint = run / arm / 'pose_step_00600.pt'
        model, model_config = load_one_shot_checkpoint(checkpoint, device='cuda')
        assert model_config['candidate'] == 305
        assert model_config['no_dev_cal_final_or_public_benchmark']
        for domain, row, affine_key, expected_axis in prepared:
            image = torch.from_numpy(np.array(images_by_domain[domain][
                row['array_row_index'], 0], dtype=np.float32, copy=True))[None, None].cuda()
            if domain == 'sagittal':
                image = F.interpolate(image, (192, 192), mode='area')
            input_sha = hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
            input_hashes = config['datasets'][domain]['selected_model_input_sha256']
            if arm == 'control':
                input_hashes[row['section_id']] = input_sha
            else:
                assert input_hashes[row['section_id']] == input_sha
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
            angle = float(torch.rad2deg(torch.acos((selected_normal * torch.as_tensor(
                weak_normal, device='cuda', dtype=selected_normal.dtype)).sum().abs().clamp(0, 1))))
            weak_axis = int(np.abs(weak_normal).argmax())
            selected_axis = int(selected_normal.abs().argmax())
            assert ('AP', 'DV', 'ML')[weak_axis] == expected_axis
            case = {'arm': arm, 'domain': domain, 'donor_id': row.get('donor_id', row.get('animal_id')),
                'animal_id': row.get('animal_id', row.get('donor_id')),
                'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                'section_id': row['section_id'], 'array_row_index': row['array_row_index'],
                'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256')),
                'model_input_sha256': input_sha,
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
        del model

summary = {'cases': len(cases), 'domains': {}, 'adapt_minus_control': {}}
for domain, *_ in datasets:
    for arm in ('control', 'adapt'):
        rows = [row for row in cases if row['domain'] == domain and row['arm'] == arm]
        summary['domains'].setdefault(domain, {})[arm] = {
            'donors': len(rows), 'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
            'top16_contains_weak_axis': sum(row['top16_contains_weak_axis'] for row in rows),
            'median_weak_normal_angle_deg': float(np.median([
                row['first_weak_normal_angle_deg'] for row in rows])),
            'median_weak_centre_gap_mm': float(np.median([
                row['first_weak_centre_gap_mm'] for row in rows]))}
    control, adapt = summary['domains'][domain]['control'], summary['domains'][domain]['adapt']
    summary['adapt_minus_control'][domain] = {key: adapt[key] - control[key]
        for key in ('first_axis_correct', 'top16_contains_weak_axis',
                    'median_weak_normal_angle_deg', 'median_weak_centre_gap_mm')}

(out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(out / 'completed.json').write_text(json.dumps({
    'cases_sha256': hashlib.sha256((out / 'cases.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'weak_labels_for_selection': False, 'calibrated': False,
    'no_dev_cal_final_or_public_benchmark': True}, indent=2), encoding='utf8')
print(json.dumps(summary))
