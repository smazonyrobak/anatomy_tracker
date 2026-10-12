"""Frozen 276 inference rule on complementary acquired TRAIN donors."""
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
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/NORMAL_FAMILY_CONFIRMATION_277_PROTOCOL_20261012.md'
parent_run = root / 'runs/pose_capture_275_v9_continuation_001'
rule_run = root / 'runs/normal_family_marginal_276_train_inner_001'
transfer_run = root / 'runs/acquired_train_transfer_272_001'
out = root / 'runs/normal_family_confirmation_277_train_inner_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent = json.loads((parent_run / 'completed.json').read_text())
rule = json.loads((rule_run / 'completed.json').read_text())
transfer = json.loads((transfer_run / 'completed.json').read_text())
assert parent['candidate'] == 275 and rule['candidate'] == 276 and transfer['candidate'] == 272
assert json.loads((rule_run / 'summary.json').read_text())['gate_pass']
checkpoint_path = parent_run / f'pose_step_{parent["selected_step"]:05d}.pt'
checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
assert checkpoint_sha256 == parent['checkpoint_sha256'][str(parent['selected_step'])]
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'])

sources = {
    'coronal': (root / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                root / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
source_sha256 = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for record_path, image_path, _ in sources.values()
                 for path in (record_path, image_path)}
config = {'candidate': 277, 'parent_selected_step': parent['selected_step'],
          'parent_checkpoint_sha256': checkpoint_sha256,
          'rule_completed_sha256': hashlib.sha256((rule_run / 'completed.json').read_bytes()).hexdigest(),
          'transfer_cases_sha256': transfer['cases_sha256'],
          'source_sha256': source_sha256,
          'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'donor_parity': 'even sha256(domain|donor_id) first byte',
          'fitted_parameters': 0, 'calibrated': False,
          'weak_affine_role': 'acquired orientation readout only',
          'dev_cal_final_or_public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))

cases = [json.loads(line) for line in (transfer_run / 'cases.jsonl').open()]
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for domain, (records_path, images_path, affine_key) in sources.items():
        records = {record['array_row_index']: record for record in
                   (json.loads(line) for line in records_path.open())}
        images = np.load(images_path, mmap_mode='r')
        for case in cases:
            if case['domain'] != domain or hashlib.sha256(
                    f'{domain}|{case["donor_id"]}'.encode()).digest()[0] % 2 != 0:
                continue
            record = records[case['array_row_index']]
            assert record['section_id'] == case['section_id']
            image = torch.from_numpy(np.asarray(images[case['array_row_index'], 0],
                                                dtype=np.float32).copy())[None, None].cuda()
            if domain == 'sagittal':
                image = F.interpolate(image, (192, 192), mode='area')
            inputs = torch.cat((image, torch.zeros(1, 4, 192, 192, device='cuda')), 1)
            prediction = model.predict(inputs)
            mass = prediction['log_mass'][0]
            branch_prior = (mass[:, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            normal = full_frame_state_to_components(prediction['state'][0])[1][:, :, 2]
            family = normal.abs().argmax(-1)
            family_mass = torch.stack([mass.exp()[family == axis].sum() for axis in range(3)])
            winning = int(family_mass.argmax())
            branch_family = torch.repeat_interleave(family, 2)
            choices = {'top_one': int(branch_prior.argmax()),
                       'family_marginal': int(branch_prior.masked_fill(
                           branch_family != winning, -torch.inf).argmax())}
            matrix = np.asarray(record[affine_key], dtype=np.float64)
            true_normal = np.cross(matrix[:, 0], matrix[:, 1])
            true_normal /= np.linalg.norm(true_normal)
            truth = torch.as_tensor(true_normal, device='cuda')
            true_axis = int(np.abs(true_normal).argmax())
            for arm, branch in choices.items():
                chosen_normal = normal[branch // 2]
                angle = float(torch.rad2deg(torch.acos((chosen_normal * truth).sum().abs().clamp(0, 1))))
                row = {'domain': domain, 'arm': arm, 'donor_id': case['donor_id'],
                       'animal_id': case['animal_id'], 'specimen_id': case['specimen_id'],
                       'experiment_id': case['experiment_id'], 'section_id': case['section_id'],
                       'source_image_sha256': case['source_image_sha256'],
                       'selected_branch': branch, 'selected_axis': int(family[branch // 2]),
                       'true_axis': true_axis, 'normal_angle_deg': angle,
                       'family_mass_ap_dv_ml': family_mass.tolist()}
                row['axis_correct'] = row['selected_axis'] == row['true_axis']
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')

summary = {'cases': {domain: sum(row['domain'] == domain and row['arm'] == 'top_one'
                                 for row in rows) for domain in sources}, 'arms': {}}
assert summary['cases'] == {'coronal': 30, 'sagittal': 18}
for arm in ('top_one', 'family_marginal'):
    summary['arms'][arm] = {}
    for domain in sources:
        chosen = [row for row in rows if row['domain'] == domain and row['arm'] == arm]
        summary['arms'][arm][domain] = {'n': len(chosen),
            'axis_correct': sum(row['axis_correct'] for row in chosen),
            'median_normal_angle_deg': float(np.median([row['normal_angle_deg'] for row in chosen]))}
raw, marginal = summary['arms']['top_one'], summary['arms']['family_marginal']
gate = {'sagittal_gain_at_least_4': marginal['sagittal']['axis_correct'] - raw['sagittal']['axis_correct'] >= 4,
        'coronal_loss_at_most_1': raw['coronal']['axis_correct'] - marginal['coronal']['axis_correct'] <= 1,
        'sagittal_median_angle_reduction_at_least_5deg':
            raw['sagittal']['median_normal_angle_deg'] -
            marginal['sagittal']['median_normal_angle_deg'] >= 5}
summary['gate'] = gate
summary['gate_pass'] = all(gate.values())
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'candidate': 277,
    'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'fitted_parameters': 0, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', **summary}), flush=True)
