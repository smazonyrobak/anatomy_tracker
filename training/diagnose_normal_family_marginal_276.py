"""Frozen top-one versus existing-mode normal-family marginal, no training."""
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
from training.global_atlas_contrast_090 import rigid_points_090


SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/NORMAL_FAMILY_MARGINAL_276_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
OUT = ROOT / 'runs/normal_family_marginal_276_train_inner_001'
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent = json.loads((PARENT / 'completed.json').read_text())
transfer = json.loads((TRANSFER / 'completed.json').read_text())
assert parent['candidate'] == 275 and transfer['candidate'] == 272
step = parent['selected_step']
checkpoint_path = PARENT / f'pose_step_{step:05d}.pt'
with checkpoint_path.open('rb') as stream:
    checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_hash == parent['checkpoint_sha256'][str(step)]
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'])
panels = {
    'new_v9': torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False),
    'frozen_270_v9': torch.load(ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
                                map_location='cpu', weights_only=False),
}
assert all(len(panel) == 64 for panel in panels.values())
sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
acquired = []
for domain, (records_path, images_path, affine_key) in sources.items():
    by_row = {row['array_row_index']: row for row in
              (json.loads(line) for line in records_path.open())}
    array = np.load(images_path, mmap_mode='r')
    for case in (json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()):
        if case['domain'] != domain or hashlib.sha256(
                f'{domain}|{case["donor_id"]}'.encode()).digest()[0] % 2 != 1:
            continue
        record = by_row[case['array_row_index']]
        assert record['section_id'] == case['section_id']
        image = torch.from_numpy(np.asarray(array[case['array_row_index'], 0],
                                            dtype=np.float32).copy())[None, None].cuda()
        if domain == 'sagittal':
            image = F.interpolate(image, (192, 192), mode='area')
        matrix = np.asarray(record[affine_key], dtype=np.float64)
        normal = np.cross(matrix[:, 0], matrix[:, 1])
        normal /= np.linalg.norm(normal)
        acquired.append({'case': case, 'image': image, 'normal': normal,
                         'axis': int(np.abs(normal).argmax())})
assert sum(item['case']['domain'] == 'coronal' for item in acquired) == 28
assert sum(item['case']['domain'] == 'sagittal' for item in acquired) == 22

config = {'candidate': 276, 'parent_selected_step': step,
    'parent_checkpoint_sha256': checkpoint_hash,
    'parent_new_panel_sha256': parent['inner_panel_sha256'],
    'parent_readouts_sha256': parent['readouts_sha256'],
    'transfer_cases_sha256': transfer['cases_sha256'],
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'candidate_decision': 'sum existing mode mass by argmax absolute plane-normal axis; '
                          'within winning family choose max original mode/reflection branch',
    'fitted_parameters': 0, 'calibrated': False,
    'weak_affine_role': 'acquired orientation readout only',
    'dev_cal_final_or_public_benchmark_used': False}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w') as output:
    for panel_name, panel in (*panels.items(),
                              ('coronal', [item for item in acquired if item['case']['domain'] == 'coronal']),
                              ('sagittal', [item for item in acquired if item['case']['domain'] == 'sagittal'])):
        for item in panel:
            synthetic = panel_name in panels
            network_input = (item['inputs'][None].cuda() if synthetic else
                torch.cat((item['image'], torch.zeros(1, 4, 192, 192, device='cuda')), 1))
            prediction = model.predict(network_input)
            log_mass = prediction['log_mass'][0]
            branch_prior = (log_mass[:, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            normal = full_frame_state_to_components(prediction['state'][0])[1][:, :, 2]
            family = normal.abs().argmax(-1)
            family_mass = torch.stack([log_mass.exp()[family == axis].sum()
                                       for axis in range(3)])
            selected_family = int(family_mass.argmax())
            branch_family = torch.repeat_interleave(family, 2)
            choices = {'top_one': int(branch_prior.argmax()),
                       'family_marginal': int(branch_prior.masked_fill(
                           branch_family != selected_family, -torch.inf).argmax())}
            for arm, branch in choices.items():
                mode = branch // 2
                chosen_normal = normal[mode]
                truth_normal = (full_frame_state_to_components(item['state'].cuda())[1][:, 2]
                                if synthetic else torch.as_tensor(item['normal'], device='cuda'))
                angle = float(torch.rad2deg(torch.acos((chosen_normal * truth_normal)
                                                  .sum().abs().clamp(0, 1))))
                result = {'panel': panel_name, 'arm': arm,
                    'physical_section_id': item['physical_section_id'] if synthetic else None,
                    'donor_id': None if synthetic else item['case']['donor_id'],
                    'section_id': None if synthetic else item['case']['section_id'],
                    'source_image_sha256': None if synthetic else item['case']['source_image_sha256'],
                    'selected_branch': branch, 'selected_axis': int(family[mode]),
                    'true_axis': int(truth_normal.abs().argmax()),
                    'normal_angle_deg': angle,
                    'family_mass_ap_dv_ml': family_mass.tolist()}
                result['axis_correct'] = result['selected_axis'] == result['true_axis']
                if synthetic:
                    chart = item['chart'].cuda()
                    truth = rigid_points_090(item['state'][None].cuda(),
                        item['reflection'][None].cuda(), chart, (192, 192))
                    points = rigid_points_090(prediction['state'][0, mode][None],
                        torch.tensor([branch % 2], device='cuda'), chart, (192, 192))
                    result['top1_site_error_mm'] = float((points - truth).norm(dim=-1).mean() / 1000)
                    result['base_index'] = item['base_index']
                rows.append(result)
                output.write(json.dumps(result, allow_nan=False) + '\n')

summary = {'cases': {name: len(panel) for name, panel in
    (*panels.items(), ('coronal', [item for item in acquired if item['case']['domain'] == 'coronal']),
     ('sagittal', [item for item in acquired if item['case']['domain'] == 'sagittal']))},
    'arms': {}}
for arm in ('top_one', 'family_marginal'):
    summary['arms'][arm] = {}
    for panel_name in ('new_v9', 'frozen_270_v9', 'coronal', 'sagittal'):
        subset = [row for row in rows if row['arm'] == arm and row['panel'] == panel_name]
        summary['arms'][arm][panel_name] = {'n': len(subset),
            'axis_correct': sum(row['axis_correct'] for row in subset),
            'median_normal_angle_deg': float(np.median([row['normal_angle_deg'] for row in subset]))}
        if panel_name in panels:
            summary['arms'][arm][panel_name]['mean_top1_site_error_mm'] = float(np.mean([
                row['top1_site_error_mm'] for row in subset]))
raw, marginal = summary['arms']['top_one'], summary['arms']['family_marginal']
for panel_name in ('new_v9', 'frozen_270_v9'):
    assert abs(raw[panel_name]['mean_top1_site_error_mm'] -
               parent['readouts'][str(step)][panel_name]['mean_top1_mm']) < 1e-3
for domain in ('coronal', 'sagittal'):
    assert raw[domain]['axis_correct'] == parent['readouts'][str(step)][domain]['first_axis_correct']
gate = {'sagittal_gain_at_least_5':
    marginal['sagittal']['axis_correct'] - raw['sagittal']['axis_correct'] >= 5,
    'coronal_loss_at_most_1': raw['coronal']['axis_correct'] - marginal['coronal']['axis_correct'] <= 1,
    'new_synthetic_mean_error_regression_at_most_0.2mm':
        marginal['new_v9']['mean_top1_site_error_mm'] - raw['new_v9']['mean_top1_site_error_mm'] <= .2,
    'frozen_synthetic_mean_error_regression_at_most_0.2mm':
        marginal['frozen_270_v9']['mean_top1_site_error_mm'] -
        raw['frozen_270_v9']['mean_top1_site_error_mm'] <= .2}
summary['gate'] = gate
summary['gate_pass'] = all(gate.values())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'candidate': 276,
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'fitted_parameters': 0, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', **summary}), flush=True)
