"""Read-only audit of the frozen, zero-training 276 decision experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090


root = Path('I:/AnatomyTracker')
run = root / 'runs/normal_family_marginal_276_train_inner_001'
parent_run = root / 'runs/pose_capture_275_v9_continuation_001'
source = Path(__file__).resolve().parent
config = json.loads((run / 'config.json').read_text())
completed = json.loads((run / 'completed.json').read_text())
parent = json.loads((parent_run / 'completed.json').read_text())
assert config['candidate'] == completed['candidate'] == 276
assert config['fitted_parameters'] == completed['fitted_parameters'] == 0
assert config['calibrated'] is completed['calibrated'] is False
assert config['dev_cal_final_or_public_benchmark_used'] is False

for name in ('config', 'rows', 'summary'):
    path = run / f'{name}.{"jsonl" if name == "rows" else "json"}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == completed[f'{name}_sha256']
assert hashlib.sha256((source / 'diagnose_normal_family_marginal_276.py').read_bytes()).hexdigest() == config['source_sha256']
assert hashlib.sha256((source.parent / 'docs/publication/NORMAL_FAMILY_MARGINAL_276_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
assert config['parent_selected_step'] == parent['selected_step']
assert config['parent_checkpoint_sha256'] == parent['checkpoint_sha256'][str(parent['selected_step'])]
assert hashlib.sha256((parent_run / f'pose_step_{parent["selected_step"]:05d}.pt').read_bytes()).hexdigest() == config['parent_checkpoint_sha256']
for name, path in (('parent_new_panel', parent_run / 'inner_panel.pt'),
                   ('parent_readouts', parent_run / 'readouts.jsonl'),
                   ('transfer_cases', root / 'runs/acquired_train_transfer_272_001/cases.jsonl')):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config[f'{name}_sha256']

rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
summary = json.loads((run / 'summary.json').read_text())
expected_n = {'new_v9': 64, 'frozen_270_v9': 64, 'coronal': 28, 'sagittal': 22}
assert summary['cases'] == expected_n and len(rows) == 2 * sum(expected_n.values())
assert all(rows[i]['arm'] == 'top_one' and rows[i + 1]['arm'] == 'family_marginal'
           and rows[i]['panel'] == rows[i + 1]['panel']
           and rows[i]['physical_section_id'] == rows[i + 1]['physical_section_id']
           and rows[i]['donor_id'] == rows[i + 1]['donor_id']
           and rows[i]['section_id'] == rows[i + 1]['section_id']
           and np.allclose(rows[i]['family_mass_ap_dv_ml'], rows[i + 1]['family_mass_ap_dv_ml'])
           for i in range(0, len(rows), 2))
assert all(abs(sum(row['family_mass_ap_dv_ml']) - 1) < 1e-5 for row in rows)
assert all(row['axis_correct'] == (row['selected_axis'] == row['true_axis']) for row in rows)

cases = [json.loads(line) for line in
         (root / 'runs/acquired_train_transfer_272_001/cases.jsonl').open()]
case_ids = {(case['domain'], case['donor_id']): case for case in cases}
for panel in ('coronal', 'sagittal'):
    selected = [row for row in rows if row['panel'] == panel and row['arm'] == 'top_one']
    assert len({row['donor_id'] for row in selected}) == expected_n[panel]
    for row in selected:
        case = case_ids[panel, row['donor_id']]
        assert row['section_id'] == case['section_id']
        assert row['source_image_sha256'] == case['source_image_sha256']
        assert hashlib.sha256(f'{panel}|{row["donor_id"]}'.encode()).digest()[0] % 2 == 1

for arm in ('top_one', 'family_marginal'):
    for panel, count in expected_n.items():
        selected = [row for row in rows if row['arm'] == arm and row['panel'] == panel]
        saved = summary['arms'][arm][panel]
        assert len(selected) == saved['n'] == count
        assert sum(row['axis_correct'] for row in selected) == saved['axis_correct']
        assert abs(np.median([row['normal_angle_deg'] for row in selected]) -
                   saved['median_normal_angle_deg']) < 1e-9
        if panel in ('new_v9', 'frozen_270_v9'):
            assert abs(np.mean([row['top1_site_error_mm'] for row in selected]) -
                       saved['mean_top1_site_error_mm']) < 1e-9

raw, marginal = summary['arms']['top_one'], summary['arms']['family_marginal']
gate = {'sagittal_gain_at_least_5': marginal['sagittal']['axis_correct'] - raw['sagittal']['axis_correct'] >= 5,
        'coronal_loss_at_most_1': raw['coronal']['axis_correct'] - marginal['coronal']['axis_correct'] <= 1,
        'new_synthetic_mean_error_regression_at_most_0.2mm': marginal['new_v9']['mean_top1_site_error_mm'] - raw['new_v9']['mean_top1_site_error_mm'] <= .2,
        'frozen_synthetic_mean_error_regression_at_most_0.2mm': marginal['frozen_270_v9']['mean_top1_site_error_mm'] - raw['frozen_270_v9']['mean_top1_site_error_mm'] <= .2}
assert gate == summary['gate'] and all(gate.values()) == summary['gate_pass']
for panel in ('new_v9', 'frozen_270_v9'):
    assert abs(raw[panel]['mean_top1_site_error_mm'] -
               parent['readouts'][str(parent['selected_step'])][panel]['mean_top1_mm']) < 1e-3
for panel in ('coronal', 'sagittal'):
    assert raw[panel]['axis_correct'] == parent['readouts'][str(parent['selected_step'])][panel]['first_axis_correct']

torch.set_num_threads(4)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
checkpoint = torch.load(parent_run / f'pose_step_{parent["selected_step"]:05d}.pt',
                        map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'])
panels = {'new_v9': torch.load(parent_run / 'inner_panel.pt', map_location='cpu', weights_only=False),
          'frozen_270_v9': torch.load(root / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
                                      map_location='cpu', weights_only=False)}
replayed = 0
with torch.inference_mode():
    for panel, items in panels.items():
        panel_rows = [row for row in rows if row['panel'] == panel]
        for index in (0, 15, 32, 63):
            item = items[index]
            prediction = model.predict(item['inputs'][None].cuda())
            log_mass = prediction['log_mass'][0]
            branch_scores = (log_mass[:, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            normal = full_frame_state_to_components(prediction['state'][0])[1][:, :, 2]
            family = normal.abs().argmax(-1)
            masses = torch.stack([log_mass.exp()[family == axis].sum() for axis in range(3)])
            winning = int(masses.argmax())
            branches = (int(branch_scores.argmax()),
                        int(branch_scores.masked_fill(torch.repeat_interleave(family, 2) != winning,
                                                      -torch.inf).argmax()))
            truth = rigid_points_090(item['state'][None].cuda(), item['reflection'][None].cuda(),
                                     item['chart'].cuda(), (192, 192))
            for arm_index, branch in enumerate(branches):
                saved = panel_rows[2 * index + arm_index]
                proposed = rigid_points_090(prediction['state'][0, branch // 2][None],
                    torch.tensor([branch % 2], device='cuda'), item['chart'][None].cuda(), (192, 192))
                error = float((proposed - truth).norm(dim=-1).mean() / 1000)
                assert saved['physical_section_id'] == item['physical_section_id']
                assert saved['selected_branch'] == branch
                assert saved['selected_axis'] == int(family[branch // 2])
                assert np.allclose(saved['family_mass_ap_dv_ml'], masses.cpu().numpy(), atol=1e-6)
                assert abs(saved['top1_site_error_mm'] - error) < 1e-4
                replayed += 1

print(json.dumps({'verified': True, 'rows': len(rows), 'replayed_synthetic_arm_rows': replayed,
                  'gate_pass': summary['gate_pass'], 'sagittal_raw': raw['sagittal']['axis_correct'],
                  'sagittal_marginal': marginal['sagittal']['axis_correct']}))
