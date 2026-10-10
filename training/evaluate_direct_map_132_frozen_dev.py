"""Frozen 132 direct-map head on the 143/145 synthetic DEV sections."""

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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_geometry_143 import dense_local_targets_143, render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
train132 = root / 'runs/v4_pose_adaptation_132'
train143 = root / 'runs/coherent_anatomy_field_143'
train145 = root / 'runs/common_support_field_145'
eval132 = root / 'runs/v4_pose_adaptation_132_dev_eval'
eval143 = root / 'runs/coherent_anatomy_field_143_dev_eval'
eval145 = root / 'runs/common_support_field_145_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
checkpoint = train132 / 'joint_step_02000.pt'
out = root / 'runs/direct_map_132_frozen_dev_baseline'
device = 'cuda'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

done132 = json.loads((train132 / 'completed.json').read_text())
config132 = json.loads((train132 / 'config.json').read_text())
done143 = json.loads((train143 / 'completed.json').read_text())
config143 = json.loads((train143 / 'config.json').read_text())
done145 = json.loads((train145 / 'completed.json').read_text())
config145 = json.loads((train145 / 'config.json').read_text())
done132eval = json.loads((eval132 / 'completed.json').read_text())
done143eval = json.loads((eval143 / 'completed.json').read_text())
done145eval = json.loads((eval145 / 'completed.json').read_text())
config143eval = json.loads((eval143 / 'config.json').read_text())
config145eval = json.loads((eval145 / 'config.json').read_text())
assert sha(train132 / 'config.json') == done132['config_sha256']
assert sha(train132 / 'draws.jsonl') == done132['draws_sha256']
assert sha(train132 / 'training.jsonl') == done132['training_sha256']
assert sha(checkpoint) == done132['checkpoint_sha256']['2000'] == \
    done132eval['checkpoint_sha256']['2000'] == config143['parent_checkpoint_sha256'] == \
    config145['pose_parent_checkpoint_sha256'] == config145eval['parent132_checkpoint_sha256']
assert sha(train132 / 'completed.json') == config143['parent_completion_sha256']
assert sha(source.parent / 'docs/publication/V4_POSE_ADAPTATION_132_PROTOCOL_20261010.md') == \
    config132['protocol_sha256'] == done132['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in config132['source_sha256'].items())
assert sha(source / 'coherent_anatomy_geometry_143.py') == \
    config143['source_sha256']['coherent_anatomy_geometry_143.py']
assert all(sha(eval132 / name) == digest for name, digest in done132eval['output_sha256'].items())
assert sha(source / 'evaluate_v4_pose_adaptation_132.py') == done132eval['evaluator_source_sha256']
assert sha(eval132 / 'completed.json') == config143eval['parent_evaluation_completion_sha256'] == \
    config145eval['parent132_eval_completion_sha256']
assert config143eval['parent_checkpoint_sha256'] == config145eval['parent132_checkpoint_sha256'] == \
    sha(checkpoint)
assert sha(train143 / 'config.json') == done143['config_sha256']
assert sha(train143 / 'completed.json') == done143eval['train_completion_sha256'] == \
    config145['parent_completion_sha256']
assert sha(train143 / 'full_step_10000.pt') == done143['checkpoint_sha256']['full']['10000'] == \
    config145['parent_checkpoint_sha256'] == config145eval['parent143_checkpoint_sha256']
assert sha(train145 / 'config.json') == done145['config_sha256']
assert sha(train145 / 'completed.json') == done145eval['train_completion_sha256']
for folder, receipt, names in (
    (eval143, done143eval, {'config.json': 'config_sha256', 'rows.jsonl': 'rows_sha256',
                            'summary.json': 'summary_sha256'}),
    (eval145, done145eval, {'config.json': 'config_sha256', 'pair_bank.jsonl': 'pair_bank_sha256',
                            'mapping_rows.jsonl': 'mapping_rows_sha256',
                            'ranking_rows.jsonl': 'ranking_rows_sha256',
                            'summary.json': 'summary_sha256'})):
    assert all(sha(folder / name) == receipt[key] for name, key in names.items())
assert sha(source / 'evaluate_coherent_anatomy_field_143.py') == done143eval['evaluator_sha256']
assert sha(source / 'evaluate_common_support_field_145.py') == done145eval['evaluator_sha256']
assert sha(eval143 / 'completed.json') == config145eval['parent143_eval_completion_sha256']
assert all(not receipt.get(key, False) for receipt in
           (done132, done132eval, done143, done143eval, done145, done145eval)
           for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                       'final_animals_used', 'external_pretrained_weights_used'))

rows132 = {(row['cohort'], row['section_id']): row for row in
           (json.loads(line) for line in (eval132 / 'synthetic_rows.jsonl').open())
           if row['arm'] == '2000'}
rows143 = {(row['cohort'], row['section_id'], row['role']): row for row in
           (json.loads(line) for line in (eval143 / 'rows.jsonl').open())
           if row['step'] == 10000 and row['arm'] == 'full'}
rows145 = {(row['cohort'], row['section_id'], row['role']): row for row in
           (json.loads(line) for line in (eval145 / 'mapping_rows.jsonl').open())
           if row['step'] == 0 and row['arm'] == 'treatment'}
assert len(rows132) == 491 and len(rows143) == len(rows145) == 982

records = {}
panel_hashes = {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    protocol = json.loads((panel / 'protocol.json').read_text())
    all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == receipt['protocol_sha256']
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in protocol['source_sha256'].items())
    assert len(all_records) == receipt['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
    assert all(row['provenance']['split'] == 'development' for row in all_records)
    records[cohort] = [row for row in all_records if row['eligible']]
    assert len(records[cohort]) == receipt['eligible'] == (248 if cohort == 'v4' else 243)
    assert all((cohort, row['section_id']) in rows132 and all(
        (cohort, row['section_id'], role) in rows143 and
        (cohort, row['section_id'], role) in rows145 for role in ('exact', 'blind_near'))
        for row in records[cohort])
    panel_hashes[cohort] = sha(panel / 'completed.json')
assert panel_hashes == config143eval['panel_receipt_sha256'] == \
    config145eval['panel_receipt_sha256']
train_plans = {row['plan_receipt'] for row in config143['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for cohort in records for row in records[cohort]}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == config132 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

axis256 = torch.arange(256, device=device) / 256
yy256, xx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((xx256, yy256), -1)
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device=device),
                                  torch.arange(64, device=device), indexing='ij')

out.mkdir(parents=True, exist_ok=False)
config = {'evaluator_sha256': sha(__file__),
    'direct_model_source_sha256': sha(source / 'arbitrary_plane_one_shot_model.py'),
    'global_support_geometry_source_sha256': sha(source / 'coherent_anatomy_geometry_143.py'),
    'checkpoint_sha256': sha(checkpoint),
    'train132_completion_sha256': sha(train132 / 'completed.json'),
    'eval132_completion_sha256': sha(eval132 / 'completed.json'),
    'eval143_completion_sha256': sha(eval143 / 'completed.json'),
    'eval145_completion_sha256': sha(eval145 / 'completed.json'),
    'panel_receipt_sha256': panel_hashes,
    'atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'cohorts': {'v4': 248, 'v3': 243},
    'map': 'frozen 132 step2000 direct head with source image, atlas and section PSF weights; native 256 output sampled at [2::4,2::4]',
    'exact': 'known synthetic target state and reflection; no pose inference',
    'blind_near': 'truth-best original 132 blind16 beam branch; <=1.5mm subset is oracle availability, not a deployable selector',
    'global_support': 'same 143/145 valid 64-grid sites whose truth CCF point lies in the rendered seven-depth atlas slab and has trilinear tissue support >0.5; no local-radius or score filter',
    'hit': 'continuous direct-map CCF point within 0.5mm of dense truth, with no correspondence-logit or dustbin filter',
    'warning': 'reused synthetic development plans, no independent animal validation, no GUI or calibrated deployment claim',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows, errors = [], {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        for record in records[cohort]:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).to(device).bool()
                truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
            assert int(valid.sum()) == record['valid_pixels']
            frozen = rows132[cohort, record['section_id']]
            branch = frozen['beam_branch_ids'][int(np.argmin(frozen['beam_rigid_mm']))]
            assert frozen['panel_file_sha256'] == record['sha256']
            assert frozen['beam_best_rigid_mm'] == min(frozen['beam_rigid_mm'])
            assert branch == rows143[cohort, record['section_id'], 'blind_near']['blind_near_branch_id']
            assert branch == rows145[cohort, record['section_id'], 'blind_near']['blind_near_branch_id']
            prediction = model.predict(image)
            observed = chart256[valid[0]]
            target_rigid = rigid_points_090(state, reflection, observed)[0]
            beam_ids = torch.tensor(frozen['beam_branch_ids'], device=device)
            beam_world = rigid_points_090(prediction['state'][:, beam_ids // 2],
                (beam_ids % 2)[None], observed)[0]
            beam_errors = (beam_world - target_rigid).norm(dim=-1).mean(-1) / 1000
            near_mm = float(beam_errors.min())
            assert int(beam_ids[beam_errors.argmin()]) == branch
            assert abs(near_mm - frozen['beam_best_rigid_mm']) < 1e-4
            assert abs(near_mm - rows143[cohort, record['section_id'], 'blind_near']['blind_near_rigid_mm']) < 1e-4
            assert (near_mm <= 1.5) == rows143[cohort, record['section_id'], 'blind_near']['blind_near_available_1p5mm']
            assert (near_mm <= 1.5) == rows145[cohort, record['section_id'], 'blind_near']['blind_near_available_1p5mm']

            states = torch.cat((state[:, None], prediction['state'][:, branch // 2:branch // 2 + 1]), 1)
            reflections = torch.stack((reflection[0], reflection.new_tensor(branch % 2)))[None]
            selected = {**prediction, 'state': states}
            mapped = model.map(selected, offsets, torch.tensor([[0, 1]], device=device),
                reflections, (256, 256), atlas, weights)
            surface64 = mapped['centre_surface_ccf_ap_dv_ml_um'][0, :, 2::4, 2::4]
            truth64 = truth[0, 2::4, 2::4]
            valid64 = valid[0, 2::4, 2::4]
            slabs, base, basis, normal = render_coherent_atlas_143(atlas, states[0],
                reflections[0], offsets.expand(2, -1), weights.expand(2, -1))
            local, _, _ = dense_local_targets_143(truth, valid, base, basis, normal, radius=6)
            x = pixel_x[None] + 64 * local[..., 0]
            y = pixel_y[None] + 64 * local[..., 1]
            z = 3 + local[..., 2] / 500
            inside = (x >= 0) & (x <= 63) & (y >= 0) & (y <= 63) & (z >= 0) & (z <= 6)
            grid = torch.stack((2 * x / 63 - 1, 2 * y / 63 - 1, z / 3 - 1), -1)[:, None]
            support = F.grid_sample(slabs[:, 1:2].float(), grid,
                align_corners=True)[:, 0, 0] > .5
            globally_supported = valid64[None] & inside & support
            distance = (surface64 - truth64).norm(dim=-1) / 1000
            for slot, role in enumerate(('exact', 'blind_near')):
                reference143 = rows143[cohort, record['section_id'], role]
                reference145 = rows145[cohort, record['section_id'], role]
                valid_error = distance[slot][valid64].cpu().numpy()
                global_error = distance[slot][globally_supported[slot]].cpu().numpy()
                assert len(valid_error) == reference143['valid_sites'] == reference145['valid_sites']
                assert len(global_error) == reference143['global_supported_sites'] == \
                    reference145['global_supported_sites']
                common = {'cohort': cohort, 'section_id': record['section_id'],
                    'physical_section_id': record['panel_physical_section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'plan_receipt_sha256': record['plan_receipt_sha256'],
                    'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'], 'appearance_mode': record['appearance_mode'],
                    'panel_file_sha256': record['sha256'],
                    'nearest_cardinal_angle_deg': frozen['nearest_cardinal_angle_deg'],
                    'blind_near_branch_id': branch, 'blind_near_rigid_mm': near_mm,
                    'blind_near_available_1p5mm': near_mm <= 1.5, 'role': role}
                row = {**common, 'valid_sites': len(valid_error),
                    'global_supported_sites': len(global_error),
                    'le_0p5mm_valid_sites': int((valid_error <= .5).sum()),
                    'le_0p5mm_global_supported_sites': int((global_error <= .5).sum()),
                    'point_error_mm_valid_mean': float(valid_error.mean()),
                    'point_error_mm_valid_median': float(np.median(valid_error)),
                    'point_error_mm_global_mean': float(global_error.mean()) if len(global_error) else None,
                    'point_error_mm_global_median': float(np.median(global_error)) if len(global_error) else None}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                errors[cohort, record['section_id'], role, 'valid'] = valid_error
                errors[cohort, record['section_id'], role, 'global_supported'] = global_error

summary = {'cohorts': {}, 'warning': config['warning']}
for cohort in panels:
    summary['cohorts'][cohort] = {}
    for role in ('exact', 'blind_near'):
        summary['cohorts'][cohort][role] = {}
        for subset in ('all', 'near_le_1p5mm'):
            group = [row for row in rows if row['cohort'] == cohort and row['role'] == role and
                     (subset == 'all' or row['blind_near_available_1p5mm'])]
            result = {'sections': len(group), 'plans': len({r['subject_plan_id'] for r in group})}
            for domain, count_key, hit_key in (
                ('valid', 'valid_sites', 'le_0p5mm_valid_sites'),
                ('global_supported', 'global_supported_sites', 'le_0p5mm_global_supported_sites')):
                site_errors = np.concatenate([errors[cohort, row['section_id'], role, domain]
                                              for row in group])
                result[domain] = {'sites': int(sum(row[count_key] for row in group)),
                    'le_0p5mm_sites': int(sum(row[hit_key] for row in group)),
                    'le_0p5mm_percent': float(100 * (site_errors <= .5).mean()),
                    'point_error_mm_mean': float(site_errors.mean()),
                    'point_error_mm_median': float(np.median(site_errors))}
            summary['cohorts'][cohort][role][subset] = result
assert len(rows) == 982
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
output_hashes = {name: sha(out / name) for name in ('config.json', 'rows.jsonl', 'summary.json')}
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'output_sha256': output_hashes, 'evaluator_sha256': sha(__file__),
    'checkpoint_sha256': sha(checkpoint), 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False}, indent=2))
assert all(sha(out / name) == digest for name, digest in output_hashes.items())
print(json.dumps({'event': 'completed', 'rows': len(rows), 'output_sha256': output_hashes}), flush=True)
