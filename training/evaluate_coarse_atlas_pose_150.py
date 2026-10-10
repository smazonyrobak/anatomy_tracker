"""Frozen new-plan synthetic DEV test of 150 plane correction versus its controls."""

import hashlib
import json
import os
import sys
from collections import defaultdict
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coarse_atlas_pose_148 import CoarseAtlasPose148
from training.coarse_atlas_pose_150 import CoarseAtlasPose150
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(rows):
    plans = defaultdict(list)
    for row in rows:
        plans[row['synthetic_subject_plan_id']].append(row)
    return {'sections': len(rows), 'plans': len(plans),
        'before_mm': float(np.mean([np.mean([row['before_rigid_mm'] for row in part])
                                    for part in plans.values()])) if plans else None,
        'after_mm': float(np.mean([np.mean([row['after_rigid_mm'] for row in part])
                                   for part in plans.values()])) if plans else None,
        'improvement_mm': float(np.mean([np.mean([row['improvement_mm'] for row in part])
                                         for part in plans.values()])) if plans else None,
        'worsened_gt_0p10_count': sum(row['improvement_mm'] < -.1 for row in rows),
        'worsened_gt_0p10_percent': 100 * sum(row['improvement_mm'] < -.1 for row in rows) / len(rows)
            if rows else None}


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_150_PROTOCOL_20261010.md'
panel = root / 'data/coarse_atlas_pose_150_fresh_dev_panel'
run150 = root / 'runs/coarse_atlas_pose_150'
run148 = root / 'runs/coarse_atlas_pose_148'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/coarse_atlas_pose_150_fresh_dev_eval'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert all(path.drive.upper() == 'I:' for path in
           (source, protocol, panel, run150, run148, parent_dir, out))
assert not out.exists()

panel_done = json.loads((panel / 'completed.json').read_text())
panel_config = json.loads((panel / 'protocol.json').read_text())
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert len(records) == panel_done['physical_sections'] == 256
assert sum(row['eligible'] for row in records) == panel_done['eligible']
assert len({row['panel_physical_section_id'] for row in records}) == len(records)
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
assert all(row['provenance']['split'] == 'development' for row in records)
assert all(sha(panel / row['file']) == row['sha256'] for row in records)
assert all(sha(panel / 'source' / name) == digest
           for name, digest in panel_config['source_sha256'].items())

done150 = json.loads((run150 / 'completed.json').read_text())
config150 = json.loads((run150 / 'config.json').read_text())
done148 = json.loads((run148 / 'completed.json').read_text())
config148 = json.loads((run148 / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(run150 / 'config.json') == done150['config_sha256']
assert sha(run150 / 'draws.jsonl') == done150['draws_sha256']
assert sha(run150 / 'training.jsonl') == done150['training_sha256']
assert sha(run150 / 'inner_records.jsonl') == done150['inner_records_sha256']
assert sha(run150 / 'inner_scores.jsonl') == done150['inner_scores_sha256']
assert sha(protocol) == config150['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in config150['source_sha256'].items())
assert all(sha(source / name) == digest for name, digest in config148['source_sha256'].items())
assert sha(run148 / 'config.json') == done148['config_sha256']
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert done150['steps'] == 6000 and done150['selected_step'] in (0, 1000, 3000, 6000)
assert done150['accepted_synthetic_presentations'] == 18000
assert set(config150['gradient_train_bases']).isdisjoint(config150['inner_checkpoint_bases'])
assert len(config150['gradient_train_bases']) == 60 and len(config150['inner_checkpoint_bases']) == 4
train_plans = {row['plan_receipt'] for row in config150['synthetic_provenance']['base_subjects']}
assert len(train_plans) == 64
assert not train_plans & {row['plan_receipt_sha256'] for row in records}
assert not any(receipt.get(key, False) for receipt in (done150, config150, done148, parent_done)
               for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                           'final_animals_used', 'external_pretrained_weights_used'))

selected = done150['selected_step']
models = {}
for arm in ('full', 'support_only'):
    path = run150 / f'{arm}_step_{selected:05d}.pt'
    assert sha(path) == done150['checkpoint_sha256'][arm][str(selected)]
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['config'] == config150 and saved['step'] == selected and saved['arm'] == arm
    model = CoarseAtlasPose150().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['matcher'], strict=True)
    models[arm] = model
    del saved
path148 = run148 / 'full_step_06000.pt'
assert sha(path148) == done148['checkpoint_sha256']['full']['6000']
saved148 = torch.load(path148, map_location='cpu', weights_only=True)
assert saved148['config'] == config148 and saved148['step'] == 6000
model148 = CoarseAtlasPose148().cuda().eval().requires_grad_(False)
model148.load_state_dict(saved148['matcher'], strict=True)
models['frozen_148_full'] = model148
del saved148

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved_pose = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved_pose['config'] == parent_config and saved_pose['step'] == 2000
pose.load_state_dict(saved_pose['model'], strict=True)
del saved_pose
axis = torch.arange(256, device='cuda') / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart256 = torch.stack((xx, yy), -1)


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'train150_completion_sha256': sha(run150 / 'completed.json'),
    'train148_completion_sha256': sha(run148 / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_path), 'selected_step': selected,
    'arms': tuple(models), 'roles': ('exact', 'blind_near', 'blind_first'),
    'metric': 'mean Euclidean rigid CCF map error in mm at all valid native256 tissue sites',
    'blind_near': 'truth-best branch from frozen 132 blind16 beam; oracle diagnostic only',
    'blind_first': 'top-prior branch from frozen 132; no 150 reranking',
    'scope': 'eight new synthetic DEV deformation plans, not biological animals or physical all-angle truth',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for index, record in enumerate(records):
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        prediction = pose.predict(image)
        prior, ids = beam_for(prediction)
        observed = chart256[valid]
        target = rigid_points_090(state, reflection, observed)[0]
        proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                    (ids % 2)[None], observed)[0]
        errors = (proposed - target).norm(dim=-1).mean(-1) / 1000
        near_slot = int(errors.argmin())
        first_id = int(prior.argmax())
        first_slot = ids.tolist().index(first_id)
        near_id = int(ids[near_slot])
        states = torch.cat((state[:, None],
            prediction['state'][:, near_id // 2:near_id // 2 + 1],
            prediction['state'][:, first_id // 2:first_id // 2 + 1]), 1)
        reflections = torch.cat((reflection[:, None],
            ids[near_slot:near_slot + 1][None] % 2,
            ids[first_slot:first_slot + 1][None] % 2), 1)
        initial = (0., float(errors[near_slot]), float(errors[first_slot]))
        angle = float(np.degrees(np.arccos(np.clip(
            np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
        angle_bin = '<15' if angle < 15 else '15-30' if angle < 30 else '30-45' if angle < 45 else '>=45'
        for arm, matcher in models.items():
            with torch.autocast('cuda', dtype=torch.float16):
                output = matcher(image, atlas, states, reflections, offsets, weights,
                                 atlas_intensity=arm != 'support_only')
            corrected = output['corrected_state'].float()
            assert corrected.shape == states.shape and bool(torch.isfinite(corrected).all())
            mapped = rigid_points_090(corrected, reflections, observed)
            after = ((mapped - target).norm(dim=-1).mean(-1) / 1000)[0].tolist()
            for role, slot in (('exact', 0), ('blind_near', 1), ('blind_first', 2)):
                row = {'section_id': record['section_id'],
                    'physical_section_id': record['panel_physical_section_id'],
                    'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                    'plan_receipt_sha256': record['plan_receipt_sha256'],
                    'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
                    'valid_sites': int(valid.sum()), 'nearest_cardinal_angle_deg': angle,
                    'nearest_cardinal_angle_bin': angle_bin,
                    'arm': arm, 'role': role,
                    'initial_branch_id': None if slot == 0 else
                        (near_id if slot == 1 else first_id),
                    'initial_reflection': int(reflections[0, slot]),
                    'before_rigid_mm': initial[slot], 'after_rigid_mm': float(after[slot]),
                    'improvement_mm': initial[slot] - float(after[slot])}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        if (index + 1) % 64 == 0:
            stream.flush()
            print(json.dumps({'event': 'development_milestone',
                              'processed_panel_sections': index + 1}), flush=True)


def metric(arm, role, subset=None, angle=None):
    group = [row for row in rows if row['arm'] == arm and row['role'] == role and
             (subset != 'near_le_1p5' or row['before_rigid_mm'] <= 1.5) and
             (angle is None or row['nearest_cardinal_angle_bin'] == angle)]
    return summarize(group)


summary = {'counts': {'panel_sections': len(records),
    'eligible_sections': sum(row['eligible'] for row in records),
    'synthetic_development_plans': 8}, 'selected_step': selected,
    'metrics': {arm: {role: {scope: {angle: metric(arm, role, scope, angle)
        for angle in (None, '<15', '15-30', '30-45', '>=45')}
        for scope in ('all', 'near_le_1p5')}
        for role in ('exact', 'blind_near', 'blind_first')}
        for arm in models},
    'warning': 'Synthetic-plan DEV only. blind_near is truth-best beam branch and cannot be selected at inference. blind_first has no post-fit reranking. No biological all-angle, calibration or public benchmark claim.'}
near = metric('full', 'blind_near', 'near_le_1p5')
support_near = metric('support_only', 'blind_near', 'near_le_1p5')
exact = metric('full', 'exact')
assert near['sections'] == support_near['sections'] and near['sections'] > 0
gate = {'v4_near_sections': near['sections'], 'v4_near_plans': near['plans'],
    'full_improvement_mm': near['improvement_mm'],
    'support_minus_full_after_mm': support_near['after_mm'] - near['after_mm'],
    'full_worsened_gt_0p10_percent': near['worsened_gt_0p10_percent'],
    'exact_after_mm': exact['after_mm'],
    'thresholds': {'full_improvement_mm_min': .30,
                   'support_minus_full_after_mm_min': .15,
                   'worsened_gt_0p10_percent_max': 30., 'exact_after_mm_max': .25}}
gate['pass'] = bool(gate['full_improvement_mm'] >= .30 and
    gate['support_minus_full_after_mm'] >= .15 and
    gate['full_worsened_gt_0p10_percent'] <= 30. and gate['exact_after_mm'] <= .25)
summary['decision'] = gate
assert len(rows) == panel_done['eligible'] * len(models) * 3
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'train150_completion_sha256': sha(run150 / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows), 'gate_pass': gate['pass']}), flush=True)
