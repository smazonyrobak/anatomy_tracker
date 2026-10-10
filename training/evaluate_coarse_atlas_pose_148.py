"""Frozen synthetic DEV readout of 148 global plane correction."""

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
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_148_PROTOCOL_20261010.md'
run = root / 'runs/coarse_atlas_pose_148'
out = root / 'runs/coarse_atlas_pose_148_dev_eval'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('full', 'support_only')
checkpoints = (0, 1000, 3000, 6000)
roles = ('exact', 'blind_near', 'blind_first')
angle_bins = ('all', '<15', '15-30', '30-45', '>=45')
device = 'cuda'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert all(path.drive.upper() == 'I:' for path in
           (root, source, protocol, run, out, parent_dir, parent_eval, *panels.values()))
assert not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_eval_done = json.loads((parent_eval / 'completed.json').read_text())
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000'] == train_config['parent_checkpoint_sha256']
assert sha(parent_dir / 'completed.json') == train_config['parent_completion_sha256']
assert all(sha(parent_eval / name) == digest
           for name, digest in parent_eval_done['output_sha256'].items())
assert sha(source / 'evaluate_v4_pose_adaptation_132.py') == parent_eval_done['evaluator_source_sha256']
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(run / 'draws.jsonl') == train_done['draws_sha256']
assert sha(run / 'training.jsonl') == train_done['training_sha256']
assert train_config['protocol_sha256'] == sha(protocol)
assert train_config['source_sha256'] == train_done['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert train_done['steps'] == train_config['steps'] == checkpoints[-1]
assert train_done['accepted_synthetic_physical_sections'] == 3 * checkpoints[-1]
assert train_done['v4_presentations'] == 2 * checkpoints[-1]
assert train_done['v3_presentations'] == checkpoints[-1]
assert not train_done['pose_unfrozen'] and not train_done['joint_feedback_trained']
assert tuple(train_config['arms']) == arms and train_config['checkpoints'] == list(checkpoints)
assert all(sha(run / f'{arm}_step_{step:05d}.pt') ==
           train_done['checkpoint_sha256'][arm][str(step)]
           for arm in arms for step in checkpoints)
assert not any(receipt.get(key, False) for receipt in
               (parent_done, parent_eval_done, train_done, train_config)
               for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                           'final_animals_used', 'external_pretrained_weights_used'))

parent_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
assert len(parent_rows) == 491
panel_records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert sum(row['eligible'] for row in records) == done['eligible']
    panel_records[cohort] = [row for row in records if row['eligible']]
    assert all((cohort, row['section_id']) in parent_rows for row in panel_records[cohort])
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243
train_plans = {row['plan_receipt'] for row in train_config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for records in panel_records.values() for row in records}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans
assert all(row['lineage']['split'] == 'train'
           for row in train_config['synthetic_provenance']['base_subjects'])
assert {row['panel_physical_section_id'] for row in panel_records['v4']}.isdisjoint(
       {row['panel_physical_section_id'] for row in panel_records['v3']})

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config
pose.load_state_dict(saved['model'], strict=True)
del saved

axis256 = torch.arange(256, device=device) / 256
gy256, gx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((gx256, gy256), -1)


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device=device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


cases = []
with torch.inference_mode():
    for cohort, panel in panels.items():
        for record in panel_records[cohort]:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).to(device).bool()
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
            frozen = parent_rows[cohort, record['section_id']]
            assert ids.tolist() == frozen['beam_branch_ids']
            assert near_id == ids[int(np.argmin(frozen['beam_rigid_mm']))]
            assert first_id == frozen['top1_branch_id']
            assert abs(float(errors[near_slot]) - frozen['beam_best_rigid_mm']) < 1e-4
            assert abs(float(errors[first_slot]) - frozen['top1_rigid_mm']) < 1e-4
            states = torch.cat((state[:, None],
                prediction['state'][:, near_id // 2:near_id // 2 + 1],
                prediction['state'][:, first_id // 2:first_id // 2 + 1]), 1)
            reflections = torch.cat((reflection[:, None],
                ids[near_slot:near_slot + 1][None] % 2,
                ids[first_slot:first_slot + 1][None] % 2), 1)
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            angle_bin = '<15' if angle < 15 else '15-30' if angle < 30 else '30-45' if angle < 45 else '>=45'
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'], 'appearance_mode': record['appearance_mode'],
                'panel_file_sha256': record['sha256'], 'valid_sites': int(valid.sum()),
                'nearest_cardinal_angle_deg': angle, 'nearest_cardinal_angle_bin_deg': angle_bin,
                'blind_near_branch_id': near_id, 'blind_first_branch_id': first_id,
                'blind_near_initial_le_1p5mm': bool(errors[near_slot] <= 1.5)}
            cases.append((panel, record, states.cpu(), reflections.cpu(),
                          (0., float(errors[near_slot]), float(errors[first_slot])), common))
assert len(cases) == 491
assert sum(case[5]['blind_near_initial_le_1p5mm'] for case in cases if case[5]['cohort'] == 'v4') == 179
assert sum(case[5]['cohort'] == 'v4' and case[4][2] <= 1.5 for case in cases) == 73
del pose, prediction, image, proposed, target, errors, state, reflection, valid


def summarize(group):
    plans = defaultdict(list)
    for row in group:
        plans[row['subject_plan_id']].append(row)
    means = {key: float(np.mean([row[key] for row in group])) if group else None
             for key in ('before_rigid_mm', 'after_rigid_mm', 'improvement_mm')}
    equal = {key: float(np.mean([np.mean([row[key] for row in part])
                                for part in plans.values()])) if plans else None
             for key in means}
    worsened = sum(row['improvement_mm'] < -.1 for row in group)
    return {'sections': len(group), 'plans': len(plans),
        'valid_native256_sites': sum(row['valid_sites'] for row in group),
        'section_equal_mean_mm': means, 'plan_equal_mean_mm': equal,
        'worsened_gt_0p10mm_count': worsened,
        'worsened_gt_0p10mm_percent': 100 * worsened / len(group) if group else None}


def paired(full, support):
    a = {row['section_id']: row for row in full}
    b = {row['section_id']: row for row in support}
    assert len(a) == len(full) and len(b) == len(support) and a.keys() == b.keys()
    assert all(a[key]['before_rigid_mm'] == b[key]['before_rigid_mm'] and
               a[key]['valid_sites'] == b[key]['valid_sites'] and
               a[key]['subject_plan_id'] == b[key]['subject_plan_id'] and
               a[key]['initial_branch_id'] == b[key]['initial_branch_id'] for key in a)
    plans = defaultdict(list)
    for key in a:
        plans[a[key]['subject_plan_id']].append(b[key]['after_rigid_mm'] - a[key]['after_rigid_mm'])
    differences = [b[key]['after_rigid_mm'] - a[key]['after_rigid_mm'] for key in a]
    return {'paired_sections': len(a), 'plans': len(plans),
        'support_minus_full_after_mm_section_equal': float(np.mean(differences)) if a else None,
        'support_minus_full_after_mm_plan_equal': float(np.mean([
            np.mean(part) for part in plans.values()])) if plans else None,
        'full_minus_support_after_mm_plan_equal': -float(np.mean([
            np.mean(part) for part in plans.values()])) if plans else None,
        'full_better_count': sum(value > 0 for value in differences)}


out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_evaluation_completion_sha256': sha(parent_eval / 'completed.json'),
    'panel_receipt_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'checkpoints': checkpoints, 'arms': arms, 'roles': roles,
    'cohorts': {key: len(value) for key, value in panel_records.items()},
    'metric': 'mean Euclidean error (mm) of corrected versus true rigid CCF maps over every valid observed native256 tissue site; no atlas-support, reachability, or match-confidence filtering',
    'blind_near': 'truth-best branch from frozen 132 blind16; DEV oracle diagnostic, not deployable selection',
    'blind_first': 'frozen 132 direct top-prior branch, independent of DEV truth; no post-correction reranking',
    'split': '64 synthetic TRAIN deformation plans versus eight disjoint synthetic DEV plans; no new physical animals',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    initial_weights = None
    for step in checkpoints:
        for arm in arms:
            saved = torch.load(run / f'{arm}_step_{step:05d}.pt',
                               map_location='cpu', weights_only=True)
            assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
            if step == 0 and arm == 'full':
                initial_weights = saved['matcher']
            if step == 0 and arm == 'support_only':
                assert all(torch.equal(initial_weights[key], saved['matcher'][key])
                           for key in initial_weights)
                initial_weights = None
            matcher = CoarseAtlasPose148().to(device).eval().requires_grad_(False)
            matcher.load_state_dict(saved['matcher'], strict=True)
            del saved
            for index, (panel, record, states_cpu, reflections_cpu, before, common) in enumerate(cases):
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                    state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                    reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).to(device).bool()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                    weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
                states, reflections = states_cpu.to(device), reflections_cpu.to(device)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = matcher(image, atlas, states, reflections, offsets, weights,
                                     atlas_intensity=arm == 'full')
                corrected = output['corrected_state'].float()
                assert corrected.shape == states.shape and bool(torch.isfinite(corrected).all())
                observed = chart256[valid]
                target = rigid_points_090(state[:, None], reflection[:, None], observed)
                predicted = rigid_points_090(corrected, reflections, observed)
                after = ((predicted - target).norm(dim=-1).mean(-1) / 1000)[0].tolist()
                assert all(np.isfinite(after))
                branch_ids = (None, common['blind_near_branch_id'], common['blind_first_branch_id'])
                for slot, role in enumerate(roles):
                    row = {**common, 'step': step, 'arm': arm, 'role': role,
                        'initial_branch_id': branch_ids[slot],
                        'initial_reflection': int(reflections[0, slot]),
                        'before_rigid_mm': before[slot], 'after_rigid_mm': after[slot],
                        'improvement_mm': before[slot] - after[slot]}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
                if (index + 1) % 64 == 0:
                    stream.flush()
                    print(json.dumps({'event': 'development_milestone', 'step': step,
                                      'arm': arm, 'sections': index + 1}), flush=True)
            del matcher

summary = {'metrics': {}, 'paired_full_support': {},
    'warning': 'Synthetic DEV rigid-gauge truth. blind_near is a truth-best frozen-beam oracle diagnostic; blind_first fixes the direct top-prior branch and does not test post-correction selection. No physical expert truth or calibration.'}
summary['cohort_denominators'] = {cohort: {
    'eligible_sections': len(panel_records[cohort]),
    'blind_near_initial_le_1p5mm': sum(case[5]['cohort'] == cohort and
        case[4][1] <= 1.5 for case in cases),
    'blind_first_initial_le_1p5mm': sum(case[5]['cohort'] == cohort and
        case[4][2] <= 1.5 for case in cases)} for cohort in panels}
for cohort in panels:
    summary['metrics'][cohort] = {}
    summary['paired_full_support'][cohort] = {}
    for step in checkpoints:
        summary['metrics'][cohort][str(step)] = {}
        summary['paired_full_support'][cohort][str(step)] = {}
        for arm in arms:
            summary['metrics'][cohort][str(step)][arm] = {}
            for role in roles:
                summary['metrics'][cohort][str(step)][arm][role] = {}
                base = [row for row in rows if row['cohort'] == cohort and row['step'] == step
                        and row['arm'] == arm and row['role'] == role]
                for angle_bin in angle_bins:
                    group = [row for row in base if angle_bin == 'all' or
                             row['nearest_cardinal_angle_bin_deg'] == angle_bin]
                    summary['metrics'][cohort][str(step)][arm][role][angle_bin] = {
                        'all_sections': summarize(group),
                        'initial_le_1p5mm': summarize([row for row in group
                            if row['before_rigid_mm'] <= 1.5])}
        for role in roles:
            summary['paired_full_support'][cohort][str(step)][role] = {}
            for angle_bin in angle_bins:
                arm_groups = {}
                for arm in arms:
                    arm_groups[arm] = [row for row in rows if row['cohort'] == cohort and
                        row['step'] == step and row['arm'] == arm and row['role'] == role and
                        (angle_bin == 'all' or row['nearest_cardinal_angle_bin_deg'] == angle_bin)]
                summary['paired_full_support'][cohort][str(step)][role][angle_bin] = {
                    'all_sections': paired(*[arm_groups[arm] for arm in arms]),
                    'initial_le_1p5mm': paired(*[[row for row in arm_groups[arm]
                        if row['before_rigid_mm'] <= 1.5] for arm in arms])}


def metric(cohort, step, arm, role, scope='all_sections'):
    return summary['metrics'][cohort][str(step)][arm][role]['all'][scope]


final = checkpoints[-1]
near = metric('v4', final, 'full', 'blind_near', 'initial_le_1p5mm')
support_near = metric('v4', final, 'support_only', 'blind_near', 'initial_le_1p5mm')
exact = metric('v4', final, 'full', 'exact')
assert near['sections'] == support_near['sections'] == 179
assert exact['sections'] == 248
gate = {'blind_near_v4_initial_le_1p5_sections': near['sections'],
    'full_plan_equal_improvement_mm': near['plan_equal_mean_mm']['improvement_mm'],
    'support_minus_full_after_mm_plan_equal':
        support_near['plan_equal_mean_mm']['after_rigid_mm'] - near['plan_equal_mean_mm']['after_rigid_mm'],
    'full_minus_support_after_mm_plan_equal':
        near['plan_equal_mean_mm']['after_rigid_mm'] - support_near['plan_equal_mean_mm']['after_rigid_mm'],
    'full_worsened_gt_0p10mm_sections': near['worsened_gt_0p10mm_count'],
    'full_worsened_gt_0p10mm_percent': near['worsened_gt_0p10mm_percent'],
    'exact_v4_full_after_mm_plan_equal': exact['plan_equal_mean_mm']['after_rigid_mm'],
    'thresholds': {'full_improvement_mm': .30, 'full_vs_support_mm': .15,
                   'worsened_percent_max': 30., 'exact_after_mm_max': .25}}
gate['pass'] = bool(gate['full_plan_equal_improvement_mm'] >= .30 and
    gate['support_minus_full_after_mm_plan_equal'] >= .15 and
    gate['full_worsened_gt_0p10mm_percent'] <= 30. and
    gate['exact_v4_full_after_mm_plan_equal'] <= .25)
v3_near = metric('v3', final, 'full', 'blind_near', 'initial_le_1p5mm')
v3_exact = metric('v3', final, 'full', 'exact')
summary['decision'] = gate
summary['v3_retention_descriptive'] = {
    'blind_near_initial_le_1p5_sections': v3_near['sections'],
    'full_plan_equal_improvement_mm': v3_near['plan_equal_mean_mm']['improvement_mm'],
    'support_minus_full_after_mm_plan_equal':
        metric('v3', final, 'support_only', 'blind_near', 'initial_le_1p5mm')['plan_equal_mean_mm']['after_rigid_mm']
        - v3_near['plan_equal_mean_mm']['after_rigid_mm'],
    'full_worsened_gt_0p10mm_percent': v3_near['worsened_gt_0p10mm_percent'],
    'exact_full_after_mm_plan_equal': v3_exact['plan_equal_mean_mm']['after_rigid_mm'],
    'no_predeclared_numeric_gate': True}
assert len(rows) == 491 * len(checkpoints) * len(arms) * len(roles)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows), 'gate_pass': gate['pass']}), flush=True)
