"""Bounded TRAIN-only native-map fit-difficulty pilot from 132 step 2000."""

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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.joint_pose_map_121 import joint_forward_121, joint_target_loss_121


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ON_POLICY_JOINT_FEEDBACK_161_STAGE1_PILOT_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_checkpoint = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/joint_feedback_161_stage1_train_pilot_001'
seed, side, updates = 202610111620, 256, 192
checkpoints = (0, 96, 192)
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source_names = (
    'train_joint_feedback_161_stage1_pilot.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py',
    'joint_pose_map_121.py')
source_sha = {name: sha(source / name) for name in source_names}
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256']['2000']
assert source_sha['arbitrary_plane_one_shot_model.py'] == parent_config['source_sha256'][
    'arbitrary_plane_one_shot_model.py']
assert source_sha['global_plane_matcher_120.py'] == parent_config['source_sha256'][
    'global_plane_matcher_120.py']
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
atlas_support_only = atlas.clone()
atlas_support_only[0].zero_()
assert torch.equal(atlas_support_only[1], atlas[1])
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
parent_weights = saved['model']
model.load_state_dict(parent_weights, strict=True)
del saved
axis = torch.arange(side, device='cuda') / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart256 = torch.stack((xx, yy), -1)
axis24 = (torch.arange(24, device='cuda') + .5) / 24 - .5 / side
yy24, xx24 = torch.meshgrid(axis24, axis24, indexing='ij')
chart24 = torch.stack((xx24, yy24), -1).reshape(-1, 2)


def draw(base, draw_seed):
    with torch.no_grad():
        original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
        return sample_one_shot_slide_artifacts_v4(
            context, [base], draw_seed, side, source_sample=original)


def candidate_record(sample, attempt, base, draw_seed):
    provenance = sample['provenance'][0]
    assert provenance['split'] == 'train' and provenance['base_lineage']['split'] == 'train'
    valid = sample['valid_mask'][0]
    normal = np.asarray(provenance['physical_unit_normal_ap_dv_ml'])
    row = {'attempt': attempt, 'base_index': base, 'draw_seed': draw_seed,
        'physical_section_id': provenance['physical_section_id'],
        'provenance': provenance, 'eligible': bool(sample['eligible'][0]),
        'valid_fraction': float(valid.float().mean()), 'valid_pixels': int(valid.sum()),
        'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(
            np.clip(np.abs(normal).max(), 0, 1)))),
        'beam_branch_ids': [], 'beam_rigid_mm': [], 'near_branch_id': None,
        'wrong_branch_id': None, 'support_dice': None,
        'support_area_gap': None, 'common_support_cells': None}
    if not row['eligible']:
        return row
    with torch.inference_mode():
        prediction = model.predict(sample['inputs'])
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
        ids = beam[0]
        state = prediction['state'][:, ids // 2]
        reflection = (ids % 2)[None]
        observed = chart256[valid]
        truth = rigid_points_090(sample['state'], sample['reflection'], observed)[0]
        proposed = rigid_points_090(state, reflection, observed)[0]
        cost = (proposed - truth).norm(dim=-1).mean(-1) / 1000
        row['beam_branch_ids'], row['beam_rigid_mm'] = ids.tolist(), cost.tolist()
        near = int(cost.argmin())
        if float(cost[near]) > 1.5:
            return row
        row['near_branch_id'] = int(ids[near])
        plane = rigid_points_090(state, reflection, chart24).reshape(16, 24, 24, 3)
        frame = full_frame_state_to_components(state)[1][0]
        world = plane[:, None] + sample['offsets'][0][None, :, None, None, None] * \
            frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world, (0., 0., 0.), (25., 25., 25.), sample['weights'].expand(16, -1))
        support = (rendered[:, 1] > .5).flatten(1).float()
        area = support.sum(-1)
        overlap = support @ support[near]
        dice = 2 * overlap / (area + area[near]).clamp_min(1)
        gap = (area - area[near]).abs() / (24 * 24)
        wrong = (cost >= cost[near] + 1.) & (dice >= .85) & (gap <= .08) & (overlap >= 32)
        if bool(wrong.any()):
            slot = int(cost.masked_fill(~wrong, torch.inf).argmin())
            row.update(wrong_branch_id=int(ids[slot]), support_dice=float(dice[slot]),
                support_area_gap=float(gap[slot]), common_support_cells=int(overlap[slot]))
        return row


def fitted(sample, row, atlas_input, remove_visual=False):
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        if remove_visual:
            prediction['feature'] = torch.zeros_like(prediction['feature'])
        near_id = row['near_branch_id']
        wrong_id = row['wrong_branch_id']
        ids = [near_id, near_id] + ([] if wrong_id is None else [wrong_id])
        modes = torch.tensor([[branch // 2 for branch in ids]], device='cuda')
        reflections = torch.tensor([[near_id % 2, int(sample['reflection'][0])]
                                    + ([] if wrong_id is None else [wrong_id % 2])],
                                   device='cuda')
        states = torch.stack((prediction['state'][0, near_id // 2], sample['state'][0])
                             + (() if wrong_id is None else
                                (prediction['state'][0, wrong_id // 2],)), 0)[None]
    joint = joint_forward_121(model, prediction, sample['inputs'], modes, reflections,
        atlas_input, sample['offsets'], sample['weights'], candidate_state=states,
        match_side=24, map_side=96)
    valid_pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    sites = valid_pixels[torch.linspace(0, len(valid_pixels) - 1, 256,
        device='cuda').long()][None]
    target = joint_target_loss_121(joint['mapped'], sample['valid_mask'], sample['centre'],
        sample['state'], sample['reflection'], sites)
    local = joint['mapped']['local_displacement_um'].float() / 1000
    warp = (local.square().mean((2, 3, 4))
        + .3 * (local[..., 1:] - local[..., :-1]).square().mean((2, 3, 4))
        + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean((2, 3, 4)))
    correction = (joint['match']['update'][..., 3:6] / 2000).square().sum(-1)
    difficulty = joint['mapped']['fit_energy'].float() + .05 * warp + .05 * correction
    return joint, target, warp, difficulty


inner_rng = np.random.default_rng(seed + 1)
train_rng = np.random.default_rng(seed + 2)
inner_attempts, inner_cases = [], []
while len(inner_cases) < 64 and len(inner_attempts) < 256:
    base = int(inner_rng.integers(60, 64))
    draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
    sample = draw(base, draw_seed)
    row = candidate_record(sample, len(inner_attempts), base, draw_seed)
    inner_attempts.append(row)
    if row['wrong_branch_id'] is not None:
        inner_cases.append(row)
assert len(inner_cases) == 64, 'fewer than 64 independent TRAIN-inner pair sections'
train_attempts, train_cases = [], []
while len(train_cases) < updates and len(train_attempts) < 1024:
    base = int(train_rng.integers(60))
    draw_seed = int(train_rng.integers(0, 2**63 - 1, dtype=np.int64))
    sample = draw(base, draw_seed)
    row = candidate_record(sample, len(train_attempts), base, draw_seed)
    train_attempts.append(row)
    if row['near_branch_id'] is not None:
        train_cases.append(row)
assert len(train_cases) == updates, 'fewer than 192 independent near-branch TRAIN sections'
assert len({r['physical_section_id'] for r in inner_attempts + train_attempts}) == \
    len(inner_attempts) + len(train_attempts)

config = {'seed': seed, 'updates_per_arm': updates, 'checkpoints': checkpoints,
    'inner_pair_sections': len(inner_cases), 'inner_attempts': len(inner_attempts),
    'train_near_sections': len(train_cases), 'train_attempts': len(train_attempts),
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol), 'source_sha256': source_sha,
    'synthetic_provenance': context['provenance'],
    'train_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'atlas_control': 'same inherited 132 full-trained weights; intensity zero throughout continuation, support unchanged',
    'rank_unit': 'one canonical near/wrong decision per independent physical section',
    'no_development_or_final_data': True, 'no_external_weights': True}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('inner_attempts.jsonl', inner_attempts),
                   ('train_attempts.jsonl', train_attempts)):
    with (run / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
print(json.dumps({'event': 'manifests_frozen', 'train_attempts': len(train_attempts),
    'train_near': len(train_cases), 'inner_attempts': len(inner_attempts),
    'inner_pairs': len(inner_cases)}), flush=True)


def inner_readout(atlas_input, remove_visual=False):
    model.eval()
    results = []
    with torch.inference_mode():
        for row in inner_cases:
            sample = draw(row['base_index'], row['draw_seed'])
            assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
            joint, target, _, difficulty = fitted(sample, row, atlas_input, remove_visual)
            results.append({'physical_section_id': row['physical_section_id'],
                'near_win': bool(difficulty[0, 0] < difficulty[0, 2]),
                'near_difficulty': float(difficulty[0, 0]),
                'wrong_difficulty': float(difficulty[0, 2]),
                'near_map_mm': float(target['map_loss'][0, 0]),
                'exact_map_mm': float(target['map_loss'][0, 1]),
                'exact_rigid_drift_mm': float(target['rigid_loss'][0, 1]),
                'valid_fraction': row['valid_fraction'],
                'nearest_cardinal_angle_deg': row['nearest_cardinal_angle_deg']})
    metrics = {'n': len(results), 'near_wins': sum(r['near_win'] for r in results),
        'near_win_fraction': float(np.mean([r['near_win'] for r in results])),
        'near_map_mm': float(np.mean([r['near_map_mm'] for r in results])),
        'exact_map_mm': float(np.mean([r['exact_map_mm'] for r in results])),
        'exact_rigid_drift_mm': float(np.mean([r['exact_rigid_drift_mm'] for r in results]))}
    return metrics, results


arm_results = {}
for arm, atlas_input in (('full_intensity', atlas), ('support_only', atlas_support_only)):
    torch.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 3)
    model.load_state_dict(parent_weights, strict=True)
    model.requires_grad_(False)
    modules = (model.global_plane_matcher_120, model.atlas_encoder, model.pair,
               model.warp_shared, model.warp_condition, model.warp, model.fit_quality_head)
    for module in modules:
        module.requires_grad_(True)
    parameters = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-4)
    directory = run / arm
    directory.mkdir(parents=True, exist_ok=False)
    checkpoints_metrics = {}
    with (directory / 'training.jsonl').open('w') as training, \
            (directory / 'inner_scores.jsonl').open('w') as scores:
        for step in range(updates + 1):
            if step in checkpoints:
                metrics, cases = inner_readout(atlas_input)
                checkpoints_metrics[step] = metrics
                scores.write(json.dumps({'step': step, 'metrics': metrics,
                    'cases': cases}, allow_nan=False) + '\n')
                scores.flush()
                torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config,
                    'calibrated': False}, directory / f'joint_stage1_step_{step:05d}.pt')
                print(json.dumps({'event': 'inner_checkpoint', 'arm': arm,
                    'step': step, 'metrics': metrics}), flush=True)
            if step == updates:
                break
            row = train_cases[step]
            sample = draw(row['base_index'], row['draw_seed'])
            assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
            model.train()
            joint, target, warp, difficulty = fitted(sample, row, atlas_input)
            loss = (target['map_loss'][0, :2].sum()
                + .5 * target['rigid_loss'][0, :2].sum()
                + .03 * warp.mean())
            if row['wrong_branch_id'] is not None:
                loss = loss + .25 * F.softplus(.1 + difficulty[0, 0] - difficulty[0, 2])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            training_row = {'step': step + 1, 'physical_section_id': row['physical_section_id'],
                'valid_fraction': row['valid_fraction'],
                'nearest_cardinal_angle_deg': row['nearest_cardinal_angle_deg'],
                'has_pair': row['wrong_branch_id'] is not None,
                'near_map_mm': float(target['map_loss'][0, 0].detach()),
                'exact_map_mm': float(target['map_loss'][0, 1].detach()),
                'exact_rigid_drift_mm': float(target['rigid_loss'][0, 1].detach()),
                'near_difficulty': float(difficulty[0, 0].detach()),
                'wrong_difficulty': (None if row['wrong_branch_id'] is None else
                                     float(difficulty[0, 2].detach())),
                'loss': float(loss.detach()), 'gradient_norm': float(gradient)}
            training.write(json.dumps(training_row, allow_nan=False) + '\n')
            if (step + 1) % 32 == 0:
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    **training_row}), flush=True)
    baseline = checkpoints_metrics[0]
    qualifying = [step for step in checkpoints if
        checkpoints_metrics[step]['near_map_mm'] <= baseline['near_map_mm'] + .10 and
        checkpoints_metrics[step]['exact_map_mm'] <= baseline['exact_map_mm'] + .10 and
        checkpoints_metrics[step]['exact_rigid_drift_mm'] <= .25]
    selected = (max(qualifying, key=lambda step: (
        checkpoints_metrics[step]['near_win_fraction'],
        -(checkpoints_metrics[step]['near_map_mm']
          + checkpoints_metrics[step]['exact_map_mm']), -step))
        if qualifying else 0)
    arm_results[arm] = {'selected_step': selected, 'qualified_steps': qualifying,
        'checkpoint_metrics': checkpoints_metrics}
    print(json.dumps({'event': 'arm_selected', 'arm': arm,
        'selected_step': selected, 'qualified_steps': qualifying}), flush=True)

full_step = arm_results['full_intensity']['selected_step']
model.load_state_dict(torch.load(run / 'full_intensity' /
    f'joint_stage1_step_{full_step:05d}.pt', map_location='cpu', weights_only=True)['model'],
    strict=True)
atlas_shifted = atlas.clone()
atlas_shifted[0] = atlas[0].roll(100, 0)
swapped_metrics, swapped_cases = inner_readout(atlas_shifted)
removed_metrics, removed_cases = inner_readout(atlas, remove_visual=True)
with (run / 'selected_full_ablations.json').open('w') as stream:
    json.dump({'atlas_intensity_ap_shift_100_voxels': {
        'metrics': swapped_metrics, 'cases': swapped_cases},
        'source_visual_features_zero': {'metrics': removed_metrics,
                                        'cases': removed_cases}}, stream, indent=2,
        allow_nan=False)

full = arm_results['full_intensity']
support = arm_results['support_only']
full_metrics = full['checkpoint_metrics'][full['selected_step']]
support_metrics = support['checkpoint_metrics'][support['selected_step']]
stage1_pass = (full['selected_step'] > 0 and
    full_metrics['near_win_fraction'] >= full['checkpoint_metrics'][0]['near_win_fraction'] + .05 and
    full_metrics['near_win_fraction'] >= support_metrics['near_win_fraction'] + .10 and
    bool(full['qualified_steps']) and bool(support['qualified_steps']))
summary = {'stage1_passed': bool(stage1_pass), 'train_attempts': len(train_attempts),
    'train_near_sections': len(train_cases),
    'train_pair_sections': sum(r['wrong_branch_id'] is not None for r in train_cases),
    'inner_attempts': len(inner_attempts), 'inner_pair_sections': len(inner_cases),
    'arms': arm_results, 'full_atlas_shift': swapped_metrics,
    'full_source_visual_zero': removed_metrics,
    'interpretation_limit': 'both arms inherit parent132 full-intensity weights; no DEV/final/real truth'}
for label, rows in (('train', train_attempts), ('inner', inner_attempts)):
    summary[f'{label}_under8pct'] = {
        'attempts': sum(r['valid_fraction'] < .08 for r in rows),
        'eligible': sum(r['valid_fraction'] < .08 and r['eligible'] for r in rows),
        'near': sum(r['valid_fraction'] < .08 and r['near_branch_id'] is not None for r in rows),
        'pairs': sum(r['valid_fraction'] < .08 and r['wrong_branch_id'] is not None for r in rows)}
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
outputs = [p for p in run.rglob('*') if p.is_file()]
receipt = {'status': 'complete', 'stage1_passed': bool(stage1_pass),
    'source_sha256': source_sha['train_joint_feedback_161_stage1_pilot.py'],
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'output_sha256': {str(p.relative_to(run)).replace('\\', '/'): sha(p) for p in outputs},
    'summary': summary}
(run / 'completed.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'run': str(run),
    'stage1_passed': bool(stage1_pass), 'full_selected': full['selected_step'],
    'support_selected': support['selected_step'],
    'full_pair_wins': full_metrics['near_wins'],
    'support_pair_wins': support_metrics['near_wins']}), flush=True)
