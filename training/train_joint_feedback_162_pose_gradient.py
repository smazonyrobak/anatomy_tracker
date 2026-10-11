"""Paired TRAIN-only fitter-to-direct-pose gradient experiment from 132 step 2000."""

import hashlib
import json
import math
import os
import sys
import time
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
from training.joint_pose_map_121 import joint_fit_loss_121, joint_forward_121, joint_target_loss_121


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ON_POLICY_JOINT_FEEDBACK_162_POSE_GRADIENT_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_checkpoint = parent_dir / 'joint_step_02000.pt'
mode = os.environ.get('JOINT162_MODE')
assert mode in ('gradient_preflight', 'train')
if mode == 'train':
    assert os.environ.get('JOINT162_TRAIN_APPROVED') == '1', 'training awaits design approval'
run = root / ('runs/joint_feedback_162_gradient_preflight_001' if mode == 'gradient_preflight'
              else 'runs/joint_feedback_162_pose_gradient_001')
seed, side, sites, updates, inner_count = 202610111622, 256, 128, 2000, 256
checkpoints = (0, 500, 1000, 1500, 2000)
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

source_names = ('train_joint_feedback_162_pose_gradient.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
    'arbitrary_plane_ribbon_v6.py', 'whole_slice_atlas_feedback_083.py',
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
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda()
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
parent_weights = saved['model']
model.load_state_dict(parent_weights, strict=True)
del saved
model.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
fitter_modules = (model.global_plane_matcher_120, model.atlas_encoder, model.pair,
                  model.warp_condition, model.warp, model.fit_quality_head)
for module in (*pose_modules, *shared_modules, *fitter_modules):
    module.requires_grad_(True)
assert not any(p.requires_grad for p in model.warp_shared.parameters())
groups = [
    {'params': [p for m in pose_modules for p in m.parameters()], 'base_lr': 2e-5},
    {'params': [p for m in shared_modules for p in m.parameters()], 'base_lr': 5e-6},
    {'params': [p for m in fitter_modules for p in m.parameters()], 'base_lr': 2e-5},
]
parameters = [p for group in groups for p in group['params']]
axis = torch.arange(side, device='cuda') / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart256 = torch.stack((xx, yy), -1)
axis24 = (torch.arange(24, device='cuda') + .5) / 24 - .5 / side
yy24, xx24 = torch.meshgrid(axis24, axis24, indexing='ij')
chart24 = torch.stack((xx24, yy24), -1).reshape(-1, 2)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)


def draw(row):
    with torch.no_grad():
        original = sample_panel_geometry_train_151(context, [row['base_index']],
                                                   row['draw_seed'], side)
        return sample_one_shot_slide_artifacts_v4(context, [row['base_index']],
            row['draw_seed'], side, source_sample=original)


def direct_loss(prediction, sample, site_seed):
    state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    valid = sample['valid_mask'].flatten(1).float()
    assert bool(valid.sum()), 'zero-valid section has no identifiable pose target'
    generator = torch.Generator(device='cuda').manual_seed(site_seed)
    pixel = torch.multinomial(valid, sites, replacement=True, generator=generator)
    chart = torch.stack((pixel.remainder(side),
        pixel.div(side, rounding_mode='floor')), -1).float() / side
    reference = rigid_points_090(sample['state'], sample['reflection'], chart)
    proposed = rigid_points_090(state, flags, chart[:, None, None])
    reference_five = rigid_points_090(sample['state'], sample['reflection'], corners)
    proposed_five = rigid_points_090(state, flags, corners)
    tissue_mm = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
    five_mm = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
    rigid_mm = (.75 * tissue_mm + .25 * five_mm) / 1000
    predicted_normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
    normal_penalty = 4 * (1 - (predicted_normal * true_normal[:, None]
        ).sum(-1).abs().clamp_max(1))
    physical = (rigid_mm + normal_penalty[..., None]).flatten(1)
    rigid_mm = rigid_mm.flatten(1)
    target_mass = F.softmax(-physical.detach() / .7, -1)
    nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
    near_mode = model.base_modes + nearest.indices
    near_cost = physical.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    direct = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
        + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
        + .5 * physical[:, :2 * model.base_modes].min(-1).values
        + (neighbourhood * near_cost).sum(-1)).mean()
    return direct, rigid_mm.detach(), physical.detach(), prior.detach(), pixel


def candidate_record(sample, prediction):
    row = {'near_branch_id': None, 'wrong_branch_id': None,
        'near_rigid_mm': None, 'wrong_rigid_mm': None, 'support_dice': None,
        'support_area_gap': None, 'common_support_cells': None,
        'beam_branch_ids': [], 'beam_rigid_mm': []}
    if not bool(sample['eligible'][0]):
        return row
    with torch.no_grad():
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
        observed = chart256[sample['valid_mask'][0]]
        truth = rigid_points_090(sample['state'], sample['reflection'], observed)[0]
        proposed = rigid_points_090(state, reflection, observed)[0]
        cost = (proposed - truth).norm(dim=-1).mean(-1) / 1000
        row['beam_branch_ids'], row['beam_rigid_mm'] = ids.tolist(), cost.tolist()
        near = int(cost.argmin())
        if float(cost[near]) > 1.5:
            return row
        row['near_branch_id'], row['near_rigid_mm'] = int(ids[near]), float(cost[near])
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
            row.update(wrong_branch_id=int(ids[slot]), wrong_rigid_mm=float(cost[slot]),
                support_dice=float(dice[slot]), support_area_gap=float(gap[slot]),
                common_support_cells=int(overlap[slot]))
    return row


def fitted_loss(sample, prediction, branch, pixel):
    ids = [branch['near_branch_id']] + ([] if branch['wrong_branch_id'] is None
                                        else [branch['wrong_branch_id']])
    modes = torch.tensor([[index // 2 for index in ids]], device='cuda')
    reflections = torch.tensor([[index % 2 for index in ids]], device='cuda')
    states = prediction['state'][:, modes[0]]
    joint = joint_forward_121(model, prediction, sample['inputs'], modes, reflections,
        atlas, sample['offsets'], sample['weights'], candidate_state=states,
        match_side=24, map_side=96)
    target = joint_target_loss_121(joint['mapped'], sample['valid_mask'], sample['centre'],
        sample['state'], sample['reflection'], pixel)
    appearance = joint_fit_loss_121(sample['inputs'], joint['mapped'], atlas,
        sample['weights'], sample['valid_mask'], side=64)
    local = joint['mapped']['local_displacement_um'].float() / 1000
    warp = (local.square().mean((2, 3, 4))
        + .3 * (local[..., 1:] - local[..., :-1]).square().mean((2, 3, 4))
        + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean((2, 3, 4)))
    correction = (joint['match']['update'][..., 3:6] / 2000).square().sum(-1)
    measured = (appearance['fit_loss'] + .5 * appearance['coverage_penalty']
                + .05 * warp + .05 * correction)
    difficulty = joint['mapped']['fit_energy'].float() + .05 * warp + .05 * correction
    spatial = .5 * target['map_loss'][0, 0] + .25 * target['rigid_loss'][0, 0]
    feedback = .1 * measured[0, 0]
    mass_term = measured.new_zeros(())
    if len(ids) == 2:
        feedback = feedback + .25 * F.softplus(.1 + measured[0, 0] - measured[0, 1])
        prior = joint['match']['input_score']
        mass_term = .1 * F.softplus(.1 + (prior[0, 1] - measured[0, 1])
            - (prior[0, 0] - measured[0, 0]))
        feedback = feedback + mass_term
    calibration_target = (appearance['fit_loss']
        + .5 * appearance['coverage_penalty']).detach()
    calibration = .2 * F.smooth_l1_loss(joint['mapped']['fit_energy'], calibration_target)
    return spatial, feedback, calibration, target, appearance, measured, difficulty, warp, mass_term


def norm(gradient):
    return 0. if gradient is None else float(gradient.norm())


def optimizer_for_arm():
    return torch.optim.AdamW([{'params': group['params'], 'lr': group['base_lr'],
        'base_lr': group['base_lr']} for group in groups], weight_decay=1e-4)


config = {'seed': seed, 'side': side, 'valid_sites': sites, 'updates_per_arm': updates,
    'inner_sections': inner_count, 'checkpoints': checkpoints, 'train_bases': list(range(60)),
    'inner_bases': list(range(60, 64)), 'parent_checkpoint_sha256': sha(parent_checkpoint),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol), 'source_sha256': source_sha,
    'synthetic_provenance': context['provenance'],
    'trainable': 'pose/anchor_pose/anchor_global; encoder/lateral; native global matcher, atlas encoder, pair, warp condition/warp, fit-quality head',
    'frozen': 'warp_shared, fitted/candidate rankers, uncertainty, all other modules',
    'loss_weights': {'direct': 'min(1, valid_fraction/.02)', 'common_near_map': .5,
        'common_near_corrected_rigid': .25, 'measured_near_difficulty': .1,
        'measured_pair_margin': .25, 'measured_pair_mass': .1,
        'detached_head_calibration': .2,
        'measured_difficulty': 'LNCC + .5 coverage + .05 warp + .05 correction',
        'sparse_fitter_reliability': 'min(1, valid_fraction/.08)'},
    'fitter_control': 'common synthetic spatial gradients reach pose in both arms; measured feedback reaches pose only in full; detached control copies every prediction tensor before feedback',
    'calibrated': False, 'public_benchmark_used': False, 'development_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))

if mode == 'gradient_preflight':
    model.train()
    rng = np.random.default_rng(seed + 9)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(64):
        row = {'base_index': int(rng.integers(60)),
               'draw_seed': int(rng.integers(0, 2**63 - 1, dtype=np.int64)),
               'site_seed': int(rng.integers(0, 2**63 - 1, dtype=np.int64))}
        sample = draw(row)
        with torch.no_grad():
            selected = candidate_record(sample, model.predict(sample['inputs']))
        if (selected['wrong_branch_id'] is not None
                and selected['near_branch_id'] // 2 != selected['wrong_branch_id'] // 2
                and float(sample['valid_mask'].float().mean()) >= .08):
            break
    assert (selected['wrong_branch_id'] is not None
        and selected['near_branch_id'] // 2 != selected['wrong_branch_id'] // 2
        and float(sample['valid_mask'].float().mean()) >= .08), \
        'no matched distinct-mode >=8%-tissue TRAIN pair in 64 attempts'
    prediction = model.predict(sample['inputs'])
    direct, _, _, _, pixel = direct_loss(prediction, sample, row['site_seed'])
    spatial, feedback, _, _, _, _, _, _, mass_term = fitted_loss(
        sample, prediction, selected, pixel)
    head = (model.pose[-1] if selected['near_branch_id'] // 2 < model.base_modes
            else model.anchor_pose[-1])
    encoder = model.encoder[0][0].weight
    lateral = model.lateral[0].weight
    matcher = model.global_plane_matcher_120['atlas'][0].weight
    full_grad = torch.autograd.grad(feedback,
        (prediction['state'], prediction['log_mass'], head.weight, encoder,
         lateral, matcher),
        retain_graph=True, allow_unused=True)
    direct_grad = torch.autograd.grad(direct, (head.weight, encoder),
        retain_graph=True, allow_unused=True)
    spatial_grad = torch.autograd.grad(spatial, (head.weight, encoder), allow_unused=True)
    selected_state_grad = full_grad[0][0, selected['near_branch_id'] // 2]
    selected_mass_grad = full_grad[1][0, selected['near_branch_id'] // 2]
    near_mode = selected['near_branch_id'] // 2
    if near_mode < model.base_modes:
        head_offset, coordinate_rows, mass_row = 21 * near_mode, 12, 19
    else:
        head_offset, coordinate_rows, mass_row = 18 * (near_mode - model.base_modes), 9, 16
    coordinate_head_grad = full_grad[2][head_offset:head_offset + coordinate_rows]
    mass_head_grad = full_grad[2][head_offset + mass_row]
    feedback_value, mass_value = float(feedback.detach()), float(mass_term.detach())
    detached_prediction = {key: value.detach() if torch.is_tensor(value) else value
                           for key, value in prediction.items()}
    del prediction, direct, spatial, feedback, mass_term
    assert not any(value.requires_grad for value in detached_prediction.values()
                   if torch.is_tensor(value))
    _, detached_feedback, detached_calibration, _, _, _, _, _, _ = fitted_loss(
        sample, detached_prediction, selected, pixel)
    assert abs(feedback_value - float(detached_feedback.detach())) < 1e-5
    detached_grad = torch.autograd.grad(detached_feedback,
        (head.weight, encoder, lateral, matcher), retain_graph=True, allow_unused=True)
    calibration_grad = torch.autograd.grad(detached_calibration,
        (model.fit_quality_head[-1].weight, matcher), allow_unused=True)
    assert all(math.isfinite(norm(g)) for g in (*full_grad, *direct_grad, *spatial_grad,
                                               *detached_grad, *calibration_grad))
    assert all(norm(g) > 0 for g in (*direct_grad, *spatial_grad, full_grad[1],
                                    full_grad[2], full_grad[3], full_grad[4],
                                    full_grad[5], detached_grad[3],
                                    calibration_grad[0]))
    assert norm(selected_state_grad[:3]) > 0 and norm(selected_state_grad[3:]) > 0
    assert norm(coordinate_head_grad) > 0 and norm(mass_head_grad) > 0
    assert norm(selected_mass_grad) > 0 and mass_value > 0
    assert norm(detached_grad[0]) == norm(detached_grad[1]) == norm(detached_grad[2]) == 0.
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    adam_state_bytes = 8 * sum(p.numel() for p in parameters)
    assert peak + adam_state_bytes < 9.5 * 1024**3, 'preflight plus Adam states exceeds 9.5 GiB'
    audit = {'mode': mode, 'attempts': attempt + 1, 'draw': row,
        'physical_section_id': sample['provenance'][0]['physical_section_id'],
        'animal_id': sample['provenance'][0]['base_lineage']['animal_id'],
        'branch': selected, 'probe_order': ['selected_state', 'direct_log_mass',
            'selected_pose_head', 'shared_encoder', 'shared_lateral', 'atlas_matcher'],
        'full_measured_feedback_gradient_norm': [norm(g) for g in full_grad],
        'full_selected_state_centre_gradient_norm': norm(selected_state_grad[:3]),
        'full_selected_state_frame_basis_gradient_norm': norm(selected_state_grad[3:]),
        'full_selected_pose_head_coordinate_rows_gradient_norm': norm(coordinate_head_grad),
        'full_selected_pose_head_mass_row_gradient_norm': norm(mass_head_grad),
        'full_selected_mode_log_mass_gradient': float(selected_mass_grad),
        'fit_mass_loss': mass_value,
        'direct_gradient_norm': [norm(g) for g in direct_grad],
        'common_spatial_head_encoder_gradient_norm': [norm(g) for g in spatial_grad],
        'detached_measured_feedback_gradient_norm': [norm(g) for g in detached_grad],
        'detached_head_calibration_gradient_norm': [norm(g) for g in calibration_grad],
        'measured_feedback_to_direct_pose_head_ratio': norm(full_grad[2]) / norm(direct_grad[0]),
        'peak_cuda_allocated_bytes': peak,
        'estimated_adam_state_bytes': adam_state_bytes,
        'peak_plus_estimated_adam_state_bytes': peak + adam_state_bytes,
        'elapsed_seconds': time.perf_counter() - started,
        'parent_checkpoint_sha256': sha(parent_checkpoint), 'source_sha256': source_sha,
        'protocol_sha256': sha(protocol), 'synthetic_provenance': context['provenance']}
    run.mkdir(parents=True, exist_ok=False)
    (run / 'gradient_preflight.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps({'event': 'gradient_preflight_complete', 'run': str(run),
        'head_ratio': audit['measured_feedback_to_direct_pose_head_ratio'],
        'peak_cuda_gib': peak / 1024**3, 'seconds': audit['elapsed_seconds']}), flush=True)
    sys.exit(0)

rng = np.random.default_rng(seed)
schedule = []
for split, count, low, high in (('train', updates, 0, 60),
                                ('inner', inner_count, 60, 64)):
    for index in range(count):
        schedule.append({'split': split, 'index': index,
            'base_index': int(rng.integers(low, high)),
            'draw_seed': int(rng.integers(0, 2**63 - 1, dtype=np.int64)),
            'site_seed': int(rng.integers(0, 2**63 - 1, dtype=np.int64))})
assert len({row['draw_seed'] for row in schedule}) == len(schedule)
seen = set()
exact_remaining = 64
for row in schedule:
    sample = draw(row)
    provenance = sample['provenance'][0]
    assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
    assert provenance['base_index'] == row['base_index']
    key = json.dumps((row['base_index'], provenance['physical_ouv_um'],
        provenance['mode'], provenance['appearance'],
        provenance['one_shot_slide_artifacts_v4']['exposure']))
    assert key not in seen
    seen.add(key)
    row['physical_section_id'] = provenance['physical_section_id']
    row['animal_id'] = provenance['base_lineage']['animal_id']
    row['valid_pixels'] = int(sample['valid_mask'][0].sum())
    row['valid_fraction'] = row['valid_pixels'] / (side * side)
    row['brain_intersecting'] = row['valid_pixels'] > 0
    row['eligible'] = bool(sample['eligible'][0])
    row['exposure'] = provenance['one_shot_slide_artifacts_v4']['exposure']
    row['low_exposure'] = provenance['one_shot_slide_artifacts_v4']['low_exposure']
    row['appearance_mode'] = provenance['mode']
    row['artifact_events'] = provenance['one_shot_slide_artifacts_v3']['events']
    normal = np.asarray(provenance['physical_unit_normal_ap_dv_ml'])
    row['nearest_cardinal_angle_deg'] = float(np.degrees(np.arccos(
        np.clip(np.abs(normal).max(), 0, 1))))
    row['exact_diagnostic'] = bool(row['split'] == 'inner' and row['eligible']
                                   and exact_remaining > 0)
    exact_remaining -= int(row['exact_diagnostic'])
    row['provenance'] = provenance
assert exact_remaining == 0
assert len({row['physical_section_id'] for row in schedule}) == len(schedule)
train_rows = schedule[:updates]
inner_rows = schedule[updates:]
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
with (run / 'draws.jsonl').open('w') as stream:
    for row in schedule:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
print(json.dumps({'event': 'draws_frozen', 'train_sections': len(train_rows),
    'inner_sections': len(inner_rows), 'sparse_train': sum(r['valid_fraction'] < .08
        for r in train_rows), 'zero_valid_train': sum(r['valid_fraction'] == 0
        for r in train_rows)}), flush=True)


def inner_readout():
    model.eval()
    cases = []
    with torch.inference_mode():
        for row in inner_rows:
            sample = draw(row)
            assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
            case = {'physical_section_id': row['physical_section_id'],
                'animal_id': row['animal_id'], 'valid_pixels': row['valid_pixels'],
                'valid_fraction': row['valid_fraction'],
                'brain_intersecting': row['brain_intersecting'],
                'nearest_cardinal_angle_deg': row['nearest_cardinal_angle_deg'],
                'exposure': row['exposure'], 'low_exposure': row['low_exposure'],
                'appearance_mode': row['appearance_mode'],
                'artifact_events': row['artifact_events'],
                'eligible': row['eligible'], 'top1_rigid_mm': None,
                'best160_rigid_mm': None, 'near_capture': False,
                'pair_available': False, 'near_map_mm': None,
                'near_corrected_rigid_mm': None, 'near_difficulty': None,
                'wrong_difficulty': None, 'pair_win': None,
                'near_measured_difficulty': None, 'wrong_measured_difficulty': None,
                'measured_pair_win': None, 'exact_map_mm': None}
            if not row['brain_intersecting']:
                cases.append(case)
                continue
            prediction = model.predict(sample['inputs'])
            _, rigid, _, prior, pixel = direct_loss(prediction, sample, row['site_seed'])
            branch = candidate_record(sample, prediction)
            case.update(top1_rigid_mm=float(rigid[0, prior[0].argmax()]),
                best160_rigid_mm=float(rigid.min()),
                near_capture=branch['near_branch_id'] is not None,
                pair_available=branch['wrong_branch_id'] is not None,
                **branch)
            if branch['near_branch_id'] is not None:
                _, _, _, target, appearance, measured, difficulty, warp, mass_term = fitted_loss(
                    sample, prediction, branch, pixel)
                case.update(near_map_mm=float(target['map_loss'][0, 0]),
                    near_corrected_rigid_mm=float(target['rigid_loss'][0, 0]),
                    near_difficulty=float(difficulty[0, 0]),
                    near_measured_difficulty=float(measured[0, 0]),
                    near_image_atlas_fit=float(appearance['fit_loss'][0, 0]),
                    near_atlas_coverage=float(appearance['atlas_coverage'][0, 0]),
                    near_warp_cost=float(warp[0, 0]), fit_mass_loss=float(mass_term))
                if branch['wrong_branch_id'] is not None:
                    case['wrong_difficulty'] = float(difficulty[0, 1])
                    case['pair_win'] = bool(difficulty[0, 0] < difficulty[0, 1])
                    case['wrong_measured_difficulty'] = float(measured[0, 1])
                    case['measured_pair_win'] = bool(measured[0, 0] < measured[0, 1])
            if row['exact_diagnostic']:
                index = torch.zeros((1, 1), device='cuda', dtype=torch.long)
                reflection = sample['reflection'][:, None].long()
                exact = joint_forward_121(model, prediction, sample['inputs'], index,
                    reflection, atlas, sample['offsets'], sample['weights'],
                    candidate_state=sample['state'][:, None], match_side=24, map_side=96)
                exact_target = joint_target_loss_121(exact['mapped'], sample['valid_mask'],
                    sample['centre'], sample['state'], sample['reflection'], pixel)
                case['exact_map_mm'] = float(exact_target['map_loss'][0, 0])
            cases.append(case)
    scored = [case for case in cases if case['brain_intersecting']]
    sparse_scored = [case for case in scored if case['valid_fraction'] < .08]
    assert scored
    near_maps = [case['near_map_mm'] for case in scored if case['near_map_mm'] is not None]
    pair_wins = [case['pair_win'] for case in cases if case['pair_win'] is not None]
    measured_pair_wins = [case['measured_pair_win'] for case in cases
                          if case['measured_pair_win'] is not None]
    metrics = {'attempt_n': len(cases), 'scored_n': len(scored),
        'zero_valid_n': len(cases) - len(scored),
        'direct_top1_rigid_mm': float(np.mean([case['top1_rigid_mm'] for case in scored])),
        'best160_rigid_mm': float(np.mean([case['best160_rigid_mm'] for case in scored])),
        'sparse_nonzero_n': len(sparse_scored),
        'sparse_direct_top1_rigid_mm': (float(np.mean([case['top1_rigid_mm']
            for case in sparse_scored])) if sparse_scored else None),
        'blind16_near_fraction_brain_intersecting': float(np.mean(
            [case['near_capture'] for case in scored])),
        'near_n': sum(case['near_capture'] for case in cases),
        'near_map_mm': float(np.mean(near_maps)) if near_maps else None,
        'pair_n': sum(case['pair_available'] for case in cases),
        'pair_win_fraction': float(np.mean(pair_wins)) if pair_wins else None,
        'measured_pair_win_fraction': (float(np.mean(measured_pair_wins))
            if measured_pair_wins else None),
        'exact_n': sum(case['exact_map_mm'] is not None for case in cases),
        'exact_map_mm': float(np.mean([case['exact_map_mm'] for case in cases
            if case['exact_map_mm'] is not None]))}
    strata = {'tissue_under8pct': [case for case in cases if case['valid_fraction'] < .08],
        'tissue_at_least8pct': [case for case in cases if case['valid_fraction'] >= .08],
        'angle_under15deg': [case for case in cases
            if case['nearest_cardinal_angle_deg'] < 15],
        'angle_15_to30deg': [case for case in cases
            if 15 <= case['nearest_cardinal_angle_deg'] < 30],
        'angle_at_least30deg': [case for case in cases
            if case['nearest_cardinal_angle_deg'] >= 30],
        'exposure_under015': [case for case in cases if case['exposure'] < .15],
        'exposure_015_to030': [case for case in cases if .15 <= case['exposure'] < .30],
        'exposure_at_least030': [case for case in cases if case['exposure'] >= .30]}
    metrics['strata'] = {name: {'attempt_n': len(rows),
        'scored_n': sum(case['brain_intersecting'] for case in rows),
        'zero_valid_n': sum(not case['brain_intersecting'] for case in rows),
        'direct_top1_rigid_mm': (float(np.mean([case['top1_rigid_mm'] for case in rows
            if case['brain_intersecting']])) if any(case['brain_intersecting']
            for case in rows) else None),
        'near_n': sum(case['near_capture'] for case in rows),
        'pair_n': sum(case['pair_available'] for case in rows),
        'near_map_mm': (float(np.mean([case['near_map_mm'] for case in rows
            if case['near_map_mm'] is not None])) if any(case['near_capture'] for case in rows)
            else None)} for name, rows in strata.items()}
    return metrics, cases


arm_results = {}
for arm in ('full', 'detached'):
    torch.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 3)
    model.load_state_dict(parent_weights, strict=True)
    optimizer = optimizer_for_arm()
    directory = run / arm
    directory.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    records = {}
    optimizer_updates = 0
    started = time.perf_counter()
    with (directory / 'training.jsonl').open('w') as training, \
            (directory / 'inner_scores.jsonl').open('w') as scores:
        for step in range(updates + 1):
            if step in checkpoints:
                metrics, cases = inner_readout()
                records[step] = {'metrics': metrics, 'cases': cases}
                scores.write(json.dumps({'step': step, 'metrics': metrics,
                    'cases': cases}, allow_nan=False) + '\n')
                scores.flush()
                path = directory / f'joint_step_{step:05d}.pt'
                temp = path.with_suffix('.tmp')
                torch.save({'step': step, 'optimizer_updates': optimizer_updates,
                    'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config,
                    'calibrated': False}, temp)
                os.replace(temp, path)
                print(json.dumps({'event': 'inner_checkpoint', 'arm': arm,
                    'step': step, 'metrics': metrics}), flush=True)
            if step == updates:
                break
            row = train_rows[step]
            sample = draw(row)
            assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
            if not row['brain_intersecting']:
                training.write(json.dumps({'step': step + 1,
                    'optimizer_updates': optimizer_updates,
                    'physical_section_id': row['physical_section_id'],
                    'animal_id': row['animal_id'], 'valid_pixels': 0,
                    'valid_fraction': 0., 'brain_intersecting': False,
                    'eligible': False, 'skipped_reason': 'zero_valid_unidentifiable_pose',
                    'direct_loss': None, 'near_branch_id': None,
                    'wrong_branch_id': None, 'fitter_loss': None}, allow_nan=False) + '\n')
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(sample['inputs'])
            direct, rigid, _, prior, pixel = direct_loss(prediction, sample, row['site_seed'])
            top1_rigid = float(rigid[0, prior[0].argmax()])
            detached_prediction = {key: value.detach() if torch.is_tensor(value) else value
                                   for key, value in prediction.items()}
            branch = candidate_record(sample, detached_prediction)
            direct_value = float(direct.detach())
            direct_reliability = min(1., row['valid_fraction'] / .02)
            (direct * direct_reliability).backward()
            del prediction, direct
            spatial_value = None
            feedback_value = None
            calibration_value = None
            mass_value = None
            near_map = None
            if branch['near_branch_id'] is not None:
                fit_prediction = model.predict(sample['inputs'])
                spatial, feedback, _, target, _, measured, difficulty, _, mass_term = fitted_loss(
                    sample, fit_prediction, branch, pixel)
                spatial_value = float(spatial.detach())
                feedback_value = float(feedback.detach())
                mass_value = float(mass_term.detach())
                near_map = float(target['map_loss'][0, 0].detach())
                reliability = min(1., row['valid_fraction'] / .08)
                if arm == 'full':
                    (reliability * (spatial + feedback)).backward()
                else:
                    (reliability * spatial).backward()
                detached_fit_prediction = {key: value.detach() if torch.is_tensor(value)
                    else value for key, value in fit_prediction.items()}
                assert not any(value.requires_grad for value in detached_fit_prediction.values()
                               if torch.is_tensor(value))
                _, detached_feedback, detached_calibration, _, _, _, _, _, _ = fitted_loss(
                    sample, detached_fit_prediction, branch, pixel)
                calibration_value = float(detached_calibration.detach())
                if arm == 'full':
                    (reliability * detached_calibration).backward()
                else:
                    (reliability * (detached_feedback + detached_calibration)).backward()
            decay = .2 + .8 * .5 * (1 + math.cos(math.pi * (step + 1) / updates))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * decay
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            optimizer_updates += 1
            training_row = {'step': step + 1, 'physical_section_id': row['physical_section_id'],
                'optimizer_updates': optimizer_updates, 'animal_id': row['animal_id'],
                'valid_pixels': row['valid_pixels'], 'valid_fraction': row['valid_fraction'],
                'brain_intersecting': True,
                'eligible': row['eligible'], 'sparse_under8pct': row['valid_fraction'] < .08,
                'top1_rigid_mm': top1_rigid, 'direct_loss': direct_value,
                'direct_reliability': direct_reliability,
                **branch,
                'near_map_mm': near_map, 'common_spatial_loss': spatial_value,
                'measured_feedback_loss': feedback_value,
                'detached_head_calibration_loss': calibration_value,
                'fit_mass_loss': mass_value,
                'fitter_reliability': min(1., row['valid_fraction'] / .08),
                'gradient_norm': float(gradient)}
            training.write(json.dumps(training_row, allow_nan=False) + '\n')
            if (step + 1) % 100 == 0:
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    **training_row}), flush=True)
    baseline = records[0]['metrics']
    qualifying = [step for step in checkpoints if
        baseline['sparse_nonzero_n'] > 0 and
        records[step]['metrics']['exact_map_mm'] <= baseline['exact_map_mm'] + .10 and
        records[step]['metrics']['sparse_direct_top1_rigid_mm'] <=
            baseline['sparse_direct_top1_rigid_mm'] + .25]
    selected = min(qualifying, key=lambda step: (
        records[step]['metrics']['direct_top1_rigid_mm'], step)) if qualifying else 0
    arm_results[arm] = {'selected_step': selected, 'qualified_steps': qualifying,
        'optimizer_updates': optimizer_updates,
        'checkpoint_metrics': {step: records[step]['metrics'] for step in checkpoints},
        'selected_cases': records[selected]['cases'],
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': time.perf_counter() - started}
    print(json.dumps({'event': 'arm_selected', 'arm': arm,
        'selected_step': selected, 'qualified_steps': qualifying}), flush=True)

full = arm_results['full']
detached = arm_results['detached']
assert full['optimizer_updates'] == detached['optimizer_updates'] == sum(
    row['brain_intersecting'] for row in train_rows)
full_cases = {case['physical_section_id']: case for case in full['selected_cases']}
detached_cases = {case['physical_section_id']: case for case in detached['selected_cases']}
assert full_cases.keys() == detached_cases.keys()
common_near = [key for key in full_cases if full_cases[key]['near_capture']
               and detached_cases[key]['near_capture']]
common_pairs = [key for key in full_cases if full_cases[key]['pair_available']
                and detached_cases[key]['pair_available']]
f = full['checkpoint_metrics'][full['selected_step']]
d = detached['checkpoint_metrics'][detached['selected_step']]
near_map_delta = (float(np.mean([detached_cases[key]['near_map_mm']
    - full_cases[key]['near_map_mm'] for key in common_near])) if common_near else None)
sparse_delta = (f['sparse_direct_top1_rigid_mm'] - d['sparse_direct_top1_rigid_mm']
    if f['sparse_nonzero_n'] and d['sparse_nonzero_n'] else None)
pair_full = (float(np.mean([full_cases[key]['pair_win'] for key in common_pairs]))
             if common_pairs else None)
pair_detached = (float(np.mean([detached_cases[key]['pair_win'] for key in common_pairs]))
                 if common_pairs else None)
train_gate = (full['selected_step'] > 0 and
    d['direct_top1_rigid_mm'] - f['direct_top1_rigid_mm'] >= .15 and
    f['direct_top1_rigid_mm'] <= .95 * d['direct_top1_rigid_mm'] and
    bool(common_near) and near_map_delta >= .10 and
    sparse_delta is not None and sparse_delta <= .25 and
    f['blind16_near_fraction_brain_intersecting'] >=
        d['blind16_near_fraction_brain_intersecting'] - .05 and
    bool(full['qualified_steps']) and bool(detached['qualified_steps']))
summary = {'train_gate_passed': bool(train_gate),
    'paired_optimizer_updates': full['optimizer_updates'],
    'train_attempts': len(train_rows), 'inner_attempts': len(inner_rows),
    'selected_steps': {'full': full['selected_step'], 'detached': detached['selected_step']},
    'all_section_top1_rigid_mm': {'full': f['direct_top1_rigid_mm'],
        'detached': d['direct_top1_rigid_mm']},
    'common_near_n': len(common_near), 'detached_minus_full_common_near_map_mm': near_map_delta,
    'common_pair_n': len(common_pairs), 'common_pair_win_fraction':
        {'full': pair_full, 'detached': pair_detached},
    'sparse_full_minus_detached_top1_mm': sparse_delta,
    'blind16_near_fraction_brain_intersecting': {
        'full': f['blind16_near_fraction_brain_intersecting'],
        'detached': d['blind16_near_fraction_brain_intersecting']},
    'train_sparse_nonzero_sections': sum(0 < row['valid_fraction'] < .08
        for row in train_rows),
    'train_zero_valid_sections': sum(row['valid_fraction'] == 0 for row in train_rows),
    'inner_sparse_sections': sum(row['valid_fraction'] < .08 for row in inner_rows),
    'arm_peak_cuda_allocated_bytes': {name: result['peak_cuda_allocated_bytes']
        for name, result in arm_results.items()},
    'arm_elapsed_seconds': {name: result['elapsed_seconds']
        for name, result in arm_results.items()},
    'no_development_or_public_data': True}
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
outputs = [path for path in run.rglob('*') if path.is_file()]
(run / 'completed.json').write_text(json.dumps({'status': 'complete',
    'train_gate_passed': bool(train_gate), 'summary': summary,
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'source_sha256': source_sha, 'protocol_sha256': sha(protocol),
    'output_sha256': {str(path.relative_to(run)).replace('\\', '/'): sha(path)
        for path in outputs}}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'run': str(run),
    'train_gate_passed': bool(train_gate), 'summary': summary}), flush=True)
