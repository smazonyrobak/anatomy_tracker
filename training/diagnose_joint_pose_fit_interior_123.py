"""TRAIN-only frozen-122 anatomical-fit comparison; protocol 123, no training."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interior(mask):
    return F.avg_pool2d(mask.float()[None, None], 17, 1, 8)[0, 0] > .999


def highpass_loss(source, target, mask):
    a = source - F.avg_pool2d(source, 9, 1, 4)
    b = target - F.avg_pool2d(target, 9, 1, 4)
    cross = F.avg_pool2d(a * b, 9, 1, 4)
    variance = F.avg_pool2d(a.square(), 9, 1, 4) * F.avg_pool2d(b.square(), 9, 1, 4)
    corr = (cross / (variance + 1e-5).sqrt()).abs().clamp(0, 1)
    return ((1 - corr) * mask).sum((1, 2, 3)) / mask.sum()


def raw_121_loss(source, target, support, valid, mask):
    weight = valid * support
    amount = F.avg_pool2d(weight, 9, 1, 4).clamp_min(1e-4)
    mean_source = F.avg_pool2d(weight * source, 9, 1, 4) / amount
    mean_target = F.avg_pool2d(weight * target, 9, 1, 4) / amount
    covariance = F.avg_pool2d(weight * source * target, 9, 1, 4) / amount - mean_source * mean_target
    source_var = (F.avg_pool2d(weight * source.square(), 9, 1, 4) / amount - mean_source.square()).clamp_min(0)
    target_var = (F.avg_pool2d(weight * target.square(), 9, 1, 4) / amount - mean_target.square()).clamp_min(0)
    corr = (covariance / (source_var * target_var + 1e-5).sqrt()).abs().clamp(0, 1)
    return ((1 - corr) * mask).sum((1, 2, 3)) / mask.sum()


def pair_readout(image, valid, mapped, rendered, states, reflection, best, wrong):
    slots = [best, wrong]
    support = rendered[slots, 1:2].clamp(0, 1)
    mask = interior((support[:, 0] > .95).all(0))
    result = {'common_pixels': int(mask.sum())}
    if result['common_pixels'] < 256:
        return result
    source = F.interpolate(image[:, :1], (96, 96), mode='area')
    donor = F.interpolate(valid[:, None].float(), (96, 96), mode='area')
    target = rendered[slots, :1] / support.clamp_min(1e-4)
    hp = highpass_loss(source, target, mask)
    raw = raw_121_loss(source, target, support, donor, mask)
    zero_hp = highpass_loss(source, torch.zeros_like(target), mask)
    zero_raw = raw_121_loss(source, torch.zeros_like(target), support, donor, mask)
    local = mapped['local_displacement_um'][0, slots]
    displacement_rms = ((local.square().sum(1)[:, mask]).mean(1).sqrt() / 1000).tolist()
    points = torch.tensor([[0., 0.], [1 / 96, 0.], [0., 1 / 96]], device='cuda')
    rigid = rigid_points_090(states[:, slots], reflection[:, slots], points)[0]
    spacing_x = (rigid[:, 1] - rigid[:, 0]).norm(dim=-1)
    spacing_y = (rigid[:, 2] - rigid[:, 0]).norm(dim=-1)
    dx = (local[:, :, :, 1:] - local[:, :, :, :-1]).square().sum(1).sqrt() / spacing_x[:, None, None]
    dy = (local[:, :, 1:, :] - local[:, :, :-1, :]).square().sum(1).sqrt() / spacing_y[:, None, None]
    strain = (dx[:, :-1].square() + dy[:, :, :-1].square()).sqrt()
    edge = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1]
    result.update(hp_best=float(hp[0]), hp_wrong=float(hp[1]),
                  hp_margin=float(hp[1] - hp[0]),
                  raw_best=float(raw[0]), raw_wrong=float(raw[1]),
                  raw_margin=float(raw[1] - raw[0]),
                  zero_hp_margin=float(zero_hp[1] - zero_hp[0]),
                  zero_raw_margin=float(zero_raw[1] - zero_raw[0]),
                  displacement_rms_mm=displacement_rms,
                  strain_mean=[float(strain[i, edge].mean()) for i in range(2)],
                  strain_p95=[float(torch.quantile(strain[i, edge], .95)) for i in range(2)])
    return result


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_POSE_FIT_INTERIOR_123_PROTOCOL_20261009.md'
run = root / 'runs/joint_pose_map_122'
checkpoint_path = run / 'joint_step_02000.pt'
out = root / 'runs/joint_pose_fit_interior_123_train'
seed, side, count = 20261009123, 256, 128
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert done['updates'] == config['updates'] == 2000 and config['blind_count'] == 16
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(checkpoint_path) == done['checkpoint_sha256']['2000']
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
prior_ids = {row['physical_section_id'] for row in map(json.loads, (run / 'draws.jsonl').open())
             if 'physical_section_id' in row}
assert len(prior_ids) == done['synthetic_draw_attempts']

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
atlas = context['atlas']
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 2000 and checkpoint['config'] == config and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint

axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
translations = torch.zeros(6, 9, device='cuda')
translations[torch.arange(6, device='cuda'), torch.tensor([3, 3, 4, 4, 5, 5], device='cuda')] = \
    torch.tensor([-1600., 1600., -1600., 1600., -1600., 1600.], device='cuda')
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
rows = []
output_config = {'seed': seed, 'draw_seed_start': draw_seed, 'eligible_sections': count,
    'split': 'TRAIN synthetic only', 'checkpoint_step': 2000, 'map_side': 96,
    'checkpoint_sha256': sha(checkpoint_path), 'run_completion_sha256': sha(run / 'completed.json'),
    'run_config_sha256': sha(run / 'config.json'), 'run_draws_sha256': sha(run / 'draws.jsonl'),
    'protocol_sha256': sha(protocol), 'source_sha256': sha(__file__),
    'frozen_source_sha256': config['source_sha256'],
    'reference_sha256': {name: sha(source / name) for name in (
        'diagnose_v3_interior_intensity_energy_118.py', 'joint_pose_map_121.py',
        'evaluate_joint_pose_map_122.py')},
    'calibrated': False, 'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2))

with (out / 'draws.jsonl').open('w') as draw_stream, (out / 'rows.jsonl').open('w') as row_stream:
    while len(rows) < count:
        virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
        with torch.inference_mode():
            sampled = sample_one_shot_slide_artifacts_v3(context, virtual, draw_seed, side=side)
        record = sampled['provenance'][0]
        assert record['base_lineage']['split'] == 'train'
        assert record['physical_section_id'] not in prior_ids
        used = bool(sampled['eligible'][0])
        draw_stream.write(json.dumps({'draw_seed': draw_seed,
            'draw_attempt': draw_seed - seed * 1000000 + 1,
            'used': used, **record}, allow_nan=False) + '\n')
        if not used:
            draw_seed += 1
            continue

        sample_sha256 = hashlib.sha256(b''.join(sampled[key].detach().cpu().numpy().tobytes()
            for key in ('inputs', 'state', 'reflection', 'centre', 'valid_mask', 'offsets', 'weights'))).hexdigest()
        image, truth, truth_reflection = sampled['inputs'], sampled['state'], sampled['reflection'].long()
        valid, offsets, weights = sampled['valid_mask'], sampled['offsets'], sampled['weights']
        row = {'draw_seed': draw_seed, 'physical_section_id': record['physical_section_id'],
               'virtual_index': virtual[0], 'base_lineage': record['base_lineage'],
               'sample_sha256': sample_sha256, 'appearance_mode': record['mode'],
               'artifacts': record['one_shot_slide_artifacts_v3']['events']}

        with torch.inference_mode():
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            for _ in range(2):
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity = -similarity
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            mode, reflection = beam // 2, beam % 2
            match = global_plane_match(model, prediction, mode, reflection,
                offsets, weights, atlas, (side, side), side=24)
            target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
            rigid = rigid_points_090(match['state'], reflection, chart)[0]
            error = (rigid - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
            best = int(error.argmin())
            blind = {'beam_branch_ids': beam[0].tolist(), 'best_slot': best,
                     'best_rigid_mm': float(error[best]), 'status': 'no_near_blind_candidate'}
            if float(error[best]) <= 1.5:
                blind_map = model.map({**prediction, 'state': match['state']}, offsets,
                    torch.arange(16, device='cuda')[None], reflection, (96, 96), atlas,
                    weights, source_shape=(side, side))
                blind_render = render_finite_thickness_coordinate_grid(atlas,
                    blind_map['coordinates'].flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
                    weights.expand(16, -1))
                support = blind_render[:, 1] > .95
                overlap = (support & support[best]).sum((-2, -1)).float()
                dice = 2 * overlap / (support.sum((-2, -1)) + support[best].sum()).clamp_min(1)
                area_gap = (support.float().mean((-2, -1)) - support[best].float().mean()).abs()
                eligible = (error > error[best] + 1.) & (dice >= .85) & (area_gap <= .08)
                blind['eligible_wrong_count'] = int(eligible.sum())
                blind['status'] = 'no_support_matched_wrong'
                if bool(eligible.any()):
                    wrong = int(error.masked_fill(~eligible, torch.inf).argmin())
                    blind.update(wrong_slot=wrong, wrong_rigid_mm=float(error[wrong]),
                                 mapped_support_dice=float(dice[wrong]),
                                 mapped_support_area_gap=float(area_gap[wrong]))
                    readout = pair_readout(image, valid, blind_map, blind_render,
                        match['state'], reflection, best, wrong)
                    blind.update(readout)
                    blind['status'] = 'scored' if readout['common_pixels'] >= 256 else 'insufficient_common_interior'
                del blind_map, blind_render
            row['blind'] = blind

            control_states = torch.cat((truth[:, None],
                compose_full_frame_state(truth.expand(6, -1), translations)[None]), 1)
            control_reflection = truth_reflection[:, None].expand(1, 7)
            control_map = model.map({**prediction, 'state': control_states}, offsets,
                torch.arange(7, device='cuda')[None], control_reflection, (96, 96), atlas,
                weights, source_shape=(side, side))
            control_render = render_finite_thickness_coordinate_grid(atlas,
                control_map['coordinates'].flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
                weights.expand(7, -1))
            control_support = control_render[:, 1] > .95
            control_overlap = (control_support & control_support[0]).sum((-2, -1)).float()
            control_dice = 2 * control_overlap / (
                control_support.sum((-2, -1)) + control_support[0].sum()).clamp_min(1)
            control_gap = (control_support.float().mean((-2, -1))
                           - control_support[0].float().mean()).abs()
            control_rigid = rigid_points_090(control_states, control_reflection, chart)[0]
            control_error = (control_rigid - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
            control_eligible = (control_error >= 1.5) & (control_dice >= .85) & (control_gap <= .08)
            control = {'eligible_wrong_count': int(control_eligible.sum()),
                       'status': 'no_support_matched_wrong'}
            if bool(control_eligible.any()):
                wrong = int(control_dice.masked_fill(~control_eligible, -1).argmax())
                control.update(wrong_slot=wrong, wrong_rigid_mm=float(control_error[wrong]),
                               mapped_support_dice=float(control_dice[wrong]),
                               mapped_support_area_gap=float(control_gap[wrong]))
                readout = pair_readout(image, valid, control_map, control_render,
                    control_states, control_reflection, 0, wrong)
                control.update(readout)
                control['status'] = 'scored' if readout['common_pixels'] >= 256 else 'insufficient_common_interior'
            row['controlled'] = control

        rows.append(row)
        row_stream.write(json.dumps(row, allow_nan=False) + '\n')
        row_stream.flush()
        draw_seed += 1

summary = {'eligible_sections': count, 'draw_attempts': draw_seed - seed * 1000000,
           'strata': {}, 'scope': 'fresh synthetic TRAIN draws; diagnostic only; no model promotion'}
for name in ('blind', 'controlled'):
    statuses = [row[name]['status'] for row in rows]
    scored = [row[name] for row in rows if row[name]['status'] == 'scored']
    hp_wins = np.array([row['hp_margin'] >= .01 for row in scored])
    raw_wins = np.array([row['raw_margin'] >= .01 for row in scored])
    margins = {metric: [row[metric] for row in scored] for metric in ('hp_margin', 'raw_margin')}
    result = {'status_counts': {status: statuses.count(status) for status in sorted(set(statuses))},
              'scored': len(scored),
              'hp_win_fraction': float(hp_wins.mean()) if scored else None,
              'raw_win_fraction': float(raw_wins.mean()) if scored else None,
              'win_fraction_gain': float(hp_wins.mean() - raw_wins.mean()) if scored else None,
              'margin_distribution': {metric: {'mean': float(np.mean(values)),
                  'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)),
                  'p90': float(np.quantile(values, .9))} if values else None
                  for metric, values in margins.items()},
              'zero_intensity_max_abs_margin': max((abs(row[key]) for row in scored
                  for key in ('zero_hp_margin', 'zero_raw_margin')), default=None),
              'common_pixels_median': float(np.median([row['common_pixels'] for row in scored]))
                  if scored else None,
              'mapped_support_dice_mean': float(np.mean([row['mapped_support_dice']
                  for row in scored])) if scored else None,
              'mapped_support_area_gap_mean': float(np.mean([row['mapped_support_area_gap']
                  for row in scored])) if scored else None,
              'displacement_rms_mm_mean': [float(np.mean([row['displacement_rms_mm'][i]
                  for row in scored])) for i in range(2)] if scored else None,
              'strain_mean': [float(np.mean([row['strain_mean'][i] for row in scored]))
                  for i in range(2)] if scored else None,
              'strain_p95_median': [float(np.median([row['strain_p95'][i] for row in scored]))
                  for i in range(2)] if scored else None}
    summary['strata'][name] = result
blind = summary['strata']['blind']
summary['blind_exploratory_gate_pass'] = bool(blind['scored'] >= 32
    and blind['hp_win_fraction'] >= .75 and blind['win_fraction_gain'] >= .10
    and blind['zero_intensity_max_abs_margin'] < 1e-6)

(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'eligible_sections': count,
    'draw_attempts': summary['draw_attempts'], 'output_sha256': {name: sha(out / name)
    for name in ('config.json', 'draws.jsonl', 'rows.jsonl', 'summary.json')},
    'checkpoint_sha256': output_config['checkpoint_sha256'],
    'protocol_sha256': output_config['protocol_sha256'],
    'source_sha256': output_config['source_sha256'], 'calibrated': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'completed': True, 'summary': summary, 'output': str(out)}, allow_nan=False))
