"""TRAIN-only test of atlas-image fit direction from blind 132 pose proposals."""

import hashlib
import json
import math
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
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_joint_uncertainty import full_frame_residual
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(len(beam), device=beam.device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(len(beam), 64, 2)[
            torch.arange(len(beam), device=beam.device), anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


def render(states, reflection, offsets, weights):
    count = len(states)
    points = rigid_points_090(states, reflection, chart, (256, 256)).reshape(count, 48, 48, 3)
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    world = points[:, None] + offsets.expand(count, -1)[:, :, None, None, None] * normal[:, None, None, None, :]
    sampled = render_finite_thickness_coordinate_grid(atlas, world, (0., 0., 0.),
        (25., 25., 25.), weights.expand(count, -1))
    support = sampled[:, 1:2].clamp(0, 1)
    return sampled[:, :1] / support.clamp_min(1e-4), support


def score(image, atlas_image, mask):
    x = F.avg_pool2d(image, 3, 1, 1)[0, 0]
    y = F.avg_pool2d(atlas_image, 3, 1, 1)[:, 0]
    count = mask.sum()
    a = (x * mask).sum() / count
    b = (y * mask).sum((-2, -1)) / count
    u = (x - a) * mask
    v = (y - b[:, None, None]) * mask
    r = (u[None] * v).sum((-2, -1)) / (
        (u.square().sum() * v.square().sum((-2, -1))).sqrt().clamp_min(1e-8))
    return (r.square() + 1e-4).sqrt()


def mismatched_intensity(image, pixels, shifted_pixels):
    flat = image.flatten(1)
    other = flat.clone()
    other[:, pixels] = flat[:, shifted_pixels]
    return other.reshape_as(image)


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/LOCAL_ATLAS_FIT_DIRECTION_175_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
out = Path(os.environ.get('AT175_RUN', str(root / 'runs/local_atlas_fit_direction_175_train_001'))).resolve()
seed, accepted_target = 202610175, int(os.environ.get('AT175_TARGET', '128'))
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert torch.cuda.is_available() and sha(parent_path) == json.loads(
    (parent_dir / 'completed.json').read_text())['checkpoint_sha256']['2000']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
model.load_state_dict(parent['model'], strict=True)
del parent
axis = (torch.arange(48, device='cuda') + .5) / 48 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
sites = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                      [127.5, 127.5]], device='cuda') / 256
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({'seed': seed, 'accepted_sections': accepted_target,
    'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'parent_checkpoint_sha256': sha(parent_path),
    'synthetic_provenance': context['provenance'],
    'diagnostic_truth_not_an_inference_input': True,
    'external_pretrained_weights_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2, allow_nan=False))

rows, attempted = [], 0
with (out / 'rows.jsonl').open('w') as stream, (out / 'draws.jsonl').open('w') as draws:
    while len(rows) < accepted_target:
        attempted += 1
        base = int(rng.integers(64))
        draw_seed = int(rng.integers(0, 2**63 - 1))
        geometry = sample_panel_geometry_train_151 if rng.random() < .2 else sample_panel_geometry_train_166
        version = 'v4' if len(rows) % 2 == 0 else 'v3'
        sampler = sample_one_shot_slide_artifacts_v4 if version == 'v4' else sample_one_shot_slide_artifacts_v3
        with torch.no_grad():
            original = geometry(context, [base], draw_seed, 256)
            sample = sampler(context, [base], draw_seed, 256, source_sample=original)
        draws.write(json.dumps({'attempt': attempted, 'used': bool(sample['eligible'][0]),
            'base_index': base, 'draw_seed': draw_seed, 'appearance': version,
            'geometry': geometry.__name__, 'provenance': sample['provenance'][0]},
            allow_nan=False) + '\n')
        if not bool(sample['eligible'][0]):
            continue
        with torch.no_grad():
            prediction = model.predict(sample['inputs'])
            prior, beam = beam_for(prediction)
            ids = beam[0]
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            truth = sample['state'][0]
            truth_flag = sample['reflection'][0]
            target_points = rigid_points_090(truth[None], truth_flag[None], sites)[0]
            proposal_points = rigid_points_090(states, flags, sites)
            physical = (proposal_points - target_points[None]).norm(dim=-1).mean(-1) / 1000
            same = flags == truth_flag
            top = int(prior[0, ids].argmax())
            near_choices = ((physical >= .5) & (physical <= 3.) & same).nonzero()[:, 0]
            near = int(near_choices[physical[near_choices].argmin()]) if len(near_choices) else None
            image = F.interpolate(sample['inputs'][:, :1], (48, 48), mode='area')
            normal = full_frame_state_to_components(truth[None])[1][0, :, 2]
            angle = math.degrees(math.acos(float(normal.abs().max().clamp(-1, 1))))
            row = {'section_index': len(rows), 'attempt': attempted, 'base_index': base,
                'draw_seed': draw_seed, 'appearance': version, 'geometry': geometry.__name__,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'base_lineage': sample['provenance'][0]['base_lineage'],
                'nearest_cardinal_angle_deg': angle, 'branches': {}}
        for name, slot in (('prior_top', top), ('near_same_reflection', near)):
            if slot is None or not bool(same[slot]) or float(physical[slot]) < .5:
                row['branches'][name] = {'eligible': False,
                    'reason': 'no_near_candidate' if slot is None else 'reflection_or_distance'}
                continue
            state = states[slot].detach()
            delta = full_frame_residual(state[None], truth[None])[0].detach()
            reflected = flags[slot].expand(3)
            with torch.no_grad():
                trial_states = compose_full_frame_state(state.expand(3, -1),
                    state.new_tensor([.2, 0., -.2])[:, None] * delta[None])
                rendered, support = render(trial_states, reflected, sample['offsets'], sample['weights'])
                common = (support[1, 0] > .5).float()
                common = (F.avg_pool2d(common[None, None], 7, 1, 3,
                    count_include_pad=True)[0, 0] >= 1 - 1e-6).float()
                count = int(common.sum())
                if count < 100:
                    row['branches'][name] = {'eligible': False, 'reason': 'low_shared_support',
                        'shared_sites': count}
                    continue
                pixels = common.flatten().nonzero()[:, 0]
                shifted_pixels = pixels.roll(max(1, count // 3))
                full = score(image, rendered, common)
                null = score(image, mismatched_intensity(rendered, pixels, shifted_pixels), common)
                moved = rigid_points_090(trial_states, reflected, sites)
                physical_moves = (moved - target_points[None]).norm(dim=-1).mean(-1) / 1000
                support_retention = [float(((support[i, 0] > .5).float()
                    * common).sum() / common.sum()) for i in range(3)]
            t = torch.zeros((), device='cuda', requires_grad=True)
            current_state = compose_full_frame_state(state[None], (t * delta)[None])
            current, _ = render(current_state, flags[slot:slot + 1],
                sample['offsets'], sample['weights'])
            full_t = score(image, current, common)[0]
            null_t = score(image, mismatched_intensity(current, pixels, shifted_pixels), common)[0]
            full_derivative = torch.autograd.grad(full_t, t, retain_graph=True)[0]
            null_derivative = torch.autograd.grad(null_t, t)[0]
            row['branches'][name] = {'eligible': True, 'slot': slot,
                'branch_id': int(ids[slot]), 'shared_sites': count,
                'physical_mm_toward_center_away': [float(x) for x in physical_moves],
                'physical_move_monotonic': bool(physical_moves[0] < physical_moves[1]
                    and physical_moves[1] < physical_moves[2]),
                'support_retention_toward_center_away': support_retention,
                'fit_score_toward_center_away': [float(x) for x in full],
                'null_score_toward_center_away': [float(x) for x in null],
                'fit_direction_positive': bool(full[0] > full[2]),
                'null_direction_positive': bool(null[0] > null[2]),
                'fit_derivative': float(full_derivative),
                'null_derivative': float(null_derivative)}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'event': 'accepted', 'count': len(rows), 'attempts': attempted}), flush=True)

assert len({r['physical_section_id'] for r in rows}) == accepted_target
summary = {'sections': len(rows), 'attempts': attempted, 'groups': {},
    'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False, 'calibrated': False}
for name in ('prior_top', 'near_same_reflection'):
    for version in ('v3', 'v4'):
        for angle_name, lo, hi in (('all', 0, 55), ('0_30', 0, 30), ('30_45', 30, 45), ('45_55', 45, 55)):
            subset = [r['branches'][name] for r in rows if r['appearance'] == version
                and lo <= r['nearest_cardinal_angle_deg'] < hi and r['branches'][name]['eligible']]
            monotonic = [x for x in subset if x['physical_move_monotonic']]
            summary['groups'][f'{name}/{version}/{angle_name}'] = {'eligible': len(subset),
                'monotonic': len(monotonic),
                'fit_forward': sum(x['fit_direction_positive'] for x in subset),
                'null_forward': sum(x['null_direction_positive'] for x in subset),
                'fit_gradient_forward': sum(x['fit_derivative'] > 0 for x in subset),
                'null_gradient_forward': sum(x['null_derivative'] > 0 for x in subset),
                'fit_forward_monotonic': sum(x['fit_direction_positive'] for x in monotonic),
                'null_forward_monotonic': sum(x['null_direction_positive'] for x in monotonic),
                'fit_gradient_forward_monotonic': sum(x['fit_derivative'] > 0 for x in monotonic),
                'mean_physical_move_mm': float(np.mean([
                    x['physical_mm_toward_center_away'][2] - x['physical_mm_toward_center_away'][0]
                    for x in subset])) if subset else None}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'output_sha256': {
    p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'completed.json'},
    'sections': len(rows), 'attempts': attempted, 'calibrated': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'sections': len(rows), 'attempts': attempted,
    'near_v3': summary['groups']['near_same_reflection/v3/all'],
    'near_v4': summary['groups']['near_same_reflection/v4/all']}), flush=True)
