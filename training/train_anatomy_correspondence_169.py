"""TRAIN-only anatomy correspondence stage of the standalone joint model.

The frozen 132 pose/map lineage supplies the direct predictor. This stage adds
randomly initialized anatomy descriptors and a bounded pose corrector; later
joint training must let their fitted evidence update the direct predictor.
"""

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

from training.anatomy_correspondence_feedback_168 import AnatomyCorrespondenceFeedback
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_checkpoint = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/anatomy_correspondence_169_train_001'
seed, side, bank_side = 202610111690, 256, 48
depths = torch.tensor([-500., -250., 0., 250., 500.], device='cuda')
attempts, inner_n, checkpoints = 7000, 96, (0, 1500, 3000, 4500, 6000)
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

source_names = ('train_anatomy_correspondence_169.py',
    'anatomy_correspondence_feedback_168.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py', 'global_atlas_contrast_090.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'global_plane_matcher_120.py')
source_sha = {name: sha(source / name) for name in source_names}
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256']['2000']
assert source_sha['arbitrary_plane_one_shot_model.py'] == parent_config['source_sha256'][
    'arbitrary_plane_one_shot_model.py']
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(parent['model'], strict=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
del parent
matcher = AnatomyCorrespondenceFeedback(side=bank_side, radius=3).cuda()
model.anatomy_correspondence_168 = matcher
optimizer = torch.optim.AdamW(matcher.parameters(), lr=1.5e-4, weight_decay=1e-4)

ax = (torch.arange(bank_side, device='cuda') + .5) / bank_side - .5 / side
yy, xx = torch.meshgrid(ax, ax, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
native = ((torch.arange(bank_side, device='cuda') + .5) * side / bank_side).long()
ny, nx = torch.meshgrid(native, native, indexing='ij')
native_index = (ny * side + nx).flatten()
query_x = torch.arange(bank_side, device='cuda')[None].expand(bank_side, -1)
query_y = torch.arange(bank_side, device='cuda')[:, None].expand(-1, bank_side)


def draw(index, inner=False):
    base = 60 + index % 4 if inner else (37 * index) % 60
    draw_seed = seed + (9000000 if inner else 0) + 1009 * index
    geometry = sample_panel_geometry_train_151 if index % 5 == 0 else sample_panel_geometry_train_166
    with torch.no_grad():
        original = geometry(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(context, [base], draw_seed,
            side, source_sample=original)
    return sample, {'index': index, 'base_index': base, 'draw_seed': draw_seed,
        'geometry': geometry.__name__,
        'physical_section_id': sample['provenance'][0]['physical_section_id'],
        'animal_id': sample['provenance'][0]['base_lineage']['animal_id'],
        'valid_pixels': int(sample['valid_mask'].sum()),
        'provenance': sample['provenance'][0]}


def render_bank(state, reflection, offsets, weights):
    batch = len(state)
    plane = rigid_points_090(state, reflection, chart, (side, side)).reshape(
        batch, bank_side, bank_side, 3)
    normal = full_frame_state_to_components(state)[1][..., :, 2]
    axial = offsets[:, None, :] + depths[None, :, None]
    coords = (plane[:, None, None] + axial[:, :, :, None, None, None]
              * normal[:, None, None, None, None, :])
    rendered = render_finite_thickness_coordinate_grid(atlas,
        coords.flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
        weights[:, None].expand(-1, len(depths), -1).flatten(0, 1))
    rendered = rendered.reshape(batch, len(depths), 2, bank_side, bank_side)
    support = rendered[:, :, 1:2].clamp(0, 1)
    return rendered[:, :, :1] / support.clamp_min(1e-4), support


def perturb(state, index, near):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, int(near), 168]))
    angle = rng.normal(size=3)
    angle *= rng.uniform(.06 if near else .20, .20 if near else .42) / np.linalg.norm(angle)
    translation = rng.normal(size=3)
    translation *= rng.uniform(300 if near else 1500,
                               900 if near else 2500) / np.linalg.norm(translation)
    scale = rng.normal(0, .035 if near else .07, 3)
    update = torch.as_tensor(np.r_[angle, translation, scale],
                             device=state.device, dtype=state.dtype)[None]
    return compose_full_frame_state(state, update)


def positive_labels(sample, state, reflection, support):
    centre = sample['centre'].reshape(1, -1, 3)[:, native_index].reshape(
        1, bank_side, bank_side, 3)
    valid = sample['valid_mask'][:, None].float()
    intact = (F.avg_pool2d(valid, 5, 1, 2).flatten(2)[:, :, native_index]
              .reshape(1, bank_side, bank_side) > .999)
    fragment = sample['fragment_displaced_mask'][:, None].float()
    fragment_mean = F.avg_pool2d(fragment, 5, 1, 2).flatten(2)[:, :, native_index].reshape(
        1, bank_side, bank_side)
    intact &= (fragment_mean < .001) | (fragment_mean > .999)
    origin, frame, basis = full_frame_state_to_components(state)
    delta = centre - origin[:, None, None]
    local = torch.einsum('bhwi,bij->bhwj', delta, frame[:, :, :2])
    raster = .5 + torch.linalg.solve(basis[:, None, None], local[..., None]).squeeze(-1)
    raster_x = torch.where(reflection[:, None, None].bool(),
        (side - 1) / side - raster[..., 0], raster[..., 0])
    key_x = (raster_x + .5 / side) * bank_side - .5
    key_y = (raster[..., 1] + .5 / side) * bank_side - .5
    dx = (key_x - query_x).round().long()
    dy = (key_y - query_y).round().long()
    normal = (delta * frame[:, None, None, :, 2]).sum(-1)
    depth = (normal[..., None] - depths).abs().argmin(-1)
    residual = (normal - depths[depth]).abs()
    radius = matcher.radius
    in_range = ((key_x >= 0) & (key_x < bank_side) & (key_y >= 0)
        & (key_y < bank_side) & (dx.abs() <= radius) & (dy.abs() <= radius)
        & (residual <= sample['offsets'].abs().amax() + 125))
    key_index = (key_y.round().long().clamp(0, bank_side - 1) * bank_side
                 + key_x.round().long().clamp(0, bank_side - 1))
    support_flat = support[:, :, 0].flatten(2)
    selected_support = support_flat.gather(2,
        key_index.flatten(1)[:, None].expand(-1, len(depths), -1))
    selected_support = selected_support.gather(1,
        depth.flatten(1)[:, None]).reshape(1, bank_side, bank_side)
    mask = intact & in_range & (selected_support > .5)
    label = (depth * (2 * radius + 1) ** 2 + (dy + radius) * (2 * radius + 1)
             + dx + radius).clamp(0, len(depths) * (2 * radius + 1) ** 2 - 1)
    return label, mask


def counterfactual(true_intensity, true_support, wrong_intensity, wrong_support):
    common = ((true_support > .5) & (wrong_support > .5)).float()
    interior = (F.avg_pool2d(common.flatten(0, 1), 7, 1, 3)
                .reshape_as(common) > .999).float()
    amount = interior.sum((2, 3, 4)).clamp_min(1)
    mean_t = (true_intensity * interior).sum((2, 3, 4)) / amount
    mean_w = (wrong_intensity * interior).sum((2, 3, 4)) / amount
    std_t = (((true_intensity - mean_t[:, :, None, None, None]).square() * interior
              ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    std_w = (((wrong_intensity - mean_w[:, :, None, None, None]).square() * interior
              ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    matched = ((wrong_intensity - mean_w[:, :, None, None, None])
               * std_t[:, :, None, None, None] / std_w[:, :, None, None, None]
               + mean_t[:, :, None, None, None]).clamp(0, 1)
    changed = torch.where(interior.bool(), matched, true_intensity)
    return changed, interior


def corrected(sample, source_desc, state):
    intensity, support = render_bank(state, sample['reflection'], sample['offsets'],
                                     sample['weights'])
    first = matcher.compare(source_desc, matcher.encode_atlas(intensity), support, depths)
    state = compose_full_frame_state(state, first['update'])
    fitted_intensity, fitted_support = render_bank(state, sample['reflection'],
                                                   sample['offsets'], sample['weights'])
    second = matcher.compare(source_desc, matcher.encode_atlas(fitted_intensity),
                             fitted_support, depths)
    effort = (first['update'] / matcher.update_limits).square().mean(-1)
    return {'state': state, 'score': second['score'] - .04 * effort,
            'update': first['update'], 'first': first,
            'initial_support': support,
            'intensity': fitted_intensity, 'support': fitted_support}


def one_case(sample, index):
    source_desc = matcher.encode_source(sample['inputs'])
    true = corrected(sample, source_desc, sample['state'])
    near_state = perturb(sample['state'], index, True)
    near = corrected(sample, source_desc, near_state)
    far = corrected(sample, source_desc, perturb(sample['state'], index, False))
    with torch.no_grad():
        label, mask = positive_labels(sample, sample['state'], sample['reflection'],
                                      true['initial_support'])
    logits = true['first']['correlation_logits'].flatten(1, 2)
    ce = F.cross_entropy(logits, label, reduction='none')
    ce = (ce * mask).sum() / mask.sum().clamp_min(1)
    with torch.no_grad():
        cf_shift = torch.zeros((1, 9), device='cuda')
        cf_shift[0, 5] = 1200. if index % 2 else -1200.
        cf_state = compose_full_frame_state(true['state'].detach(), cf_shift)
        cf_intensity, cf_support = render_bank(cf_state, sample['reflection'],
                                               sample['offsets'], sample['weights'])
    changed, common = counterfactual(true['intensity'], true['support'],
                                      cf_intensity, cf_support)
    true_common = matcher.compare(source_desc,
        matcher.encode_atlas(true['intensity']), common, depths)['score']
    wrong_common = matcher.compare(source_desc,
        matcher.encode_atlas(changed), common, depths)['score']
    common_sites = int(common[:, len(depths) // 2].sum().detach())
    cf_rank = F.softplus(.06 + wrong_common - true_common).mean() if common_sites >= 12 else ce.new_zeros(())
    plane_rank = F.softplus(.06 + far['score'] - true['score']).mean()
    native_valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    site = native_valid[torch.linspace(0, len(native_valid) - 1, 128,
                                       device=native_valid.device).long()]
    chart_sites = torch.stack((site.remainder(side), site.div(side, rounding_mode='floor')),
                              -1).float()[None] / side
    truth = rigid_points_090(sample['state'], sample['reflection'], chart_sites)
    before = rigid_points_090(near_state, sample['reflection'], chart_sites)
    after = rigid_points_090(near['state'], sample['reflection'], chart_sites)
    before_mm = (before - truth).norm(dim=-1).mean() / 1000
    after_mm = (after - truth).norm(dim=-1).mean() / 1000
    keep = (true['update'] / matcher.update_limits).square().mean()
    loss = .12 * ce + .5 * cf_rank + .2 * plane_rank + .5 * after_mm + .05 * keep
    return loss, {'ce': float(ce.detach()), 'ce_labels': int(mask.sum()),
        'cf_rank': float(cf_rank.detach()), 'cf_sites': common_sites,
        'plane_rank': float(plane_rank.detach()),
        'near_before_mm': float(before_mm.detach()),
        'near_after_mm': float(after_mm.detach()),
        'true_score': float(true['score'].detach()),
        'far_score': float(far['score'].detach()),
        'cf_true_score': float(true_common.detach()),
        'cf_wrong_score': float(wrong_common.detach()),
        'true_beats_far': bool((true['score'] > far['score']).item()),
        'true_beats_cf': bool((true_common > wrong_common).item()),
        'loss': float(loss.detach())}


config = {'scope': 'TRAIN-only synthetic anatomy correspondence stage, no DEV/public',
    'seed': seed, 'attempts': attempts, 'inner_n': inner_n, 'checkpoints': checkpoints,
    'train_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'geometry': '166 support-intersecting 80%, 151 full-box 20%; independent draw per attempt',
    'source_sha256': source_sha, 'parent_checkpoint_sha256': sha(parent_checkpoint),
    'parent_step': 2000, 'synthetic_provenance': context['provenance'],
    'depth_offsets_um': depths.tolist(), 'descriptor_side': bank_side,
    'posterior_calibrated': False, 'no_external_pretrained_weights': True,
    'dense_truth_train_only': True}
if os.environ.get('JOINT169_MODE') == 'preflight':
    sample, record = draw(1)
    loss, measures = one_case(sample, 1)
    loss.backward()
    with torch.no_grad():
        coverage = []
        for check_index in range(2, 18):
            check_sample, check_record = draw(check_index)
            if check_record['valid_pixels']:
                _, check = one_case(check_sample, check_index)
                coverage.append((check['ce_labels'], check['cf_sites']))
    print(json.dumps({'event': 'preflight', 'physical_section_id':
        record['physical_section_id'], 'valid_pixels': record['valid_pixels'],
        'gradient_norm': float(torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5.)),
        'further_ce_cf_sites': coverage, **measures}, allow_nan=False), flush=True)
    sys.exit(0)
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))


def inner_readout(step):
    matcher.eval()
    cases = []
    with torch.no_grad():
        for index in range(inner_n):
            sample, record = draw(index, inner=True)
            if not record['valid_pixels']:
                cases.append({**record, 'scored': False})
                continue
            _, measures = one_case(sample, index + 9000000)
            cases.append({**record, 'scored': True, **measures})
    scored = [r for r in cases if r['scored']]
    summary = {'step': step, 'attempt_n': len(cases), 'scored_n': len(scored),
        'valid_labels': sum(r['ce_labels'] for r in scored),
        'cf_pair_n': sum(r['cf_sites'] >= 12 for r in scored),
        'true_beats_cf': float(np.mean([r['true_beats_cf'] for r in scored if r['cf_sites'] >= 12])),
        'true_beats_far': float(np.mean([r['true_beats_far'] for r in scored])),
        'near_before_mm': float(np.mean([r['near_before_mm'] for r in scored])),
        'near_after_mm': float(np.mean([r['near_after_mm'] for r in scored])),
        'ce': float(np.mean([r['ce'] for r in scored])),
        'loss': float(np.mean([r['loss'] for r in scored]))}
    with (run / f'inner_step_{step:05d}.json').open('w') as stream:
        json.dump({'summary': summary, 'cases': cases}, stream, allow_nan=False)
    path = run / f'joint_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config,
                'calibrated': False}, temp)
    os.replace(temp, path)
    print(json.dumps({'event': 'inner_checkpoint', **summary}), flush=True)
    matcher.train()


started = time.perf_counter()
updates = 0
inner_readout(0)
with (run / 'training.jsonl').open('w') as stream:
    for index in range(attempts):
        sample, record = draw(index)
        if not record['valid_pixels']:
            stream.write(json.dumps({**record, 'optimizer_update': None,
                                     'skip_reason': 'blank_plane'}, allow_nan=False) + '\n')
            continue
        optimizer.zero_grad(set_to_none=True)
        loss, measures = one_case(sample, index)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5.,
                                              error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        row = {**record, 'optimizer_update': updates, 'gradient_norm': float(grad), **measures}
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if updates % 100 == 0:
            stream.flush()
            print(json.dumps({'event': 'train_milestone', 'attempt': index + 1,
                              'updates': updates, **measures}), flush=True)
        if updates in checkpoints:
            inner_readout(updates)
        if updates == checkpoints[-1]:
            break

done = {'optimizer_updates': updates, 'attempts_seen': index + 1,
    'elapsed_seconds': time.perf_counter() - started,
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints if (run / f'joint_step_{step:05d}.pt').exists()},
    'output_sha256': {name: sha(path) for path in run.iterdir() if path.is_file()
                      for name in [path.name] if name != 'completed.json'}}
(run / 'completed.json').write_text(json.dumps(done, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', **done}), flush=True)
