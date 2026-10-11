"""Frozen 132 blind-beam coupled pose-volume assay; TRAIN only, no weight updates."""

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
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
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


def render(states, reflection, offsets, weights, side):
    axis = (torch.arange(side, device=states.device, dtype=states.dtype) + .5) / side - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    pairs = []
    for start in range(0, len(states), 8):
        state = states[start:start + 8]
        flag = reflection[start:start + 8]
        plane = rigid_points_090(state, flag, chart).reshape(len(state), side, side, 3)
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        world = (plane[:, None] + offsets[0, :, None, None, None]
                 * normal[:, None, None, None, :])
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world, (0., 0., 0.), (25., 25., 25.), weights.expand(len(state), -1))
        support = rendered[:, 1:2].clamp(0, 1)
        pairs.append(torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1))
    return torch.cat(pairs)


def permute_inside(pairs, seed):
    values = pairs[:, 0].flatten(1)
    inside = (pairs[:, 1] > .5).flatten(1)
    generator = torch.Generator(device=values.device).manual_seed(seed)
    keys = torch.rand(values.shape, device=values.device, generator=generator)
    order = keys.masked_fill(~inside, 2).argsort(-1)
    ranks = (inside.long().cumsum(-1) - 1).clamp_min(0)
    shuffled = values.gather(1, order).gather(1, ranks)
    return torch.where(inside, shuffled, values).reshape_as(pairs[:, 0])


def ncc(source, images, masks):
    n = masks.sum((-2, -1)).clamp_min(1)
    x = source[None].expand_as(images)
    mx = (x * masks).sum((-2, -1)) / n
    my = (images * masks).sum((-2, -1)) / n
    a = (x - mx[:, None, None]) * masks
    b = (images - my[:, None, None]) * masks
    return (a * b).sum((-2, -1)).abs() / (
        a.square().sum((-2, -1)) * b.square().sum((-2, -1))).sqrt().clamp_min(1e-8)


def choose_branch(winners, images, pairs, valid, source, prior, direct):
    left, right = torch.triu_indices(16, 16, offset=1, device=prior.device)
    chosen_images = images[winners]
    support = pairs[winners, 1] > .5
    a, b = support[left], support[right]
    common = F.avg_pool2d((a & b)[:, None].float(), 5, 1, 2)[:, 0] >= 1 - 1e-6
    intersection = (a & b).sum((-2, -1))
    union = (a | b).sum((-2, -1)).clamp_min(1)
    eligible = ((common.sum((-2, -1)) >= 64) & (intersection / union >= .45)
                & ((a.sum((-2, -1)) - b.sum((-2, -1))).abs() / (96 * 96) <= .15)
                & valid[left] & valid[right])
    margin = ncc(source, chosen_images[left], common) - ncc(source, chosen_images[right], common)
    margin = margin * eligible
    total = prior.new_zeros(16)
    count = prior.new_zeros(16)
    total.index_add_(0, left, margin)
    total.index_add_(0, right, -margin)
    count.index_add_(0, left, eligible.float())
    count.index_add_(0, right, eligible.float())
    rank = total / count.clamp_min(1) + .01 * (prior - prior.max())
    rank = rank.masked_fill(count < 2, -torch.inf)
    choice = int(rank.argmax()) if bool(torch.isfinite(rank).any()) else direct
    return choice, count, rank


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/COUPLED_POSE_VOLUME_181_TRAIN_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/coupled_pose_volume_181_train_001'
assert source.drive.upper() == protocol.drive.upper() == out.drive.upper() == 'I:'
assert torch.cuda.is_available() and not out.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed = 20261011181
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
    checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 2000 and checkpoint['calibrated'] is False
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
grid = torch.cartesian_prod(*([torch.tensor([-1., 0., 1.], device='cuda')] * 3))
update = torch.zeros((27, 9), device='cuda')
update[:, :2] = .16 * grid[:, :2]
update[:, 5] = 1000 * grid[:, 2]
assert bool((update[13] == 0).all())
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / 256
source_files = ('assay_coupled_pose_volume_181.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'global_atlas_contrast_090.py',
    'global_plane_matcher_120.py', 'train_spatial_blind_verifier_178.py',
    'assay_acquired_atlas_information_180.py')
config = {'seed': seed, 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source.parent / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'sections': 'one independently drawn eligible TRAIN section per base and v3/v4 appearance; all attempts logged',
    'blind_beam': 'frozen 132 8-base/6-anchor/2-normal-diverse',
    'volume': '16 x 27 joint +/-0.16-rad u/v tilt and +/-1000-um normal offset; central state unchanged',
    'render': '48px all actions, 96px union of each arm original-plus-top-two shortlist; each arm selects only its own shortlist; 9-sample true PSF',
    'score': '3x3-smoothed absolute Pearson on eroded fixed original-branch support minus 0.5 support loss',
    'cross_branch': 'pairwise common 5x5-eroded support, 180 eligibility, mean signed margin + 0.01 centered frozen prior',
    'null': 'independent within-own-support atlas-intensity permutation at both resolutions; same states, priors and shortlist union',
    'evaluation_sites': '128 evenly raster-spaced observed-valid tissue pixels plus five frame points; truth withheld until selection',
    'calibrated': False, 'development_used': False, 'final_animals_used': False,
    'public_benchmark_used': False, 'real_images_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')

rows, physical_ids = [], set()
with (out / 'draws.jsonl').open('w') as draws, (out / 'rows.jsonl').open('w') as accepted:
    for appearance in ('v3', 'v4'):
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        for base in range(64):
            while True:
                draw_seed = int(rng.integers(0, 2**63 - 1000))
                with torch.inference_mode():
                    geometry = sample_panel_geometry_train_166(context, [base], draw_seed, 256)
                    sample = sampler(context, [base], draw_seed, 256, source_sample=geometry)
                provenance = sample['provenance'][0]
                physical_id = provenance['physical_section_id']
                assert provenance['split'] == 'train' and physical_id not in physical_ids
                physical_ids.add(physical_id)
                draw = {'appearance': appearance, 'base_index': base, 'draw_seed': draw_seed,
                    'physical_section_id': physical_id, 'eligible': bool(sample['eligible'][0]),
                    'valid_pixels': int(sample['valid_mask'][0].sum()),
                    'model_inputs_sha256': hashlib.sha256(sample['inputs'].detach().cpu()
                        .contiguous().numpy().tobytes()).hexdigest(), 'provenance': provenance}
                draws.write(json.dumps(draw, allow_nan=False) + '\n')
                if draw['eligible']:
                    break

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model.predict(sample['inputs'])
                prior_all, beam = beam_for(prediction)
                ids = beam[0]
                original = prediction['state'][0, ids // 2]
                flags = ids.remainder(2)
                prior = prior_all[0, ids]
                direct = int(prior.argmax())
                states = compose_full_frame_state(original.repeat_interleave(27, 0),
                    update.repeat(16, 1))
                reflections = flags.repeat_interleave(27)
                coarse_pairs = render(states, reflections, sample['offsets'], sample['weights'], 48)
                coarse_null = permute_inside(coarse_pairs, draw_seed + 48)
                original_support48 = coarse_pairs.reshape(16, 27, 2, 48, 48)[:, 13, 1] > .5
                original_mask48 = (F.avg_pool2d(original_support48[:, None].float(), 3, 1, 1)
                                   [:, 0] >= 1 - 1e-6)
                mask48 = original_mask48.repeat_interleave(27, 0).float()
                retention48 = (coarse_pairs[:, 1] * mask48).sum((-2, -1)) / mask48.sum((-2, -1)).clamp_min(1)
                valid48 = ((mask48.sum((-2, -1)) >= 64) & (retention48 >= .9))
                observed48 = F.avg_pool2d(F.interpolate(sample['inputs'][:, :1],
                    (48, 48), mode='area'), 3, 1, 1)[0, 0]
                atlas48 = F.avg_pool2d(coarse_pairs[:, :1], 3, 1, 1)[:, 0]
                null48 = F.avg_pool2d(coarse_null[:, None], 3, 1, 1)[:, 0]
                full48 = (ncc(observed48, atlas48, mask48) - .5 * (1 - retention48)
                          ).masked_fill(~valid48, -torch.inf).reshape(16, 27)
                null48_score = (ncc(observed48, null48, mask48) - .5 * (1 - retention48)
                                ).masked_fill(~valid48, -torch.inf).reshape(16, 27)
                full_shortlist = {27 * branch + position for branch in range(16)
                    for position in (13, *full48[branch].topk(2).indices.tolist())}
                null_shortlist = {27 * branch + position for branch in range(16)
                    for position in (13, *null48_score[branch].topk(2).indices.tolist())}
                fine_indices = sorted(full_shortlist | null_shortlist)
                fine_index = torch.tensor(fine_indices, device='cuda')
                fine_branch = fine_index // 27
                full_allowed = torch.tensor([index in full_shortlist for index in fine_indices],
                    device='cuda')
                null_allowed = torch.tensor([index in null_shortlist for index in fine_indices],
                    device='cuda')
                fine_pairs = render(states[fine_index], reflections[fine_index],
                    sample['offsets'], sample['weights'], 96)
                fine_null = permute_inside(fine_pairs, draw_seed + 96)
                original_positions = torch.tensor([fine_indices.index(27 * branch + 13)
                    for branch in range(16)], device='cuda')
                original_support96 = fine_pairs[original_positions, 1] > .5
                original_mask96 = (F.avg_pool2d(original_support96[:, None].float(), 5, 1, 2)
                                   [:, 0] >= 1 - 1e-6)
                mask96 = original_mask96[fine_branch].float()
                retention96 = (fine_pairs[:, 1] * mask96).sum((-2, -1)) / mask96.sum((-2, -1)).clamp_min(1)
                valid96 = ((mask96.sum((-2, -1)) >= 64) & (retention96 >= .9))
                observed96 = F.avg_pool2d(F.interpolate(sample['inputs'][:, :1],
                    (96, 96), mode='area'), 3, 1, 1)[0, 0]
                atlas96 = F.avg_pool2d(fine_pairs[:, :1], 3, 1, 1)[:, 0]
                null96 = F.avg_pool2d(fine_null[:, None], 3, 1, 1)[:, 0]
                full96 = (ncc(observed96, atlas96, mask96) - .5 * (1 - retention96)
                          ).masked_fill(~(valid96 & full_allowed), -torch.inf)
                null96_score = (ncc(observed96, null96, mask96) - .5 * (1 - retention96)
                                ).masked_fill(~(valid96 & null_allowed), -torch.inf)
                full_winners, null_winners = [], []
                for branch in range(16):
                    positions = (fine_branch == branch).nonzero()[:, 0]
                    centre = int(original_positions[branch])
                    best_full = positions[full96[positions].argmax()]
                    best_null = positions[null96_score[positions].argmax()]
                    full_winners.append(int(best_full) if bool(torch.isfinite(full96[best_full])) else centre)
                    null_winners.append(int(best_null) if bool(torch.isfinite(null96_score[best_null])) else centre)
                full_winners = torch.tensor(full_winners, device='cuda')
                null_winners = torch.tensor(null_winners, device='cuda')
                full_valid = torch.isfinite(full96[full_winners])
                null_valid = torch.isfinite(null96_score[null_winners])
                full_branch, full_pairs, full_rank = choose_branch(full_winners, atlas96,
                    fine_pairs, full_valid, observed96, prior, direct)
                null_branch, null_pairs, null_rank = choose_branch(null_winners, null96,
                    fine_pairs, null_valid, observed96, prior, direct)
                full_position = int(full_winners[full_branch])
                null_position = int(null_winners[null_branch])
                full_index = fine_indices[full_position]
                null_index = fine_indices[null_position]
            torch.cuda.synchronize()
            paired_seconds = time.perf_counter() - start

            with torch.inference_mode():
                pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
                pixels = pixels[torch.linspace(0, len(pixels) - 1, 128,
                    device='cuda').long()]
                chart = torch.stack((pixels.remainder(256),
                    pixels.div(256, rounding_mode='floor')), -1).float() / 256
                truth = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
                tissue_mm = (rigid_points_090(states, reflections, chart) - truth[None]
                             ).norm(dim=-1).mean(-1) / 1000
                true_five = rigid_points_090(sample['state'], sample['reflection'], five)[0]
                five_mm = (rigid_points_090(states, reflections, five) - true_five[None]
                           ).norm(dim=-1).mean(-1) / 1000
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                angle = math.degrees(math.acos(float(normal.abs().max().clamp(-1, 1))))
                direct_index = 27 * direct + 13
                original_error = tissue_mm[13::27]
                row = {**draw, 'nearest_cardinal_angle_deg': angle,
                    'beam_branch_ids': ids.tolist(), 'beam_prior_logits': prior.tolist(),
                    'beam_state12': original.tolist(), 'beam_reflection': flags.tolist(),
                    'original_tissue_mm16': original_error.tolist(),
                    'coarse_tissue_mm432': tissue_mm.tolist(),
                    'coarse_full_score432': [float(x) if math.isfinite(float(x)) else None
                        for x in full48.flatten()],
                    'coarse_null_score432': [float(x) if math.isfinite(float(x)) else None
                        for x in null48_score.flatten()],
                    'fine_coarse_indices': fine_indices,
                    'full_fine_coarse_indices': sorted(full_shortlist),
                    'null_fine_coarse_indices': sorted(null_shortlist),
                    'fine_tissue_mm': tissue_mm[fine_index].tolist(),
                    'fine_support_retention': retention96.tolist(),
                    'coarse_truth_best_mm': float(tissue_mm.min()),
                    'original_truth_best_mm': float(original_error.min()),
                    'fine_truth_best_mm': float(tissue_mm[fine_index].min()),
                    'full_fine_truth_best_mm': float(tissue_mm[sorted(full_shortlist)].min()),
                    'direct_tissue_mm': float(tissue_mm[direct_index]),
                    'full_tissue_mm': float(tissue_mm[full_index]),
                    'null_tissue_mm': float(tissue_mm[null_index]),
                    'direct_five_mm': float(five_mm[direct_index]),
                    'full_five_mm': float(five_mm[full_index]),
                    'null_five_mm': float(five_mm[null_index]),
                    'direct_branch_id': int(ids[direct]),
                    'full_branch_id': int(ids[full_branch]),
                    'null_branch_id': int(ids[null_branch]),
                    'full_variant': full_index % 27, 'null_variant': null_index % 27,
                    'full_selected_state12': states[full_index].tolist(),
                    'null_selected_state12': states[null_index].tolist(),
                    'full_selected_retention': float(retention96[full_position]),
                    'null_selected_retention': float(retention96[null_position]),
                    'full_pair_count': int(full_pairs[full_branch]),
                    'null_pair_count': int(null_pairs[null_branch]),
                    'full_supported_selection': bool(full_pairs[full_branch] >= 2),
                    'null_supported_selection': bool(null_pairs[null_branch] >= 2),
                    'full_rank16': [float(x) if math.isfinite(float(x)) else None for x in full_rank],
                    'null_rank16': [float(x) if math.isfinite(float(x)) else None for x in null_rank],
                    'fine_full_score': [float(x) if math.isfinite(float(x)) else None for x in full96],
                    'fine_null_score': [float(x) if math.isfinite(float(x)) else None for x in null96_score],
                    'paired_inference_seconds': paired_seconds}
                rows.append(row)
                accepted.write(json.dumps(row, allow_nan=False) + '\n')
        draws.flush()
        accepted.flush()
        print(json.dumps({'event': 'appearance_done', 'appearance': appearance,
            'eligible': sum(row['appearance'] == appearance for row in rows)}), flush=True)


def summary(part):
    if not part:
        return {'sections': 0}
    return {'sections': len(part),
        'mean_tissue_mm': {name: float(np.mean([row[f'{name}_tissue_mm'] for row in part]))
            for name in ('direct', 'full', 'null')},
        'capture_le_1p5': {name: float(np.mean([row[f'{name}_tissue_mm'] <= 1.5 for row in part]))
            for name in ('direct', 'full', 'null')},
        'truth_best_capture_le_0p5': {name: float(np.mean([row[f'{name}_truth_best_mm'] <= .5
            for row in part])) for name in ('original', 'coarse', 'fine', 'full_fine')},
        'truth_best_capture_le_1p5': {name: float(np.mean([row[f'{name}_truth_best_mm'] <= 1.5
            for row in part])) for name in ('original', 'coarse', 'fine', 'full_fine')},
        'mean_five_mm': {name: float(np.mean([row[f'{name}_five_mm'] for row in part]))
            for name in ('direct', 'full', 'null')},
        'mean_selected_support_retention': float(np.mean([row['full_selected_retention'] for row in part])),
        'supported_selection_fraction': float(np.mean([row['full_supported_selection'] for row in part])),
        'paired_inference_p95_seconds': float(np.quantile([row['paired_inference_seconds']
            for row in part], .95))}


result = {'by_appearance': {}, 'gate': {}}
for appearance in ('v3', 'v4'):
    part = [row for row in rows if row['appearance'] == appearance]
    population = summary(part)
    result['by_appearance'][appearance] = {'population': population,
        'sparse_tissue_le_10pct': summary([row for row in part if row['valid_pixels'] <= 6553]),
        'oblique_ge_30deg': summary([row for row in part
            if row['nearest_cardinal_angle_deg'] >= 30])}
    result['gate'][appearance] = {
        'eligible_64': len(part) == 64,
        'fine_capture_gain_0p10': population['truth_best_capture_le_0p5']['full_fine']
            >= population['truth_best_capture_le_0p5']['original'] + .10,
        'selected_mean_gain_0p25': population['mean_tissue_mm']['full']
            <= population['mean_tissue_mm']['direct'] - .25,
        'selected_capture_gain_0p10': population['capture_le_1p5']['full']
            >= population['capture_le_1p5']['direct'] + .10,
        'full_over_null_mean_0p10': population['mean_tissue_mm']['full']
            <= population['mean_tissue_mm']['null'] - .10,
        'five_point_nonregression': population['mean_five_mm']['full']
            <= population['mean_five_mm']['direct'],
        'supported_selection_ge_0p90': population['supported_selection_fraction'] >= .90,
        'support_retention_ge_0p99': population['mean_selected_support_retention'] >= .99,
        'paired_p95_le_18_seconds': population['paired_inference_p95_seconds'] <= 18.}
result['continuation_gate_passes'] = all(all(checks.values()) for checks in result['gate'].values())
(out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'draw_attempts': len(physical_ids), 'unique_physical_section_ids': len(physical_ids),
    'eligible_sections': len(rows), 'source_sha256': config['source_sha256'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'protocol_sha256': config['protocol_sha256'],
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'draws.jsonl', 'rows.jsonl', 'summary.json')},
    'continuation_gate_passes': result['continuation_gate_passes'],
    'calibrated': False, 'development_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'completed', 'gate_passes': result['continuation_gate_passes']}), flush=True)
