"""Read-only TRAIN-inner 384-vs-matched-192 atlas-correlation ceiling."""
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090


def correlation(left, right):
    left, right = left - left.mean(), right - right.mean()
    return float((left * right).sum() /
                 (left.square().sum() * right.square().sum()).sqrt().clamp_min(1e-8))


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed = 20261015384
rng = np.random.default_rng(seed)
context = load_fov_balance_v10('cuda')
model, _ = load_one_shot_checkpoint(
    root / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt')
model.requires_grad_(False)
shuffle_rng = torch.Generator(device='cuda').manual_seed(seed + 1)
axis = torch.arange(384, device='cuda') / 384
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
dense_chart = torch.stack((xx, yy), -1).reshape(1, 1, -1, 2)
rows, attempts = [], []

with torch.inference_mode():
    for base in rng.permutation(64):
        variant = int(rng.integers(VARIANTS))
        draw_seed = int(rng.integers(2**63 - 1, dtype=np.int64))
        sample = sample_fov_balance_v10(context, [int(base) * VARIANTS + variant], draw_seed, 384)
        intact = (sample['valid_mask'] & ~sample['fragment_displaced_mask'] &
                  (sample['visible'] > .95))
        pure192 = F.avg_pool2d(intact[:, None].float(), 2)[0, 0] > .999
        accepted = bool(sample['eligible'][0]) and int(pure192.sum()) >= 64
        attempts.append({'base': int(base), 'variant': variant, 'seed': draw_seed,
                         'id': sample['provenance'][0]['physical_section_id'], 'accepted': accepted})
        if not accepted:
            continue
        inputs384 = sample['inputs']
        inputs192 = F.avg_pool2d(inputs384, 2)
        source384, source192 = inputs384[0, 0], inputs192[0, 0]
        prediction = model.predict(inputs192)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = prior.topk(16, -1).indices
        state = prediction['state'].gather(1, (branch // 2)[..., None].expand(-1, -1, 12))
        sites = torch.multinomial(sample['valid_mask'][0].flatten().float(), 128,
            replacement=True, generator=torch.Generator(device='cuda').manual_seed(draw_seed))
        chart = torch.stack((sites.remainder(384), sites.div(384, rounding_mode='floor')), -1).float()[None] / 384
        truth = sample['centre'].reshape(1, -1, 3)[:, sites]
        rigid = rigid_points_090(state, branch % 2, chart[:, None], (384, 384))
        errors = ((rigid - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0].cpu().numpy()
        scores = {name: [] for name in ('384', '192', 'support', 'shuffled')}
        cells = []
        for candidate in range(16):
            current = state[:, candidate:candidate + 1]
            reflection = branch[:, candidate:candidate + 1] % 2
            plane = rigid_points_090(current, reflection, dense_chart, (384, 384)).reshape(1, 384, 384, 3)
            frame = full_frame_state_to_components(current[:, 0])[1]
            world = plane[:, None] + sample['offsets'][:, :, None, None, None] * frame[:, None, None, None, :, 2]
            rendered = render_finite_thickness_coordinate_grid(
                context['atlas'], world, (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            support384 = rendered[1].clamp(0, 1)
            atlas384 = rendered[0] / support384.clamp_min(1e-4)
            support192 = F.avg_pool2d(support384[None, None], 2)[0, 0]
            atlas192 = F.avg_pool2d(atlas384[None, None], 2)[0, 0]
            full_support192 = F.avg_pool2d((support384 > .95).float()[None, None], 2)[0, 0] > .999
            mask192 = pure192 & full_support192
            count = int(mask192.sum())
            cells.append(count)
            if count < 64:
                for arm in scores:
                    scores[arm].append(float('-inf'))
                continue
            mask384 = mask192.repeat_interleave(2, 0).repeat_interleave(2, 1)
            image384, atlas = source384[mask384], atlas384[mask384]
            scores['384'].append(abs(correlation(image384, atlas)))
            scores['192'].append(abs(correlation(source192[mask192], atlas192[mask192])))
            scores['support'].append(float(support192[pure192].mean()))
            scores['shuffled'].append(abs(correlation(image384,
                atlas[torch.randperm(len(atlas), device='cuda', generator=shuffle_rng)])))
        selected = {'prior': 0, **{arm: int(np.argmax(values)) if max(values) > -float('inf')
                    else 0 for arm, values in scores.items()}}
        rows.append({'id': attempts[-1]['id'], 'base': int(base), 'variant': variant,
                     'seed': draw_seed, 'cells': cells, 'errors_mm': errors.tolist(),
                     'scores': {arm: [value if np.isfinite(value) else None for value in values]
                                for arm, values in scores.items()}, 'selected': selected})
        print(json.dumps({'case': len(rows), 'attempts': len(attempts),
                          'selected': selected, 'prior_mm': float(errors[0]),
                          'best_mm': float(errors.min())}), flush=True)
        if len(rows) == 32:
            break

means = {arm: float(np.mean([r['errors_mm'][r['selected'][arm]] for r in rows]))
         for arm in ('384', '192', 'support', 'shuffled', 'prior')}
tails = {arm: sum(r['errors_mm'][r['selected'][arm]] > 5 for r in rows)
         for arm in means}
gains = {arm: means[arm] - means['384'] for arm in means if arm != '384'}
screen = (len(rows) == 32 and all(gain >= .25 for gain in gains.values()) and
          all(tails['384'] <= tails[arm] for arm in gains))
report = {'scope': 'fresh TRAIN-inner v10/FOV-balanced draws; read-only frozen 309 checkpoint',
          'draw_seed': seed, 'resolution': {'observed': 384, 'control': 'exact 2x2 area downsample of same 384 draw'},
          'proposals': 'frozen 309 prior top 16 from 192 downsample; same candidate bank and 384 truth sites for every arm',
          'score': 'absolute Pearson atlas intensity on true intact tissue and full atlas support; 384 bank pooled for matched 192 control',
          'screen': '384 mean top1 at least 0.25 mm below every control, with no >5 mm tail increase against any control',
          'attempts': attempts, 'rows': rows, 'accepted': len(rows), 'oracle_mean_mm': float(np.mean([
              min(r['errors_mm']) for r in rows])), 'means_mm': means, 'tails_gt5mm': tails,
          'gain_384_vs_control_mm': gains, 'promotion_screen_pass': screen}
output = root / 'runs/resolution_ceiling_384_001'
output.mkdir(parents=True, exist_ok=False)
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'accepted': len(rows), 'attempts': len(attempts), 'means_mm': means,
                  'tails_gt5mm': tails, 'gain_384_vs_control_mm': gains,
                  'promotion_screen_pass': screen, 'report': str(output / 'report.json')}), flush=True)
