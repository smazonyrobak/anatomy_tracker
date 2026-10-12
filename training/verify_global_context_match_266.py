"""Independent post-exit audit of the frozen 266 TRAIN-inner pilot."""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/global_context_match_266_train_inner_001'
SOURCE = Path(__file__).resolve().parent
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_context_match_266 import GlobalContextMatcher, atlas_input_and_masks


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(name):
    return [json.loads(line) for line in (RUN / name).read_text(encoding='utf8').splitlines()]


completed = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
assert completed['batches'] == 1500 and completed['presentations'] == 3000
assert config['seed'] == 2026101266 and config['batches'] == 1500 and config['batch_size'] == 2
assert config['train_bases'] == list(range(56)) and config['inner_bases'] == list(range(56, 64))
assert config['inner_sections_per_base'] == 8 and config['atlas_grid_um'] == 200
assert not any(config[k] for k in ('prior_weights_or_features', 'calibrated',
                                  'biological_validation', 'public_benchmark_used'))
assert not any(completed[k] for k in ('calibrated', 'biological_validation', 'public_benchmark_used'))
for name in ('config', 'inner_records', 'draws', 'training', 'inner_readout'):
    assert sha(RUN / f'{name}.json' if name == 'config' else RUN / f'{name}.jsonl') == completed[f'{name}_sha256'], name
for step in (500, 1500):
    assert sha(RUN / f'match_step_{step:05d}.pt') == completed['checkpoints_sha256'][str(step)]
for name, expected in config['source_sha256'].items():
    assert sha(SOURCE / name) == expected, name
assert sha(SOURCE.parent / 'docs/publication/GLOBAL_CONTEXT_MATCH_266_PROTOCOL_20261012.md') == config['protocol_sha256']

manifest = rows('inner_records.jsonl')
draws = rows('draws.jsonl')
training = rows('training.jsonl')
saved = rows('inner_readout.jsonl')
assert len(manifest) == 64 and Counter(r['base'] for r in manifest) == Counter({b: 8 for b in range(56, 64)})
assert all(r['subject'] // VARIANTS == r['base'] for r in manifest)
assert all(0 <= r['subject'] // VARIANTS < 56 and 1 <= r['batch'] <= 1500 for r in draws)
accepted = [r for r in draws if r['used']]
assert len(accepted) == 3000 and Counter(r['batch'] for r in accepted) == Counter({b: 2 for b in range(1, 1501)})
train_ids = [r['physical_section_id'] for r in accepted]
inner_ids = [r['physical_section_id'] for r in manifest]
assert len(set(train_ids)) == 3000 and len(set(inner_ids)) == 64 and not set(train_ids) & set(inner_ids)
assert [r['batch'] for r in training] == [1, *range(100, 1501, 100)]
assert all(r['presentations'] == 2 * r['batch'] for r in training)
assert len(saved) == 3 * 2 * 64

panel = []
for record in manifest:
    path = RUN / record['file']
    assert sha(path) == record['sha256'], record['file']
    with np.load(path) as arrays:
        image = torch.from_numpy(arrays['image'].copy())
        coordinates = torch.from_numpy(arrays['coordinates'].copy())
        valid = torch.from_numpy(arrays['valid'].copy())
    assert image.shape == (5, 192, 192) and coordinates.shape == (256, 3) and valid.shape == (256,)
    panel.append({'image': image, 'coordinates': coordinates, 'valid': valid.bool(), **record})

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.dumps(context['provenance'], sort_keys=True) == json.dumps(config['synthetic_provenance'], sort_keys=True)
atlas, fine_mask, coarse_mask = atlas_input_and_masks(context['atlas'])
assert int(fine_mask.sum()) == config['atlas_codebook_valid_fine']
assert int(coarse_mask.sum()) == config['atlas_codebook_valid_coarse']
support_atlas = atlas.clone()
support_atlas[0] = 0
d, h, w = atlas.shape[-3:]
dc, hc, wc = (d + 1) // 2, (h + 1) // 2, (w + 1) // 2
fine_indices = fine_mask.nonzero()[:, 0]
coarse_indices = coarse_mask.nonzero()[:, 0]
fine_lookup = torch.full((d * h * w,), -1, device='cuda', dtype=torch.long)
coarse_lookup = torch.full((dc * hc * wc,), -1, device='cuda', dtype=torch.long)
fine_lookup[fine_indices] = torch.arange(len(fine_indices), device='cuda')
coarse_lookup[coarse_indices] = torch.arange(len(coarse_indices), device='cuda')
ty, tx = torch.meshgrid(torch.arange(16, device='cuda'), torch.arange(16, device='cuda'), indexing='ij')
position = torch.stack((tx.flatten(), ty.flatten()), -1).float()


def score(model, source):
    model.eval()
    result = []
    with torch.inference_mode():
        fine, coarse = model.encode_atlas(source)
        fine, coarse = fine[fine_mask], coarse[coarse_mask]
        for start in range(0, 64, 4):
            block = panel[start:start + 4]
            image = torch.stack([r['image'] for r in block]).cuda()
            coordinates = torch.stack([r['coordinates'] for r in block]).cuda()
            visible = torch.stack([r['valid'] for r in block]).cuda()
            voxel = torch.div(coordinates, 200., rounding_mode='floor').long()
            coarse_voxel = torch.div(coordinates, 400., rounding_mode='floor').long()
            inside = ((voxel >= 0) & (voxel < voxel.new_tensor((d, h, w)))).all(-1)
            inside &= ((coarse_voxel >= 0) & (coarse_voxel < coarse_voxel.new_tensor((dc, hc, wc)))).all(-1)
            voxel = voxel.clamp_min(0)
            coarse_voxel = coarse_voxel.clamp_min(0)
            voxel[..., 0].clamp_(max=d - 1)
            voxel[..., 1].clamp_(max=h - 1)
            voxel[..., 2].clamp_(max=w - 1)
            coarse_voxel[..., 0].clamp_(max=dc - 1)
            coarse_voxel[..., 1].clamp_(max=hc - 1)
            coarse_voxel[..., 2].clamp_(max=wc - 1)
            fine_label = fine_lookup[(voxel[..., 0] * h + voxel[..., 1]) * w + voxel[..., 2]]
            coarse_label = coarse_lookup[(coarse_voxel[..., 0] * hc + coarse_voxel[..., 1]) * wc + coarse_voxel[..., 2]]
            valid = visible & inside & (fine_label >= 0) & (coarse_label >= 0)
            query, match_logit = model.encode_image(image)
            logits = model.similarity(query, fine)
            coarse_logits = model.similarity(query, coarse)
            top = logits.topk(8, dim=-1).indices
            flat = fine_indices[top]
            matched = torch.stack((flat // (h * w), flat // w % h, flat % w), -1)
            distance = ((matched.float() + .5) * 200 - coordinates[:, :, None]).norm(dim=-1) / 1000
            selected = match_logit.sigmoid() > .5
            for b, record in enumerate(block):
                truth, predicted = valid[b], selected[b]
                within = distance[b] < .5
                good = (truth & predicted & within.any(-1)).nonzero()[:, 0]
                selected_position = position[good]
                separation = (selected_position[:, None] - selected_position[None]).norm(dim=-1) >= 3
                triple = any(bool((separation[j] & separation[k]).any())
                             for j in range(len(good)) for k in range(j + 1, len(good))
                             if bool(separation[j, k]))
                result.append({'base': record['base'], 'mode': record['mode'],
                    'angle_deg': record['angle_deg'], 'valid_sites': int(truth.sum()),
                    'selected_sites': int(predicted.sum()), 'matchable_tp': int((truth & predicted).sum()),
                    'top1_hits': int((truth & within[:, 0]).sum()),
                    'top8_hits': int((truth & within.any(-1)).sum()),
                    'auto_top8_hits': len(good), 'three_separated': triple,
                    'fine_ce': float(F.cross_entropy(logits[b, truth], fine_label[b, truth])) if bool(truth.any()) else None,
                    'coarse_ce': float(F.cross_entropy(coarse_logits[b, truth], coarse_label[b, truth])) if bool(truth.any()) else None,
                    'top1_mean_mm': float(distance[b, truth, 0].mean()) if bool(truth.any()) else None,
                    'best8_mean_mm': float(distance[b, truth].min(-1).values.mean()) if bool(truth.any()) else None})
    return result


torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])
model = GlobalContextMatcher().cuda()
evaluated = []
for step in (0, 500, 1500):
    checkpoint = torch.load(RUN / f'match_step_{step:05d}.pt', map_location='cpu', weights_only=True) if step else None
    if checkpoint is not None:
        assert checkpoint['batch'] == step and not checkpoint['calibrated']
    for arm, source in (('full', atlas), ('support', support_atlas)):
        if checkpoint is not None:
            model.load_state_dict(checkpoint[arm])
        for row in score(model, source):
            evaluated.append({'batch': step, 'arm': arm, **row})

for actual, frozen in zip(evaluated, saved, strict=True):
    for key, value in frozen.items():
        other = actual[key]
        assert (abs(value - other) <= 1e-3 * max(1, abs(value)) if isinstance(value, float)
                else value == other), (actual['batch'], actual['arm'], actual['base'], key, value, other)


def summarize(rows):
    by_base = []
    for base in sorted({r['base'] for r in rows}):
        group = [r for r in rows if r['base'] == base]
        if any(r['valid_sites'] for r in group):
            by_base.append({key: float(np.mean([r[key] / r['valid_sites'] for r in group if r['valid_sites']]))
                            for key in ('top1_hits', 'top8_hits', 'auto_top8_hits')})
    total_valid = sum(r['valid_sites'] for r in rows)
    total_selected = sum(r['selected_sites'] for r in rows)
    total_tp = sum(r['matchable_tp'] for r in rows)
    if not total_valid:
        return {'cases': len(rows), 'valid_sites': 0}
    return {'cases': len(rows), 'valid_sites': total_valid,
        'top1_recall_0p5mm_base_equal': float(np.mean([b['top1_hits'] for b in by_base])),
        'top8_recall_0p5mm_base_equal': float(np.mean([b['top8_hits'] for b in by_base])),
        'auto_top8_recall_0p5mm_base_equal': float(np.mean([b['auto_top8_hits'] for b in by_base])),
        'top8_recall_0p5mm_site_weighted': sum(r['top8_hits'] for r in rows) / total_valid,
        'selected_site_fraction': total_selected / (256 * len(rows)),
        'matchable_precision': total_tp / total_selected if total_selected else None,
        'matchable_recall': total_tp / total_valid,
        'auto_top8_precision': sum(r['auto_top8_hits'] for r in rows) / total_tp if total_tp else None,
        'three_separated_fraction': float(np.mean([r['three_separated'] for r in rows])),
        'top1_coordinate_error_mm': float(np.mean([r['top1_mean_mm'] for r in rows if r['top1_mean_mm'] is not None])),
        'best8_coordinate_error_mm': float(np.mean([r['best8_mean_mm'] for r in rows if r['best8_mean_mm'] is not None])),
        'fine_ce': float(np.mean([r['fine_ce'] for r in rows if r['fine_ce'] is not None])),
        'base_top8_recall': [b['top8_hits'] for b in by_base]}


summary = {(step, arm): summarize([r for r in evaluated if r['batch'] == step and r['arm'] == arm])
           for step in (0, 500, 1500) for arm in ('full', 'support')}
full0, full500, full1500 = (summary[step, 'full'] for step in (0, 500, 1500))
support1500 = summary[1500, 'support']
paired_base_delta = np.array(full1500['base_top8_recall']) - np.array(support1500['base_top8_recall'])
strata = {}
for arm in ('full', 'support'):
    final = [r for r in evaluated if r['batch'] == 1500 and r['arm'] == arm]
    strata[arm] = {kind: {value: summarize([r for r in final if selector(r) == value])
                          for value in sorted({selector(r) for r in final})}
                   for kind, selector in (('mode', lambda r: r['mode']),
                                          ('angle', lambda r: '<20' if r['angle_deg'] < 20 else
                                           '20-40' if r['angle_deg'] < 40 else '>=40'))}

print(json.dumps({'verified': True, 'unique_training_sections': len(train_ids),
    'inner_sections': len(inner_ids), 'checkpoint_sha256': completed['checkpoints_sha256'],
    'metrics': {str(step): {arm: summary[step, arm] for arm in ('full', 'support')}
                for step in (0, 500, 1500)}, 'final_strata': strata,
    'gate_evidence': {'full_top8_gain_0_to_500': full500['top8_recall_0p5mm_base_equal'] - full0['top8_recall_0p5mm_base_equal'],
        'full_top8_gain_0_to_1500': full1500['top8_recall_0p5mm_base_equal'] - full0['top8_recall_0p5mm_base_equal'],
        'full_minus_support_top8_1500': float(paired_base_delta.mean()),
        'paired_base_top8_deltas_1500': paired_base_delta.tolist(),
        'three_separated_ge_25pct': full1500['three_separated_fraction'] >= .25,
        'blind_pose_and_deformation_not_evaluated': True}}, allow_nan=False))
