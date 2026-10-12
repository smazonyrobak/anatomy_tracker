"""Direct atlas-intensity diagnostic on the frozen support-matched 299 panel."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64


SEED = 20261012302
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/CANDIDATE_INTENSITY_302_PROTOCOL_20261012.md'
PANEL = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001/inner_panel.pt'
READOUT = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001/readout.jsonl'
RUN = ROOT / 'reports/candidate_intensity_302_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pearson(left, right):
    left, right = left - left.mean(), right - right.mean()
    return float((left * right).sum() /
                 (left.square().sum() * right.square().sum()).sqrt().clamp_min(1e-8))


panel = torch.load(PANEL, map_location='cpu', weights_only=False)
readout = [json.loads(line) for line in READOUT.open(encoding='utf8')]
assert len(panel) == len(readout) == 64
assert [item['physical_section_id'] for item in panel] == [row['physical_section_id'] for row in readout]
context = load_streaming_synthetic_v7_64('cuda')
config = {'candidate': 302, 'seed': SEED, 'panel_sha256': sha(PANEL),
    'readout_sha256': sha(READOUT), 'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'audit_candidate_intensity_302.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'synthetic_provenance': context['provenance'], 'reused_adaptive_panel': True,
    'oracle_tissue_mask_for_diagnostic_only': True,
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rng = torch.Generator(device='cuda').manual_seed(SEED)
rows = []
with (RUN / 'cases.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for index, (item, learned) in enumerate(zip(panel, readout)):
            if item['pair'] is None:
                continue
            subject = item['base'] * VARIANTS + item['variant']
            sample = sample_one_shot_slide_artifacts_v9(
                context, [subject], item['seed'], 192)
            assert sample['provenance'][0]['physical_section_id'] == item['physical_section_id']
            assert torch.equal(sample['inputs'][0].cpu(), item['inputs'])
            near, far = item['pair']
            bank = item['bank'].cuda()
            intact = (sample['valid_mask'] & ~sample['fragment_displaced_mask']
                      & (sample['visible'] > .95))[:, None]
            pure = F.avg_pool2d(intact.float(), 4)[0, 0] > .999
            mask = (pure & (bank[near, 2, 1] > .95)
                    & (bank[far, 2, 1] > .95))
            assert int(mask.sum()) >= 16
            image = F.avg_pool2d(sample['inputs'][:, :1], 4)[0, 0][mask]
            scores = []
            null_scores = []
            for candidate in (near, far):
                atlas = bank[candidate, 2, 0] / bank[candidate, 2, 1].clamp_min(1e-4)
                values = atlas[mask]
                scores.append(pearson(image, values))
                null_scores.extend(abs(pearson(image,
                    values[torch.randperm(len(values), device='cuda', generator=rng)]))
                    for _ in range(16))
            margin = abs(scores[0]) - abs(scores[1])
            row = {'physical_section_id': item['physical_section_id'],
                'base': item['base'], 'slot': item['slot'],
                'obliquity_deg': item['obliquity_deg'],
                'appearance_inverted': bool(item['provenance']['appearance']['invert']),
                'brush_available': item['brush_available'],
                'near_index': near, 'far_index': far,
                'near_error_mm': float(item['error_mm'][near]),
                'far_error_mm': float(item['error_mm'][far]),
                'common_pure_tissue_blocks': int(mask.sum()),
                'near_signed_correlation': scores[0],
                'far_signed_correlation': scores[1],
                'absolute_correlation_margin': margin,
                'near_wins': margin > 0,
                'null_mean_absolute_correlation': float(np.mean(null_scores)),
                'learned_full_near_wins': learned['arms']['full']['near_far_evidence_win'],
                'learned_support_near_wins': learned['arms']['support_only']['near_far_evidence_win'],
                'provenance': item['provenance']}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if len(rows) % 16 == 0:
                print(json.dumps({'event': 'pair_milestone', 'pairs': len(rows)}), flush=True)
            del sample, bank

assert len(rows) == 49
summary = {'n_pairs': len(rows),
    'direct_near_wins': sum(row['near_wins'] for row in rows),
    'learned_full_near_wins': sum(row['learned_full_near_wins'] for row in rows),
    'learned_support_near_wins': sum(row['learned_support_near_wins'] for row in rows),
    'median_absolute_correlation_margin': float(np.median(
        [row['absolute_correlation_margin'] for row in rows])),
    'median_abs_near': float(np.median([abs(row['near_signed_correlation']) for row in rows])),
    'median_abs_far': float(np.median([abs(row['far_signed_correlation']) for row in rows])),
    'median_null_mean_abs': float(np.median(
        [row['null_mean_absolute_correlation'] for row in rows])),
    'median_common_pure_tissue_blocks': float(np.median(
        [row['common_pure_tissue_blocks'] for row in rows])),
    'direct_near_wins_oblique_45plus': sum(row['near_wins'] for row in rows
        if row['obliquity_deg'] >= 45),
    'pairs_oblique_45plus': sum(row['obliquity_deg'] >= 45 for row in rows)}
summary['diagnostic_signal_present'] = (summary['direct_near_wins'] >= 35
    and summary['median_absolute_correlation_margin'] >= .05)
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
receipt = {'candidate': 302, 'summary': summary,
    'output_sha256': {name: sha(RUN / name) for name in
        ('config.json', 'cases.jsonl', 'summary.json')},
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
