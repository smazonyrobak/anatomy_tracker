"""Frozen step-1300 TRAIN-only beam availability and mass-rank audit."""

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

from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = Path(__file__).resolve().parent
train_run = root / 'runs/joint_anatomy_model_153_train151_002'
out = root / 'runs/joint_anatomy_model_153_train_beam_audit_001'
checkpoint = train_run / 'joint_step_01300.pt'
completed = json.loads((train_run / 'completed.json').read_text())
assert not out.exists() and completed['selected_step'] == 1300
assert sha(checkpoint) == completed['checkpoint_sha256']['1300']
assert sha(train_run / 'inner_draws.jsonl') == completed['inner_draws_sha256']
for name, expected in completed['source_sha256'].items():
    assert sha(source / name) == expected

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
context = load_streaming_synthetic_v7_64(device='cuda')
model = JointAnatomyModel153().cuda().eval()
model.load_state_dict(saved['model'], strict=True)
assert saved['step'] == 1300 and saved['calibrated'] is False
del saved

inner = [json.loads(line) for line in (train_run / 'inner_draws.jsonl').open()]
inner = [row for row in inner if row['used']]
assert len(inner) == 16 and {row['base_index'] for row in inner} == set(range(60, 64))
rows = []
with torch.inference_mode():
    for record in inner:
        base, draw_seed = record['base_index'], record['draw_seed']
        original = sample_panel_geometry_train_151(context, [base], draw_seed, 256)
        sampler = (sample_one_shot_slide_artifacts_v4 if record['appearance_version'] == 'v4'
                   else sample_one_shot_slide_artifacts_v3)
        sample = sampler(context, [base], draw_seed, 256, source_sample=original)
        assert sample['provenance'][0]['physical_section_id'] == record['provenance']['physical_section_id']
        valid = sample['valid_mask'][0]
        pixels = valid.flatten().nonzero()[:, 0]
        chart = torch.stack((pixels.remainder(256), pixels.div(256, rounding_mode='floor')), -1).float() / 256
        truth = rigid_points_090(sample['state'], sample['reflection'], chart)
        prediction = model.pose.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()[0:160]
        costs = []
        for start in range(0, 80, 8):
            state = prediction['state'][:, start:start + 8, None].expand(-1, -1, 2, -1)
            reflections = torch.arange(2, device='cuda')[None, None].expand(1, 8, 2)
            proposed = rigid_points_090(state, reflections, chart[None, None])
            costs.append(((proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
                          / 1000).flatten())
        error = torch.cat(costs)
        near = error <= 1.5
        order = prior.argsort(descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(0, order, torch.arange(1, 161, device='cuda'))
        top2, top16 = order[:2], order[:16]
        near_rank = int(rank[near].min()) if bool(near.any()) else None
        row = {'base_index': base, 'slot': record['slot'], 'draw_seed': draw_seed,
               'appearance_version': record['appearance_version'],
               'physical_section_id': sample['provenance'][0]['physical_section_id'],
               'valid_pixels': int(valid.sum()), 'valid_fraction': float(valid.float().mean()),
               'visible_support_mass': record['provenance']['visible_support_mass'],
               'valid_queries': record['valid_queries'],
               'top2_distinct_modes': int(torch.unique(top2.div(2, rounding_mode='floor')).numel()),
               'top2_near': bool(near[top2].any()), 'top16_near': bool(near[top16].any()),
               'oracle_near_count': int(near.sum()), 'oracle_best_mm': float(error.min()),
               'direct_top1_mm': float(error[order[0]]),
               'best_near_mass_rank': near_rank,
               'best_near_prior_mass': float(prior.exp()[near].max()) if near_rank else None,
               'total_near_prior_mass': float(prior.exp()[near].sum()),
               'oracle_best_branch_rank': int(rank[error.argmin()]),
               'near_base_branches': int(near[:32].sum()),
               'near_anchor_branches': int(near[32:].sum()),
               'rigid_error_mm_by_branch': error.cpu().tolist(),
               'prior_log_mass_by_branch': prior.cpu().tolist()}
        rows.append(row)
        print(json.dumps({'event': 'case', **{key: value for key, value in row.items()
            if key not in ('rigid_error_mm_by_branch', 'prior_log_mass_by_branch')}}), flush=True)
        del original, sample, prediction


def summary(items):
    near_rows = [row for row in items if row['oracle_near_count']]
    return {'n': len(items), 'oracle_near_cases': len(near_rows),
            'top2_near_cases': sum(row['top2_near'] for row in items),
            'top16_near_cases': sum(row['top16_near'] for row in items),
            'top2_distinct_mode_cases': sum(row['top2_distinct_modes'] == 2 for row in items),
            'mean_oracle_best_mm': float(np.mean([row['oracle_best_mm'] for row in items])),
            'mean_direct_top1_mm': float(np.mean([row['direct_top1_mm'] for row in items])),
            'median_near_mass': (float(np.median([row['total_near_prior_mass'] for row in near_rows]))
                                 if near_rows else None),
            'median_best_near_mass_rank': (float(np.median([row['best_near_mass_rank'] for row in near_rows]))
                                           if near_rows else None)}


result = {'all': summary(rows),
          'valid_fraction_below_8pct': summary([row for row in rows if row['valid_fraction'] < .08]),
          'valid_fraction_at_least_8pct': summary([row for row in rows if row['valid_fraction'] >= .08]),
          'by_base': {str(base): summary([row for row in rows if row['base_index'] == base])
                      for base in range(60, 64)}}
out.mkdir(parents=True, exist_ok=False)
(out / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
(out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'training_completion_sha256': sha(train_run / 'completed.json'),
    'checkpoint_sha256': sha(checkpoint), 'inner_draws_sha256': sha(train_run / 'inner_draws.jsonl'),
    'diagnostic_source_sha256': sha(__file__), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'split': 'held_out_train_bases_60_to_63', 'dev_or_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': result}), flush=True)
