"""Measure prior-rank coverage of all frozen 226 pose proposals on the 235 cases."""

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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
run = root / 'runs/blind_beam_coverage_238_001'
protocol = source.parent / 'docs/publication/BLIND_BEAM_COVERAGE_238_PROTOCOL_20261011.md'
previous_path = root / 'runs/blind_candidate_reachability_235_001/cases.jsonl'
checkpoint = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(checkpoint) == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
assert sha(previous_path) == '1c081c4870b07af5d16cadfa1e43dafb62bbebacacc0b484a9732c03691f4e4c'
previous = [json.loads(line) for line in previous_path.read_text().splitlines()]
assert len(previous) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(saved['model'], strict=True)
del saved

widths = (1, 2, 4, 8, 16, 32, 64, 160)
rows = []
with torch.inference_mode():
    for number, prior_row in enumerate(previous, 1):
        base, appearance, seed = (prior_row[k] for k in ('base', 'appearance', 'seed'))
        framed = sample_support_aware_framing_217(
            sample_panel_geometry_train_166(context, [base], seed, side=256), seed)
        sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v7
        sample = sampler(context, [base], seed, side=256, source_sample=framed)
        assert bool(sample['eligible'][0])
        assert sample['provenance'][0]['physical_section_id'] == prior_row['physical_section_id']
        assert hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest() == prior_row['input_sha256']
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1,
            min(128, len(pixels)), device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        truth = sample['centre'][0].reshape(-1, 3)[pixels]
        prediction = model.predict(sample['inputs'])
        states = prediction['state'][0].repeat_interleave(2, 0)
        reflection = torch.arange(2, device='cuda').repeat(model.modes)
        log_prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        rigid = rigid_points_090(states, reflection, sites)
        error = ((rigid - truth[None]).norm(dim=-1).mean(-1) / 1000).cpu().numpy()
        ordered = log_prior.argsort(descending=True).cpu().numpy()
        assert np.array_equal(ordered[:16], [c['index'] for c in prior_row['top16_candidates']])
        assert np.allclose(error[ordered[:16]], [c['tissue_mm'] for c in prior_row['top16_candidates']], atol=1e-5)
        near = np.flatnonzero(error[ordered] < 2)
        rows.append({'case_index': number - 1, 'base': base, 'appearance': appearance,
            'seed': seed, 'physical_section_id': prior_row['physical_section_id'],
            'input_sha256': prior_row['input_sha256'],
            'visible_tissue_fraction': prior_row['visible_tissue_fraction'],
            'first_near_rank': int(near[0] + 1) if len(near) else None,
            'free_near': bool(np.any(error[:32] < 2)),
            'anchor_near': bool(np.any(error[32:] < 2)),
            'best_mm': {str(k): float(error[ordered[:k]].min()) for k in widths}})
        if number % 16 == 0:
            print(json.dumps({'event': 'cases', 'count': number}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    summary[appearance] = {'cases': len(group),
        'near_count': {str(k): sum(r['best_mm'][str(k)] < 2 for r in group) for k in widths},
        'best_median_mm': {str(k): float(np.median([r['best_mm'][str(k)] for r in group])) for k in widths},
        'free_near': sum(r['free_near'] for r in group),
        'anchor_near': sum(r['anchor_near'] for r in group),
        'missing_all160': sum(r['first_near_rank'] is None for r in group)}
assert summary['all']['near_count']['16'] == 51
run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
names = ('arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py',
    'support_aware_framing_217.py', 'arbitrary_plane_geometry.py')
(run / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'training_updates': 0, 'reserved_animal_used': False,
    'public_benchmark_used': False, 'calibrated': False,
    'checkpoint_sha256': sha(checkpoint), 'prior_rows_sha256': sha(previous_path),
    'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in names},
    'output_sha256': {name: sha(run / name) for name in ('rows.jsonl', 'summary.json')},
}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary['all']}), flush=True)
