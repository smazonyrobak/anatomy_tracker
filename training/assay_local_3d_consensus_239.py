"""Fresh TRAIN-inner conditional assay of local 3-D geometric consensus."""

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
from training.local_3d_patch_match_237 import local_3d_patch_match
from training.local_3d_consensus_fit_239 import fit_local_3d_consensus
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
run = root / 'runs/local_3d_consensus_239_001'
protocol = source.parent / 'docs/publication/LOCAL_3D_CONSENSUS_239_PROTOCOL_20261011.md'
checkpoint = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(checkpoint) == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
context = load_streaming_synthetic_v7_64(device='cuda')
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(saved['model'], strict=True)
del saved

rows = []
draws = []
rng = np.random.default_rng(20261011239)
with torch.inference_mode():
    for base in (56, 57, 58, 59):
        for appearance, sampler in (('v4', sample_one_shot_slide_artifacts_v4),
                                    ('v7', sample_one_shot_slide_artifacts_v7)):
            accepted = 0
            while accepted < 8:
                seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                framed = sample_support_aware_framing_217(
                    sample_panel_geometry_train_166(context, [base], seed, side=256), seed)
                sample = sampler(context, [base], seed, side=256, source_sample=framed)
                physical_id = sample['provenance'][0]['physical_section_id']
                eligible = bool(sample['eligible'][0])
                draws.append({'base': base, 'appearance': appearance, 'seed': seed,
                              'physical_section_id': physical_id, 'used': eligible})
                if not eligible:
                    continue
                accepted += 1
                pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
                pixels = pixels[torch.linspace(0, len(pixels) - 1,
                    min(128, len(pixels)), device='cuda').long()]
                sites = torch.stack((pixels.remainder(256),
                    pixels.div(256, rounding_mode='floor')), -1).float() / 256
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                prediction = model.predict(sample['inputs'])
                states = prediction['state'][0].repeat_interleave(2, 0)
                reflections = torch.arange(2, device='cuda').repeat(model.modes)
                prior = (prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                beam = prior.topk(16).indices
                error = (rigid_points_090(states, reflections, sites) - truth[None]).norm(dim=-1).mean(-1) / 1000
                near_candidates = beam[error[beam] < 2]
                near = int(near_candidates[error[near_candidates].argmin()]) if len(near_candidates) else None
                row = {'case_index': len(rows), 'base': base, 'appearance': appearance,
                    'seed': seed, 'physical_section_id': physical_id,
                    'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
                    'visible_tissue_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'near_branch': near, 'true_reflection': int(sample['reflection'][0]),
                    'results': {}}
                roles = {'true': (sample['state'], sample['reflection'])}
                if near is not None:
                    roles['near'] = (states[near:near + 1], reflections[near:near + 1])
                for role, (state, reflection) in roles.items():
                    initial = float((rigid_points_090(state, reflection, sites)[0]
                        - truth).norm(dim=-1).mean() / 1000)
                    row['results'][role] = {'initial_mm': initial,
                        'reflection': int(reflection[0]), 'arms': {}}
                    for arm in ('intact', 'null'):
                        null_seed = None if arm == 'intact' else int.from_bytes(hashlib.sha256(
                            f"239|{physical_id}|{role}".encode()).digest()[:8], 'little') % (2**63)
                        match = local_3d_patch_match(sample['inputs'][:, :1], state,
                            reflection, context['atlas'], sample['offsets'], sample['weights'],
                            null_seed=null_seed)
                        fitter_seed = int.from_bytes(hashlib.sha256(
                            f"239|fit|{physical_id}|{role}".encode()).digest()[:8], 'little') % (2**63)
                        fitted, diagnostics = fit_local_3d_consensus(state, reflection, match, fitter_seed)
                        fitted_error = float((rigid_points_090(fitted, reflection, sites)[0]
                            - truth).norm(dim=-1).mean() / 1000)
                        row['results'][role]['arms'][arm] = {'fitted_mm': fitted_error,
                            'supported_sites': int(match['support'].sum()), **diagnostics}
                rows.append(row)
                if len(rows) % 16 == 0:
                    print(json.dumps({'event': 'cases', 'count': len(rows)}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    near_group = [r for r in group if r['near_branch'] is not None]
    summary[appearance] = {'cases': len(group), 'near_cases': len(near_group),
        'near_initial_mean_mm': float(np.mean([r['results']['near']['initial_mm'] for r in near_group])),
        'near_intact_fitted_mean_mm': float(np.mean([r['results']['near']['arms']['intact']['fitted_mm'] for r in near_group])),
        'near_null_fitted_mean_mm': float(np.mean([r['results']['near']['arms']['null']['fitted_mm'] for r in near_group])),
        'true_intact_fitted_mean_mm': float(np.mean([r['results']['true']['arms']['intact']['fitted_mm'] for r in group])),
        'true_null_fitted_mean_mm': float(np.mean([r['results']['true']['arms']['null']['fitted_mm'] for r in group])),
        'near_reflection_correct': sum(r['results']['near']['reflection'] == r['true_reflection'] for r in near_group),
        'near_intact_updated': sum(r['results']['near']['arms']['intact']['updated'] for r in near_group),
        'true_intact_updated': sum(r['results']['true']['arms']['intact']['updated'] for r in group)}
all_result = summary['all']
summary['stage_a_pass'] = (all_result['near_initial_mean_mm'] - all_result['near_intact_fitted_mean_mm'] >= .25
    and all_result['near_null_fitted_mean_mm'] - all_result['near_intact_fitted_mean_mm'] >= .15
    and all_result['true_intact_fitted_mean_mm'] <= .20
    and all(summary[a]['near_initial_mean_mm'] > summary[a]['near_intact_fitted_mean_mm'] for a in ('v4', 'v7')))
assert len(rows) == 64 and sum(r['appearance'] == 'v4' for r in rows) == 32
run.mkdir(parents=True, exist_ok=False)
with (run / 'draws.jsonl').open('w', encoding='utf-8') as stream:
    for draw in draws:
        stream.write(json.dumps(draw, allow_nan=False) + '\n')
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
    'local_3d_patch_match_237.py', 'local_3d_consensus_fit_239.py',
    'support_aware_framing_217.py', 'arbitrary_plane_geometry.py')
(run / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'draw_attempts': len(draws), 'training_updates': 0,
    'reserved_animal_used': False, 'public_benchmark_used': False,
    'calibrated': False, 'checkpoint_sha256': sha(checkpoint),
    'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in names},
    'output_sha256': {name: sha(run / name) for name in ('draws.jsonl', 'rows.jsonl', 'summary.json')},
}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary['all'], 'stage_a_pass': summary['stage_a_pass']}), flush=True)
