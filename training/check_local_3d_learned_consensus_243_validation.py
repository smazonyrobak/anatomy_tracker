"""Development-only learned 2-D/3-D match and robust-consensus mechanism check."""

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
from training.local_3d_consensus_fit_239 import fit_local_3d_consensus
from training.local_3d_descriptor_240 import Local3DDescriptor240, render_candidate_bank_240
from training.local_3d_learned_match_243 import local_3d_learned_match
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
source = Path(__file__).resolve().parent
training_run = root / 'runs/learned_coherent_3d_match_240_train_001'
run = root / 'runs/local_3d_learned_consensus_243_validation_001'
protocol = source.parent / 'docs/publication/LEARNED_LOCAL_CONSENSUS_243_PROTOCOL_20261012.md'
parent = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


receipt = json.loads((training_run / 'completed.json').read_text(encoding='utf-8'))
config = json.loads((training_run / 'config.json').read_text(encoding='utf-8'))
manifest = json.loads((training_run / 'validation_manifest.json').read_text(encoding='utf-8'))
assert receipt['training_exited'] and sha(training_run / 'best.pt') == receipt['checkpoint_sha256']
assert sha(parent) == config['parent_sha256']
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
context = load_streaming_synthetic_v7_64(device='cuda')
saved_parent = torch.load(parent, map_location='cpu', weights_only=True)
proposer = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(proposer, enabled=True)
proposer.load_state_dict(saved_parent['model'], strict=True)
descriptor = Local3DDescriptor240().cuda().eval().requires_grad_(False)
descriptor.load_state_dict(torch.load(training_run / 'best.pt',
    map_location='cpu', weights_only=True)['descriptor'], strict=True)
del saved_parent

rows = []
with torch.inference_mode():
    for item in manifest:
        sampler = sample_one_shot_slide_artifacts_v4 if item['appearance'] == 'v4' else sample_one_shot_slide_artifacts_v7
        framed = sample_support_aware_framing_217(
            sample_panel_geometry_train_166(context, [item['base']], item['seed'], side=256), item['seed'])
        sample = sampler(context, [item['base']], item['seed'], side=256, source_sample=framed)
        assert bool(sample['eligible'][0])
        assert sample['provenance'][0]['physical_section_id'] == item['physical_section_id']
        assert hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest() == item['input_sha256']
        pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
        pixels = pixels[torch.linspace(0, len(pixels) - 1,
            min(128, len(pixels)), device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        truth = sample['centre'][0].reshape(-1, 3)[pixels]
        prediction = proposer.predict(sample['inputs'])
        states = prediction['state'][0].repeat_interleave(2, 0)
        reflections = torch.arange(2, device='cuda').repeat(proposer.modes)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        beam = prior.topk(16).indices
        error = (rigid_points_090(states, reflections, sites) - truth[None]).norm(dim=-1).mean(-1) / 1000
        near_choices = beam[error[beam] < 2]
        near = int(near_choices[error[near_choices].argmin()]) if len(near_choices) else None
        source_grid = descriptor.encode_source(sample['inputs'][:, :1])
        row = {'physical_section_id': item['physical_section_id'],
            'appearance': item['appearance'], 'near_branch': near,
            'direct_mm': float(error[prior.argmax()]), 'near': None, 'true': None}
        roles = [('true', sample['state'], sample['reflection'])]
        if near is not None:
            roles.append(('near', states[near:near + 1], reflections[near:near + 1]))
        for role, state, reflection in roles:
            initial = float((rigid_points_090(state, reflection, sites)[0]
                - truth).norm(dim=-1).mean() / 1000)
            result = {'initial_mm': initial, 'reflection': int(reflection[0]), 'arms': {}}
            for arm in ('intact', 'null'):
                null_seed = None if arm == 'intact' else int.from_bytes(hashlib.sha256(
                    f'240|development-null|{item["physical_section_id"]}|{role}'.encode()).digest()[:8],
                    'little') % (2**63)
                bank, support = render_candidate_bank_240(context['atlas'], state,
                    reflection, sample['offsets'], sample['weights'], null_seed=null_seed)
                atlas_grid = descriptor.encode_atlas(bank)
                match = local_3d_learned_match(source_grid, atlas_grid, support, state, reflection)
                fitter_seed = int.from_bytes(hashlib.sha256(
                    f'243|fit|{item["physical_section_id"]}|{role}'.encode()).digest()[:8],
                    'little') % (2**63)
                fitted, diagnostics = fit_local_3d_consensus(state, reflection, match, fitter_seed)
                fitted_mm = float((rigid_points_090(fitted, reflection, sites)[0]
                    - truth).norm(dim=-1).mean() / 1000)

                chart = match['query_chart']
                yx = ((chart[:, [1, 0]] * 256).round().long()).clamp(0, 255)
                indices = yx[:, 0] * 256 + yx[:, 1]
                tissue_world = sample['centre'][0].reshape(-1, 3)[indices]
                valid = sample['valid_mask'][0].flatten()[indices].bool() & match['support']
                site_error = (match['matched_world'] - tissue_world).norm(dim=-1) / 1000
                errors = site_error.cpu().tolist()
                valid_sites = valid.cpu().tolist()
                supported = match['support']
                result['arms'][arm] = {
                    'fitted_mm': fitted_mm,
                    'matched_sites': int(supported.sum()),
                    'valid_match_sites': int(valid.sum()),
                    'mean_raw_cosine': float(match['raw_score'][supported].mean()) if bool(supported.any()) else None,
                    'mean_local_score': float(match['score'][supported].mean()) if bool(supported.any()) else None,
                    'mean_confidence': float(match['confidence'][supported].mean()) if bool(supported.any()) else None,
                    'mean_site_error_mm': float(site_error[valid].mean()) if bool(valid.any()) else None,
                    'site_error_mm': [e if keep else None for e, keep in zip(errors, valid_sites)],
                    **diagnostics,
                }
            row[role] = result
        rows.append(row)
        if len(rows) % 8 == 0:
            print(json.dumps({'event': 'cases', 'count': len(rows)}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    near_group = [r for r in group if r['near'] is not None]
    summary[appearance] = {'cases': len(group), 'near_cases': len(near_group),
        'near_initial_mean_mm': float(np.mean([r['near']['initial_mm'] for r in near_group])) if near_group else None,
        'true_initial_mean_mm': float(np.mean([r['true']['initial_mm'] for r in group]))}
    for role, cases in (('near', near_group), ('true', group)):
        for arm in ('intact', 'null'):
            fits = [r[role]['arms'][arm] for r in cases]
            site_errors = [e for fit in fits for e in fit['site_error_mm'] if e is not None]
            raw_scores = [fit['mean_raw_cosine'] for fit in fits if fit['mean_raw_cosine'] is not None]
            local_scores = [fit['mean_local_score'] for fit in fits if fit['mean_local_score'] is not None]
            summary[appearance][f'{role}_{arm}_fitted_mean_mm'] = (
                float(np.mean([fit['fitted_mm'] for fit in fits])) if fits else None)
            summary[appearance][f'{role}_{arm}_site_error_mean_mm'] = (
                float(np.mean(site_errors)) if site_errors else None)
            summary[appearance][f'{role}_{arm}_matched_sites_mean'] = (
                float(np.mean([fit['matched_sites'] for fit in fits])) if fits else None)
            summary[appearance][f'{role}_{arm}_valid_match_sites_mean'] = (
                float(np.mean([fit['valid_match_sites'] for fit in fits])) if fits else None)
            summary[appearance][f'{role}_{arm}_raw_cosine_mean'] = (
                float(np.mean(raw_scores)) if raw_scores else None)
            summary[appearance][f'{role}_{arm}_local_score_mean'] = (
                float(np.mean(local_scores)) if local_scores else None)
            summary[appearance][f'{role}_{arm}_inliers_total'] = sum(fit['inliers'] for fit in fits)
            summary[appearance][f'{role}_{arm}_accepted'] = sum(fit['accepted'] for fit in fits)
assert len(rows) == 32
run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
(run / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'training_updates': 0,
    'checkpoint_sha256': sha(training_run / 'best.pt'),
    'training_receipt_sha256': sha(training_run / 'completed.json'),
    'validation_manifest_sha256': sha(training_run / 'validation_manifest.json'),
    'parent_checkpoint_sha256': sha(parent),
    'protocol_sha256': sha(protocol),
    'script_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in (
        'local_3d_learned_match_243.py', 'local_3d_descriptor_240.py',
        'local_3d_consensus_fit_239.py')},
    'public_benchmark_used': False, 'reserved_animal_used': False, 'calibrated': False,
    'output_sha256': {name: sha(run / name) for name in ('rows.jsonl', 'summary.json')},
}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary['all']}), flush=True)
