"""Development-only fixed-site score ablation on 240's validation sections."""

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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.local_3d_descriptor_240 import (
    Local3DDescriptor240, query_chart_240, render_candidate_bank_240,
    sample_candidate_features_240,
)
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
source = Path(__file__).resolve().parent
training_run = root / 'runs/learned_coherent_3d_match_240_train_001'
run = root / 'runs/learned_3d_fixed_site_242_validation_001'
protocol = source.parent / 'docs/publication/FIXED_SITE_SCORE_242_PROTOCOL_20261012.md'
parent = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def refine_fixed_sites(source_features, atlas_features, support, candidate_state,
                       reflection, steps=24, learning_rate=.12):
    with torch.inference_mode(False), torch.enable_grad():
        source_grid = source_features.detach().clone()
        atlas_grid = atlas_features.detach().clone()
        availability = support.detach().clone()
        start = candidate_state.detach().clone()
        centre, frame, basis = full_frame_state_to_components(start)
        edges = frame[:, :, :2] @ basis
        original = torch.stack((centre[0], edges[0, :, 0], edges[0, :, 1]))
        chart = query_chart_240(source_grid.device, source_grid.dtype)
        canonical_x = torch.where(torch.as_tensor(reflection, device=chart.device).bool(),
                                  255 / 256 - chart[:, 0], chart[:, 0])
        design = torch.stack((torch.ones_like(canonical_x), canonical_x - .5,
                              chart[:, 1] - .5), -1)
        anchors = start.new_tensor(((1, -.5, -.5), (1, .5, -.5),
                                    (1, -.5, .5), (1, .5, .5), (1, 0, 0)))
        query = source_grid[:, :, ::2, ::2].flatten(2).transpose(1, 2)
        initial_keys, initial_availability = sample_candidate_features_240(
            atlas_grid, availability, start, reflection, (design @ original)[None])
        fixed = initial_availability[0] > .5
        initial_score = (query[:, fixed] * initial_keys[:, fixed]).sum(-1).mean(-1)
        delta = torch.zeros_like(original, requires_grad=True)
        best_delta = delta.detach().clone()
        best_score = initial_score.detach()
        site_loss_rejections = 0
        optimizer = torch.optim.Adam((delta,), lr=learning_rate)
        for _ in range(steps):
            coefficients = original + 1000 * delta
            keys, covered = sample_candidate_features_240(
                atlas_grid, availability, start, reflection, (design @ coefficients)[None])
            score = (query[:, fixed] * keys[:, fixed]).sum(-1).mean(-1)
            all_fixed_supported = bool((covered[:, fixed] > .5).all())
            site_loss_rejections += not all_fixed_supported
            if all_fixed_supported and bool((score > best_score).all()):
                best_score = score.detach()
                best_delta = delta.detach().clone()
            gradient, = torch.autograd.grad(-score.sum(), delta)
            optimizer.zero_grad(set_to_none=True)
            delta.grad = gradient
            optimizer.step()
            with torch.no_grad():
                movement = torch.linalg.vector_norm(anchors @ delta * 1000, dim=-1).mean()
                delta.mul_(2499 / movement.clamp_min(2499))
        coefficients = original + 1000 * delta
        keys, covered = sample_candidate_features_240(
            atlas_grid, availability, start, reflection, (design @ coefficients)[None])
        trial_score = (query[:, fixed] * keys[:, fixed]).sum(-1).mean(-1)
        all_fixed_supported = bool((covered[:, fixed] > .5).all())
        site_loss_rejections += not all_fixed_supported
        if all_fixed_supported and bool((trial_score > best_score).all()):
            best_delta = delta.detach().clone()
        coefficients = original + 1000 * best_delta
        keys, covered = sample_candidate_features_240(
            atlas_grid, availability, start, reflection, (design @ coefficients)[None])
        final_score = (query[:, fixed] * keys[:, fixed]).sum(-1).mean(-1)
        movement = torch.linalg.vector_norm(anchors @ (coefficients - original), dim=-1).mean()
        edge_u, edge_v = coefficients[1], coefficients[2]
        sine = torch.linalg.vector_norm(torch.cross(edge_u, edge_v, dim=-1)) / (
            torch.linalg.vector_norm(edge_u) * torch.linalg.vector_norm(edge_v)).clamp_min(1)
        valid_frame = bool((sine > 1e-3).detach()
                           and (torch.linalg.vector_norm(edge_u) > 1).detach()
                           and (torch.linalg.vector_norm(edge_v) > 1).detach())
        accepted = bool((final_score >= initial_score + .02).all()
                        and (movement <= 2500).all() and valid_frame
                        and (covered[:, fixed] > .5).all())
        origin = coefficients[0] - .5 * (edge_u + edge_v)
        fitted = (full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((origin, edge_u, edge_v))[None])) if accepted else start)
        return fitted.detach(), {
            'accepted': accepted,
            'fixed_sites': int(fixed.sum()),
            'initial_score': float(initial_score[0].detach()),
            'best_trial_score': float(final_score[0].detach()),
            'fitted_score': float((final_score if accepted else initial_score)[0].detach()),
            'initial_supported': int((initial_availability > .5).sum()),
            'best_trial_supported': int((covered > .5).sum()),
            'site_loss_rejections': int(site_loss_rejections),
            'five_anchor_movement_um': float(movement.detach()),
            'valid_frame': valid_frame,
        }


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
with torch.no_grad():
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
                fitted, diagnostics = refine_fixed_sites(source_grid, atlas_grid,
                    support, state, reflection)
                fitted_mm = float((rigid_points_090(fitted, reflection, sites)[0]
                    - truth).norm(dim=-1).mean() / 1000)
                result['arms'][arm] = {'fitted_mm': fitted_mm, **diagnostics}
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
            summary[appearance][f'{role}_{arm}_fitted_mean_mm'] = (
                float(np.mean([r[role]['arms'][arm]['fitted_mm'] for r in cases])) if cases else None)
            summary[appearance][f'{role}_{arm}_updated'] = sum(
                r[role]['arms'][arm]['accepted'] for r in cases)
            summary[appearance][f'{role}_{arm}_mean_score_gain'] = (
                float(np.mean([r[role]['arms'][arm]['fitted_score'] -
                               r[role]['arms'][arm]['initial_score'] for r in cases])) if cases else None)
assert len(rows) == 32
run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
(run / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'training_updates': 0,
    'checkpoint_sha256': sha(training_run / 'best.pt'),
    'training_receipt_sha256': sha(training_run / 'completed.json'),
    'validation_manifest_sha256': sha(training_run / 'validation_manifest.json'),
    'script_sha256': sha(__file__), 'local_3d_descriptor_240_sha256': sha(source / 'local_3d_descriptor_240.py'),
    'protocol_sha256': sha(protocol),
    'public_benchmark_used': False, 'reserved_animal_used': False, 'calibrated': False,
    'output_sha256': {name: sha(run / name) for name in ('rows.jsonl', 'summary.json')},
}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary['all']}), flush=True)
