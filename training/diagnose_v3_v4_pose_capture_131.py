"""Read-only frozen-128 direct-pose capture on independent TRAIN v3/v4 sections."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
out = root / 'runs/v3_v4_pose_capture_131'
parent = root / 'runs/joint_in_path_correspondence_128_treatment'
checkpoint = parent / 'joint_step_06000.pt'
parent_done = json.loads((parent / 'completed.json').read_text())
parent_config = json.loads((parent / 'config.json').read_text())
assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
assert sha(checkpoint) == parent_done['checkpoint_sha256']['6000']
assert sha(parent / 'config.json') == parent_done['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())

side = 256
seeds = {'v3': 2026101013101, 'v4': 2026101013102}
samplers = {'v3': sample_one_shot_slide_artifacts_v3,
            'v4': sample_one_shot_slide_artifacts_v4}
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 6000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
rows, draws = [], []
with torch.inference_mode():
    for cohort, sampler in samplers.items():
        for base in range(64):
            attempt = 0
            while True:
                draw_seed = int(np.random.SeedSequence(
                    [seeds[cohort], base, attempt, 0]).generate_state(1, dtype=np.uint64)[0])
                variant = int(np.random.default_rng(np.random.SeedSequence(
                    [seeds[cohort], base, attempt, 1])).integers(64))
                virtual_index = 64 * base + variant
                sample = sampler(context, [virtual_index], draw_seed, side=side)
                provenance = sample['provenance'][0]
                eligible = bool(sample['eligible'][0])
                draws.append({'cohort': cohort, 'base_index': base, 'attempt': attempt,
                    'seed': draw_seed, 'virtual_index': virtual_index,
                    'physical_section_id': provenance['physical_section_id'],
                    'eligible': eligible, 'provenance': provenance})
                if eligible:
                    break
                attempt += 1

            image = sample['inputs']
            valid = sample['valid_mask'][0].flatten().bool()
            target = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            for _ in range(2):
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity = -similarity
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            ids = beam[0]
            proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                        (ids % 2)[None], chart)[0]
            costs = ((proposed - target).norm(dim=-1)[:, valid].mean(-1) / 1000).tolist()
            best_slot = int(np.argmin(costs))
            selected_slot = int(prior[0, ids].argmax())
            top_prior_id = int(prior[0].argmax())
            top_prior_slot = ids.tolist().index(top_prior_id)
            tissue = image[0, 0].flatten()[valid]
            observed_mean = float(tissue.mean())
            normal = np.asarray(provenance['virtual_unit_normal'])
            angle = float(np.degrees(np.arccos(np.clip(np.abs(normal).max(), 0, 1))))
            exposure = (provenance['one_shot_slide_artifacts_v4']['exposure']
                        if cohort == 'v4' else 1.0)
            rows.append({'cohort': cohort, 'base_index': base,
                'synthetic_subject_id': provenance['base_lineage']['subject_id'],
                'physical_section_id': provenance['physical_section_id'],
                'virtual_index': virtual_index, 'seed': draw_seed,
                'attempts_before_eligible': attempt, 'appearance_mode': provenance['mode'],
                'nearest_cardinal_angle_deg': angle, 'applied_exposure_multiplier': exposure,
                'observed_valid_tissue_mean': observed_mean,
                'observed_valid_tissue_p95': float(torch.quantile(tissue, .95)),
                'valid_pixels': int(valid.sum()),
                'physical_plane_sha256': hashlib.sha256(json.dumps(
                    provenance['virtual_ouv_um'], separators=(',', ':')).encode()).hexdigest(),
                'beam_branch_ids': ids.tolist(), 'beam_rigid_mm': costs,
                'beam_best': {'branch_id': int(ids[best_slot]), 'rigid_mm': costs[best_slot]},
                'top_prior': {'branch_id': top_prior_id, 'rigid_mm': costs[top_prior_slot]},
                'selected_direct_prior': {'branch_id': int(ids[selected_slot]),
                                          'rigid_mm': costs[selected_slot]},
                'provenance': provenance})
        print(json.dumps({'event': 'cohort_completed', 'cohort': cohort,
                          'eligible_sections': 64}), flush=True)

assert len(rows) == 128 and len({r['physical_section_id'] for r in rows}) == 128
assert all(len({r['synthetic_subject_id'] for r in rows if r['cohort'] == cohort}) == 64
           for cohort in samplers)
assert all(r['top_prior']['branch_id'] == r['selected_direct_prior']['branch_id']
           for r in rows)

groups = {'cohort': {name: [r for r in rows if r['cohort'] == name] for name in samplers},
    'cohort_mode': {f'{name}/{mode}': [r for r in rows
        if r['cohort'] == name and r['appearance_mode'] == mode]
        for name in samplers for mode in ('raw', 'exact_black', 'imperfect_brush')},
    'cohort_angle_deg': {f'{name}/[{low},{high})': [r for r in rows
        if r['cohort'] == name and low <= r['nearest_cardinal_angle_deg'] < high]
        for name in samplers for low, high in ((0, 15), (15, 30), (30, 45), (45, 55))},
    'cohort_applied_exposure': {f'{name}/[{low},{high})': [r for r in rows
        if r['cohort'] == name and low <= r['applied_exposure_multiplier'] < high]
        for name in samplers for low, high in ((0, .15), (.15, .30), (.30, .60), (.60, 1.21))},
    'cohort_observed_tissue_mean': {f'{name}/[{low},{high})': [r for r in rows
        if r['cohort'] == name and low <= r['observed_valid_tissue_mean'] < high]
        for name in samplers for low, high in ((0, .03), (.03, .10), (.10, .30), (.30, 1.01))}}
summary = {}
for dimension, partition in groups.items():
    summary[dimension] = {}
    for name, group in partition.items():
        summary[dimension][name] = {'sections': len(group),
            'distinct_subject_ids': len({r['synthetic_subject_id'] for r in group}),
            'mean_mm': {key: float(np.mean([r[key]['rigid_mm'] for r in group])) if group else None
                for key in ('beam_best', 'top_prior', 'selected_direct_prior')},
            'capture_le_1p5': {key: float(np.mean([r[key]['rigid_mm'] <= 1.5 for r in group]))
                if group else None for key in ('beam_best', 'top_prior', 'selected_direct_prior')}}

names = ('diagnose_v3_v4_pose_capture_131.py', 'evaluate_joint_in_path_correspondence_128.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'checkpoint_sha256': sha(checkpoint),
    'checkpoint_completion_sha256': sha(parent / 'completed.json'),
    'checkpoint_config_sha256': sha(parent / 'config.json'),
    'source_sha256': {name: sha(source / name) for name in names},
    'synthetic_context_provenance': context['provenance'], 'cohort_seed_prefixes': seeds,
    'cohort_design': 'one eligible independent draw per TRAIN local-deformation base and cohort;'
                     ' disjoint v3/v4 seeds, independently sampled virtual variants and planes;'
                     ' no deliberate paired appearances or geometry deduplication',
    'metric': 'mean 3D rigid-gauge error over observed valid tissue pixels, mm;'
              ' rigid_points_090 and 256-grid x/W,y/H exactly as frozen 128 original_rigid_mm',
    'exposure_bins': 'applied v4 multiplier (v3 identity=1); observed valid-tissue mean'
                     ' is a separate common intensity proxy',
    'beam': 'frozen 128 direct-pose blind16: first 8 of modes 0:16, first 6 of'
            ' anchor modes 16:80, then two antipodal-normal diversity additions',
    'scope': 'TRAIN synthetic bases, not independent validation, animal truth, calibration or benchmark',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for filename, values in (('rows.jsonl', rows), ('draws.jsonl', draws)):
    with (out / filename).open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'eligible_sections': len(rows), 'draws': len(draws),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'rows.jsonl', 'draws.jsonl')},
    'checkpoint_sha256': config['checkpoint_sha256'],
    'evaluator_source_sha256': config['source_sha256']['diagnose_v3_v4_pose_capture_131.py'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'eligible_sections': len(rows),
                  'draws': len(draws), 'summary': summary['cohort']}), flush=True)
