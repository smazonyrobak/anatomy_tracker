"""Paired TRAIN-inner pose readout; run only after both selected arms exit."""

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
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
frozen = root / 'runs/pose_framing_221_inner_readout_001'
anchor_223 = os.environ.get('ANATOMY_POSE_ANCHOR_CONDITIONAL_223_EVAL') == '1'
tag = '223' if anchor_223 else '222'
run_name = 'pose_anchor_conditional_223' if anchor_223 else 'pose_selection_222'
out = root / 'runs' / f'{run_name}_paired_readout_001'
parent = root / 'runs/pose_framing_221_framed_train_001/joint_step_00400.pt'
arms = {arm: root / f'runs/{run_name}_{arm}_train_001'
        for arm in ('control', 'intervention')}
protocol = source.parent / 'docs/publication' / (
    'POSE_ANCHOR_CONDITIONAL_223_PROTOCOL_20261011.md' if anchor_223 else
    'POSE_SELECTION_222_PROTOCOL_20261011.md')
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
frozen_done = json.loads((frozen / 'completed.json').read_text())
assert frozen_done['case_count'] == 160
assert all(sha(frozen / name) == digest
           for name, digest in frozen_done['output_sha256'].items())
assert all(sha(source / name) == digest
           for name, digest in frozen_done['source_sha256'].items())
fixed = [json.loads(line) for line in (frozen / 'cases.jsonl').open()]
assert {panel: sum(row['panel'] == panel for row in fixed)
        for panel in ('framed', 'unframed', 'legacy')} == {
        'framed': 64, 'unframed': 64, 'legacy': 32}
parent_done = json.loads((parent.parent / 'completed.json').read_text())
parent_config = json.loads((parent.parent / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['400']
assert sha(parent.parent / 'config.json') == parent_done['config_sha256']
assert frozen_done['input_sha256'][str(parent)] == sha(parent)
checkpoints = {'221_framed_400': parent}
arm_done, arm_config = {}, {}
for arm, directory in arms.items():
    arm_done[arm] = json.loads((directory / 'completed.json').read_text())
    arm_config[arm] = json.loads((directory / 'config.json').read_text())
    assert arm_done[arm]['updates'] == 400
    assert sha(directory / 'config.json') == arm_done[arm]['config_sha256']
    assert arm_done[arm]['parent_checkpoint_sha256'] == sha(parent)
    assert arm_done[arm]['protocol_sha256'] == sha(protocol)
    assert arm_done[arm]['source_sha256'] == arm_config[arm]['source_sha256']
    assert all(sha(source / name) == digest
               for name, digest in arm_config[arm]['source_sha256'].items())
    assert sha(directory / 'draws.jsonl') == arm_done[arm]['draws_sha256']
    assert sha(directory / 'training.jsonl') == arm_done[arm]['training_sha256']
    assert arm_done[arm]['calibrated'] is False
    assert arm_done[arm]['public_benchmark_used'] is False
    for step in (0, 200, 400):
        path = directory / f'joint_step_{step:05d}.pt'
        checkpoints[f'{tag}_{arm}_{step}'] = path
checkpoint_sha256 = {label: sha(path) for label, path in checkpoints.items()}
assert all(checkpoint_sha256[f'{tag}_{arm}_{step}'] ==
           arm_done[arm]['checkpoint_sha256'][str(step)]
           for arm in arms for step in (0, 200, 400))

context = load_streaming_synthetic_v7_64(device='cuda')
cases, new_draws = [], []
for row in fixed:
    panel, base, seed, appearance = (row[key]
        for key in ('panel', 'base', 'seed', 'appearance'))
    sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v3
    with torch.no_grad():
        if panel == 'legacy':
            virtual = int(np.random.default_rng(seed).integers(len(context['subjects'])))
            sample = sampler(context, [virtual], seed, side=256)
        else:
            source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
            if panel == 'framed':
                source_sample = sample_support_aware_framing_217(source_sample, seed)
            sample = sampler(context, [base], seed, side=256, source_sample=source_sample)
    assert bool(sample['eligible'][0])
    assert row['provenance'] == json.loads(json.dumps(sample['provenance'][0], allow_nan=False))
    assert row['physical_section_id'] == sample['provenance'][0]['physical_section_id']
    assert row['input_sha256'] == hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest()
    pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
        device='cuda').long()]
    sites = torch.stack((pixels.remainder(256),
        pixels.div(256, rounding_mode='floor')), -1).float() / 256
    cases.append({**row, 'input': sample['inputs'].cpu(), 'sites': sites.cpu(),
        'dense': sample['centre'][0].reshape(-1, 3)[pixels].cpu(),
        'state': sample['state'].cpu(), 'reflection': sample['reflection'].cpu()})

fresh_seed_prefix = 202610112340000 if anchor_223 else 202610112240000
for base in (60, 61, 62, 63):
    accepted = attempt = 0
    while accepted < 16:
        seed = fresh_seed_prefix + 10000 * base + 1009 * attempt
        appearance = 'v4' if accepted % 2 == 0 else 'v3'
        sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v3
        with torch.no_grad():
            source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
            source_sample = sample_support_aware_framing_217(source_sample, seed)
            sample = sampler(context, [base], seed, side=256, source_sample=source_sample)
        used = bool(sample['eligible'][0])
        new_draws.append({'panel': 'fresh_framed', 'base': base, 'attempt': attempt,
            'seed': seed, 'appearance': appearance, 'used': used,
            'physical_section_id': sample['provenance'][0]['physical_section_id']})
        attempt += 1
        if not used:
            continue
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        cases.append({'panel': 'fresh_framed', 'base': base, 'seed': seed,
            'appearance': appearance,
            'physical_section_id': sample['provenance'][0]['physical_section_id'],
            'provenance': sample['provenance'][0],
            'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
            'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
            'input': sample['inputs'].cpu(), 'sites': sites.cpu(),
            'dense': sample['centre'][0].reshape(-1, 3)[pixels].cpu(),
            'state': sample['state'].cpu(), 'reflection': sample['reflection'].cpu()})
        accepted += 1
assert len(cases) == 224
assert sum(case['panel'] == 'fresh_framed' for case in cases) == 64

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
rows = []
with torch.inference_mode():
    for label, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        expected_step = 400 if label == '221_framed_400' else int(label.rsplit('_', 1)[-1])
        assert saved['step'] == expected_step and saved['calibrated'] is False
        assert saved['config'] == (parent_config if label == '221_framed_400'
                                   else arm_config[label.split('_')[1]])
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for case in cases:
            sites = case['sites'].cuda()
            rigid = rigid_points_090(case['state'].cuda(),
                case['reflection'].cuda(), sites)[0]
            dense = case['dense'].cuda()
            prediction = model.predict(case['input'].cuda())
            states = prediction['state'][0].repeat_interleave(2, 0)
            reflection = torch.arange(2, device='cuda').repeat(model.modes)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            proposed = rigid_points_090(states, reflection, sites)
            affine = (proposed - rigid).norm(dim=-1).mean(-1) / 1000
            deformed = (proposed - dense).norm(dim=-1).mean(-1) / 1000
            first, top16 = int(prior.argmax()), prior.topk(16).indices
            anchor_start = 2 * model.base_modes
            top_anchor = anchor_start + int(prior[anchor_start:].argmax())
            rows.append({'checkpoint': label, 'checkpoint_sha256': checkpoint_sha256[label],
                'panel': case['panel'], 'base': case['base'], 'seed': case['seed'],
                'appearance': case['appearance'],
                'physical_section_id': case['physical_section_id'],
                'input_sha256': case['input_sha256'],
                'visible_fraction': case['visible_fraction'],
                'top1_branch_id': first,
                'top1_family': 'base' if first < anchor_start else 'anchor',
                'top1_affine_mm': float(affine[first]),
                'top1_dense_mm': float(deformed[first]),
                'best160_affine_mm': float(affine.min()),
                'best160_dense_mm': float(deformed.min()),
                'top_anchor_affine_mm': float(affine[top_anchor]),
                'top_anchor_dense_mm': float(deformed[top_anchor]),
                'best_anchor_affine_mm': float(affine[anchor_start:].min()),
                'best_anchor_dense_mm': float(deformed[anchor_start:].min()),
                'top_anchor_dense_capture_2mm': bool(deformed[top_anchor] < 2),
                'free_minus_anchor_max_log_prob_nats': float(
                    prior[:anchor_start].max() - prior[anchor_start:].max()),
                'top16_affine_capture_2mm': bool((affine[top16] < 2).any()),
                'top16_dense_capture_2mm': bool((deformed[top16] < 2).any()),
                'anchor_probability_mass': float(prediction['log_mass'][0,
                    model.base_modes:].exp().sum()),
                'base_probability_mass': float(prediction['log_mass'][0,
                    :model.base_modes].exp().sum()),
                'true_affine_to_dense_mm': float((rigid - dense).norm(dim=-1).mean() / 1000)})
        print(json.dumps({'event': 'readout', 'checkpoint': label,
            'cases': len(cases)}), flush=True)
for arm in arms:
    zero = [row for row in rows if row['checkpoint'] == f'{tag}_{arm}_0']
    baseline = [row for row in rows if row['checkpoint'] == '221_framed_400']
    assert all(a['input_sha256'] == b['input_sha256'] and
        abs(a['top1_affine_mm'] - b['top1_affine_mm']) < 1e-5 and
        abs(a['top1_dense_mm'] - b['top1_dense_mm']) < 1e-5
        for a, b in zip(zero, baseline, strict=True))

summary = {}
for panel in ('framed', 'unframed', 'legacy', 'fresh_framed'):
    summary[panel] = {}
    for label in checkpoints:
        group = [row for row in rows if row['panel'] == panel and row['checkpoint'] == label]
        summary[panel][label] = {'n': len(group),
            **{metric.replace('_mm', '') + '_median_mm': float(np.median([
                row[metric] for row in group]))
               for metric in ('top1_affine_mm', 'top1_dense_mm',
                              'best160_affine_mm', 'best160_dense_mm',
                              'top_anchor_affine_mm', 'top_anchor_dense_mm',
                              'best_anchor_affine_mm', 'best_anchor_dense_mm')},
            'top_anchor_dense_capture_2mm': sum(row['top_anchor_dense_capture_2mm']
                for row in group),
            'free_minus_anchor_max_log_prob_median_nats': float(np.median([
                row['free_minus_anchor_max_log_prob_nats'] for row in group])),
            'top16_affine_capture_2mm': sum(row['top16_affine_capture_2mm'] for row in group),
            'top16_dense_capture_2mm': sum(row['top16_dense_capture_2mm'] for row in group),
            'top1_base_count': sum(row['top1_family'] == 'base' for row in group),
            'top1_anchor_count': sum(row['top1_family'] == 'anchor' for row in group),
            'anchor_probability_mass_median': float(np.median([
                row['anchor_probability_mass'] for row in group])),
            'base_probability_mass_median': float(np.median([
                row['base_probability_mass'] for row in group]))}
summary['paired_400'] = {}
for panel in ('framed', 'unframed', 'legacy', 'fresh_framed'):
    control = summary[panel][f'{tag}_control_400']
    intervention = summary[panel][f'{tag}_intervention_400']
    summary['paired_400'][panel] = {
        key + '_intervention_minus_control': intervention[key] - control[key]
        for key in ('top1_affine_median_mm', 'top1_dense_median_mm',
                    'best160_affine_median_mm', 'best160_dense_median_mm',
                    'top_anchor_affine_median_mm', 'top_anchor_dense_median_mm',
                    'best_anchor_affine_median_mm', 'best_anchor_dense_median_mm',
                    'top_anchor_dense_capture_2mm',
                    'free_minus_anchor_max_log_prob_median_nats',
                    'top16_affine_capture_2mm', 'top16_dense_capture_2mm')}

out.mkdir(parents=True, exist_ok=False)
config = {'evaluator_source_sha256': sha(source / 'evaluate_pose_selection_222.py'),
    'experiment': run_name,
    'evaluation_source_sha256': {name: sha(source / name) for name in (
        'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_panel_geometry_train_166.py',
        'support_aware_framing_217.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'global_atlas_contrast_090.py')},
    'protocol_sha256': sha(protocol),
    'frozen_221_completion_sha256': sha(frozen / 'completed.json'),
    'frozen_221_output_sha256': frozen_done['output_sha256'],
    'frozen_221_source_sha256': frozen_done['source_sha256'],
    'checkpoint_sha256': checkpoint_sha256,
    'training_completion_sha256': {arm: sha(directory / 'completed.json')
        for arm, directory in arms.items()},
    'fresh_framed_seed_rule': f'{fresh_seed_prefix} + 10000*base + 1009*attempt; 16 eligible per base 60-63; no deduplication',
    'metric': 'mean 3D CCF distance at up to 128 fixed tissue sites, mm; affine target from true plane, dense target from synthetic deformation',
    'scope': 'TRAIN-inner only; no independent animal truth',
    'development_calibration_final_used': False,
    'public_benchmark_used': False, 'calibrated': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
with (out / 'draws.jsonl').open('w') as stream:
    for row in new_draws:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
with (out / 'cases.jsonl').open('w') as stream:
    for case in cases:
        stream.write(json.dumps({key: value for key, value in case.items()
            if key not in ('input', 'sites', 'dense', 'state', 'reflection')},
            allow_nan=False) + '\n')
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'case_count': len(cases),
    'readout_count': len(rows), 'source_sha256': config['evaluator_source_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'draws.jsonl', 'cases.jsonl', 'rows.jsonl', 'summary.json')},
    'development_calibration_final_used': False,
    'public_benchmark_used': False, 'calibrated': False}, indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'paired_readout_complete', 'cases': len(cases),
                  'rows': len(rows)}), flush=True)
