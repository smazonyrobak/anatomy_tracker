"""TRAIN-inner fixed-case readout for the 226 pose exposure, not qualification."""

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
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
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
training = root / 'runs/pose_appearance_exposure_226_train_001'
parent_run = root / 'runs/pose_framing_221_framed_train_001'
retained_run = root / 'runs/pose_framing_221_inner_readout_001'
run = root / 'runs/pose_appearance_exposure_226_train_readout_001'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
parent = parent_run / 'joint_step_00400.pt'
parent_done = json.loads((parent_run / 'completed.json').read_text())
done = json.loads((training / 'completed.json').read_text())
retained_done = json.loads((retained_run / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['400']
assert done['parent_checkpoint_sha256'] == sha(parent)
assert sha(retained_run / 'cases.jsonl') == retained_done['output_sha256']['cases.jsonl']
checkpoints = [('221_parent', parent)] + [
    (f'226_{step:04d}', training / f'joint_step_{step:05d}.pt')
    for step in (0, 400, 800, 1200)]
assert all(sha(path) == done['checkpoint_sha256'][str(int(label.split('_')[-1]))]
           for label, path in checkpoints[1:])
context = load_streaming_synthetic_v7_64(device='cuda')
cases = []
retained = [json.loads(line) for line in (retained_run / 'cases.jsonl').open()]
retained = [row for row in retained if row['panel'] in ('framed', 'legacy')]
assert len(retained) == 96


def add_case(panel, base, seed, appearance, expected=None):
    sampler = (sample_one_shot_slide_artifacts_v7 if appearance == 'v7' else
               sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else
               sample_one_shot_slide_artifacts_v3)
    with torch.no_grad():
        if panel == 'legacy':
            virtual = int(np.random.default_rng(seed).integers(len(context['subjects'])))
            sample = sampler(context, [virtual], seed, side=256)
        else:
            source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
            source_sample = sample_support_aware_framing_217(source_sample, seed)
            sample = sampler(context, [base], seed, side=256, source_sample=source_sample)
    if not bool(sample['eligible'][0]):
        return False
    image_hash = hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest()
    identity = sample['provenance'][0]['physical_section_id']
    if expected is not None:
        assert identity == expected['physical_section_id'] and image_hash == expected['input_sha256']
    pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
        device='cuda').long()]
    sites = torch.stack((pixels.remainder(256), pixels.div(256, rounding_mode='floor')), -1).float() / 256
    normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
    cases.append({'panel': panel, 'base': base, 'seed': seed, 'appearance': appearance,
        'physical_section_id': identity, 'input_sha256': image_hash,
        'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
        'cardinal_angle_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp_max(1)))),
        'input': sample['inputs'].cpu(), 'sites': sites.cpu(),
        'truth': sample['centre'][0].reshape(-1, 3)[pixels].cpu(), 'normal': normal.cpu()})
    return True


for row in retained:
    assert add_case(row['panel'], row['base'], row['seed'], row['appearance'], row)
draws = []
for base in (60, 61, 62, 63):
    accepted = attempt = 0
    while accepted < 16:
        seed = 202610112260000 + 10000 * base + 1009 * attempt
        used = add_case('v7_framed', base, seed, 'v7')
        draws.append({'base': base, 'attempt': attempt, 'seed': seed, 'used': used,
            'physical_section_id': cases[-1]['physical_section_id'] if used else None})
        attempt += 1
        accepted += int(used)
assert len(cases) == 160 and len({case['physical_section_id'] for case in cases}) == 160

rows = []
for label, path in checkpoints:
    checkpoint_hash = sha(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()
    attach_global_plane_matcher(model, enabled=True)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    with torch.inference_mode():
        for case in cases:
            prediction = model.predict(case['input'].cuda())
            states = prediction['state'][0].repeat_interleave(2, 0)
            reflection = torch.arange(2, device='cuda').repeat(model.modes)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            proposed = rigid_points_090(states, reflection, case['sites'].cuda())
            error = (proposed - case['truth'].cuda()[None]).norm(dim=-1).mean(-1) / 1000
            normal = full_frame_state_to_components(states)[1][..., :, 2]
            angle = torch.rad2deg(torch.acos((normal * case['normal'].cuda()).sum(-1).abs().clamp_max(1)))
            top, best16 = int(prior.argmax()), prior.topk(16).indices
            rows.append({'checkpoint': label, 'checkpoint_sha256': checkpoint_hash,
                'panel': case['panel'], 'base': case['base'], 'seed': case['seed'],
                'appearance': case['appearance'], 'physical_section_id': case['physical_section_id'],
                'input_sha256': case['input_sha256'], 'visible_fraction': case['visible_fraction'],
                'cardinal_angle_deg': case['cardinal_angle_deg'], 'top1_tissue_mm': float(error[top]),
                'all160_oracle_tissue_mm': float(error.min()), 'top1_axis_deg': float(angle[top]),
                'top16_capture_2mm': bool((error[best16] < 2).any())})
    del model
    torch.cuda.empty_cache()

summary = {}
for label, _ in checkpoints:
    summary[label] = {}
    for panel in ('framed', 'legacy', 'v7_framed'):
        panel_rows = [row for row in rows if row['checkpoint'] == label and row['panel'] == panel]
        summary[label][panel] = {'n': len(panel_rows),
            'top1_median_mm': float(np.median([row['top1_tissue_mm'] for row in panel_rows])),
            'all160_median_mm': float(np.median([row['all160_oracle_tissue_mm'] for row in panel_rows])),
            'axis_median_deg': float(np.median([row['top1_axis_deg'] for row in panel_rows])),
            'top16_capture_2mm': float(np.mean([row['top16_capture_2mm'] for row in panel_rows]))}
        print(json.dumps({'event': 'readout', 'checkpoint': label, 'panel': panel,
                          **summary[label][panel]}), flush=True)

run.mkdir(parents=True, exist_ok=False)
for name, records in (('draws.jsonl', draws), ('cases.jsonl', [
    {key: value for key, value in case.items() if key not in ('input', 'sites', 'truth', 'normal')}
    for case in cases]), ('rows.jsonl', rows)):
    with (run / name).open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(run / 'completed.json').write_text(json.dumps({
    'case_count': len(cases), 'row_count': len(rows),
    'protocol_sha256': sha(source.parent / 'docs/publication/POSE_APPEARANCE_EXPOSURE_226_PROTOCOL_20261011.md'),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_pose_appearance_exposure_226.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_panel_geometry_train_166.py', 'support_aware_framing_217.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'arbitrary_plane_one_shot_slide_artifacts_v6.py',
        'arbitrary_plane_one_shot_slide_artifacts_v7.py')},
    'checkpoint_sha256': {label: sha(path) for label, path in checkpoints},
    'output_sha256': {name: sha(run / name) for name in
        ('draws.jsonl', 'cases.jsonl', 'rows.jsonl', 'summary.json')},
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False,
    'calibrated': False}, indent=2))
print(json.dumps({'event': 'completed', 'case_count': len(cases), 'row_count': len(rows)}), flush=True)
