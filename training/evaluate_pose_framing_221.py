"""Frozen TRAIN-only pose readout for the two 221 framing continuations."""

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
run = root / 'runs/pose_framing_221_inner_readout_001'
arms = {arm: root / f'runs/pose_framing_221_{arm}_train_001'
        for arm in ('unframed', 'framed')}
parent = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt'
parent_done = json.loads(parent.with_name('completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['800'] and not run.exists()
for arm, directory in arms.items():
    done = json.loads((directory / 'completed.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    assert done['updates'] == 400 and done['new_geometry_166_presentations'] == 800
    assert config['parent_checkpoint_sha256'] == sha(parent)
    assert config['support_aware_framing_217'] == (arm == 'framed')
    assert sha(directory / 'config.json') == done['config_sha256']
    assert all(sha(directory / f'joint_step_{step:05d}.pt') ==
               done['checkpoint_sha256'][str(step)] for step in (0, 200, 400))

context = load_streaming_synthetic_v7_64(device='cuda')
cases, draws = [], []
for panel in ('unframed', 'framed', 'legacy'):
    for base in (60, 61, 62, 63) if panel != 'legacy' else (0,):
        accepted = attempt = 0
        target = 16 if panel != 'legacy' else 32
        while accepted < target:
            seed = (202610112210000 + {'unframed': 0, 'framed': 1000000,
                    'legacy': 2000000}[panel] + 10000 * base + 1009 * attempt)
            appearance = 'v4' if accepted % 2 == 0 else 'v3'
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            with torch.no_grad():
                if panel == 'legacy':
                    virtual = int(np.random.default_rng(seed).integers(len(context['subjects'])))
                    sample = sampler(context, [virtual], seed, side=256)
                else:
                    source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
                    if panel == 'framed':
                        source_sample = sample_support_aware_framing_217(source_sample, seed)
                    sample = sampler(context, [base], seed, side=256,
                                     source_sample=source_sample)
            used = bool(sample['eligible'][0])
            draws.append({'panel': panel, 'base': base, 'attempt': attempt, 'seed': seed,
                          'appearance': appearance, 'used': used,
                          'physical_section_id': sample['provenance'][0]['physical_section_id']})
            attempt += 1
            if not used:
                continue
            pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
                device='cuda').long()]
            sites = torch.stack((pixels.remainder(256),
                pixels.div(256, rounding_mode='floor')), -1).float() / 256
            normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            cardinal_angle = float(torch.rad2deg(torch.acos(normal.abs().max().clamp_max(1))))
            cases.append({'panel': panel, 'base': base, 'seed': seed,
                'appearance': appearance, 'physical_section_id':
                sample['provenance'][0]['physical_section_id'],
                'provenance': sample['provenance'][0],
                'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'cardinal_angle_deg': cardinal_angle,
                'input': sample['inputs'].cpu(), 'sites': sites.cpu(),
                'truth': sample['centre'][0].reshape(-1, 3)[pixels].cpu(),
                'normal': normal.cpu()})
            accepted += 1
assert len(cases) == 160 and len({row['physical_section_id'] for row in cases}) == 160

checkpoints = [('187_step_0800', parent)] + [
    (f'221_{arm}_step_{step:04d}', directory / f'joint_step_{step:05d}.pt')
    for arm, directory in arms.items() for step in (200, 400)]
records = []
for label, path in checkpoints:
    checkpoint_hash = sha(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()
    attach_global_plane_matcher(model, enabled=True)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    for case in cases:
        with torch.inference_mode():
            prediction = model.predict(case['input'].cuda())
            states = prediction['state'][0].repeat_interleave(2, 0)
            reflection = torch.arange(2, device='cuda').repeat(model.modes)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            proposed = rigid_points_090(states, reflection, case['sites'].cuda())
            error = (proposed - case['truth'].cuda()[None]).norm(dim=-1).mean(-1) / 1000
            normal = full_frame_state_to_components(states)[1][..., :, 2]
            angle = torch.rad2deg(torch.acos((normal * case['normal'].cuda()).sum(-1
                ).abs().clamp_max(1)))
            first, top16 = int(prior.argmax()), prior.topk(16).indices
            records.append({'checkpoint': label, 'checkpoint_sha256': checkpoint_hash,
                'panel': case['panel'], 'base': case['base'], 'seed': case['seed'],
                'appearance': case['appearance'],
                'physical_section_id': case['physical_section_id'],
                'input_sha256': case['input_sha256'],
                'visible_fraction': case['visible_fraction'],
                'cardinal_angle_deg': case['cardinal_angle_deg'],
                'top1_tissue_mm': float(error[first]),
                'top16_oracle_tissue_mm': float(error[top16].min()),
                'all160_oracle_tissue_mm': float(error.min()),
                'top1_axis_deg': float(angle[first]),
                'all160_oracle_axis_deg': float(angle[int(error.argmin())]),
                'top16_capture_2mm': bool((error[top16] < 2).any())})
    del model
    torch.cuda.empty_cache()
    for panel in ('unframed', 'framed', 'legacy'):
        rows = [row for row in records if row['checkpoint'] == label and row['panel'] == panel]
        print(json.dumps({'event': 'readout', 'checkpoint': label, 'panel': panel,
            'n': len(rows), 'top1_median_mm': float(np.median([r['top1_tissue_mm'] for r in rows])),
            'all160_median_mm': float(np.median([r['all160_oracle_tissue_mm'] for r in rows])),
            'axis_median_deg': float(np.median([r['top1_axis_deg'] for r in rows])),
            'top16_capture_2mm': sum(r['top16_capture_2mm'] for r in rows) / len(rows)}),
            flush=True)

run.mkdir(parents=True, exist_ok=False)
with (run / 'draws.jsonl').open('w') as stream:
    for row in draws:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
with (run / 'cases.jsonl').open('w') as stream:
    for case in cases:
        stream.write(json.dumps({key: value for key, value in case.items()
            if key not in ('input', 'sites', 'truth', 'normal')}, allow_nan=False) + '\n')
with (run / 'readout.jsonl').open('w') as stream:
    for row in records:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'completed.json').write_text(json.dumps({
    'case_count': len(cases), 'readout_count': len(records),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_pose_framing_221.py', 'arbitrary_plane_panel_geometry_train_166.py',
        'support_aware_framing_217.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py')},
    'input_sha256': {str(path): sha(path) for _, path in checkpoints},
    'output_sha256': {name: sha(run / name) for name in
        ('draws.jsonl', 'cases.jsonl', 'readout.jsonl')},
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False,
    'calibrated': False}, indent=2))
print(json.dumps({'event': 'completed', 'case_count': len(cases),
                  'readout_count': len(records)}), flush=True)
