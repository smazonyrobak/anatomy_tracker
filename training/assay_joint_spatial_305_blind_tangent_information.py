"""No-training atlas-direction assay around 305's actual blind pose choices."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090

source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/JOINT_SPATIAL_305_BLIND_TANGENT_INFORMATION_PROTOCOL_20261014.md'
parent_run = root / 'runs/joint_spatial_305_acquired_rehearsal_001'
checkpoint = parent_run / 'intervention/pose_local_00400.pt'
output = root / 'runs/joint_spatial_305_blind_tangent_information_001'
assert source.drive.upper() == output.drive.upper() == 'I:' and not output.exists()
assert json.loads((parent_run / 'completed.json').read_text())['summary']['step_400_gate_pass']
with checkpoint.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_sha == json.loads((parent_run / 'intervention/completed.json').read_text(
    ))['output_sha256'][checkpoint.name]

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
model.eval()
context = load_streaming_synthetic_v7_64('cuda')
seed = 20261014308
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
steps = torch.tensor((math.pi / 15, math.pi / 15, math.pi / 15,
                      1000., 1000., 1000., math.log(1.1), math.log(1.1), .1),
                     device='cuda')
updates = torch.zeros(18, 9, device='cuda')
updates[0::2, torch.arange(9)] = steps
updates[1::2, torch.arange(9)] = -steps
config = {'candidate': 305, 'seed': seed, 'cases_per_base': 8,
          'train_inner_bases': list(range(56, 64)), 'source_side': 192, 'score_side': 96,
          'tangent_axes': ['rot_u', 'rot_v', 'roll', 'translate_u', 'translate_v',
                           'translate_normal', 'log_span_u', 'log_span_v', 'shear'],
          'tangent_half_steps': steps.tolist(), 'case_choice': 'blind prior top one',
          'minimum_physical_pair_gap_mm': .10, 'minimum_weighted_common_cells': 64,
          'minimum_support_iou': .45, 'maximum_support_area_gap': .15,
          'checkpoint_sha256': checkpoint_sha,
          'parent_completion_sha256': hashlib.sha256((parent_run / 'completed.json').read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'generator_provenance': context['provenance'], 'calibrated': False,
          'no_dev_cal_final_or_public_benchmark': True}
output.mkdir(parents=True, exist_ok=False)
(output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

rows, cases = [], []
with (output / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (output / 'pairs.jsonl').open('w', encoding='utf8') as pairs, \
        torch.inference_mode():
    for base in range(56, 64):
        accepted = attempted = 0
        while accepted < 8:
            attempted += 1
            variant = int(rng.integers(VARIANTS))
            draw_seed = int(rng.integers(2**63 - 1, dtype=np.int64))
            sample = sample_postartifact_gauge_v1(
                context, [base * VARIANTS + variant], draw_seed, 192)
            used = bool(sample['eligible'][0])
            identity = sample['provenance'][0]['physical_section_id']
            draws.write(json.dumps({'base': base, 'attempt': attempted,
                'accepted_slot': accepted if used else None, 'variant': variant,
                'seed': draw_seed, 'accepted': used, 'provenance': sample['provenance'][0]},
                allow_nan=False) + '\n')
            if not used:
                continue
            accepted += 1
            prediction = model.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            branch = int(prior.argmax())
            mode, reflection = divmod(branch, 2)
            initial = prediction['state'][0, mode]
            states = compose_full_frame_state(initial.expand(18, -1), updates)
            flags = torch.full((1, 18), reflection, dtype=torch.long, device='cuda')
            rendered = render_atlas_planes_090(context['atlas'], states[None], flags,
                sample['offsets'], sample['weights'], candidate_chunk=2,
                source_shape=(192, 192))[0]
            source_image = F.avg_pool2d(F.interpolate(
                sample['inputs'][:, :1], (96, 96), mode='area'), 3, 1, 1)[0, 0]
            atlas_image = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
            support = rendered[:, 1] > .5
            source_q = F.interpolate(prediction['reliability_logit'], (96, 96),
                mode='bilinear', align_corners=False).sigmoid()[0, 0]
            q192 = F.interpolate(prediction['reliability_logit'], (192, 192),
                mode='bilinear', align_corners=False).sigmoid()[0, 0]
            valid = sample['valid_mask'][0]
            pixel_gen = torch.Generator(device='cuda').manual_seed(draw_seed)
            pixels = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                                       generator=pixel_gen)
            chart = torch.stack((pixels.remainder(192),
                pixels.div(192, rounding_mode='floor')), -1).float() / 192
            truth = sample['centre'][0].reshape(-1, 3)[pixels]
            mapped = rigid_points_090(states, flags[0], chart, (192, 192))
            errors = (mapped - truth[None]).norm(dim=-1).mean(-1) / 1000
            initial_error = float((rigid_points_090(initial[None], flags[0, :1], chart,
                (192, 192))[0] - truth).norm(dim=-1).mean() / 1000)
            true_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            obliquity = math.degrees(math.acos(float(true_normal.abs().max().clamp(0, 1))))
            case = {'physical_section_id': identity, 'base': base, 'variant': variant,
                'seed': draw_seed, 'blind_branch': branch, 'visible_fraction': float(valid.float().mean()),
                'obliquity_deg': obliquity, 'q_valid_mean': float(q192[valid].mean()),
                'q_exterior_mean': float(q192[~valid].mean()) if bool((~valid).any()) else None,
                'blind_initial_error_mm': initial_error}
            cases.append(case)
            null_atlas = []
            for branch_id in range(18):
                intensity = rendered[branch_id, 0].flatten().clone()
                inside = support[branch_id].flatten().nonzero().flatten()
                permutation = torch.randperm(inside.numel(), device='cuda', generator=
                    torch.Generator(device='cuda').manual_seed(draw_seed + branch_id + 1))
                intensity[inside] = intensity[inside][permutation]
                null_atlas.append(intensity.reshape(96, 96))
            null_atlas = F.avg_pool2d(torch.stack(null_atlas)[:, None], 3, 1, 1)[:, 0]
            for axis in range(9):
                plus, minus = 2 * axis, 2 * axis + 1
                shared = support[plus] & support[minus]
                common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
                weighted = source_q * common
                mass = float(weighted.sum())
                union = int((support[plus] | support[minus]).sum())
                iou = int(shared.sum()) / union if union else 0.
                area_gap = abs(int(support[plus].sum()) - int(support[minus].sum())) / 96**2
                physical_gap = float((errors[plus] - errors[minus]).abs())
                eligible = (physical_gap >= .10 and mass >= 64 and iou >= .45 and
                            area_gap <= .15)
                scores = {'intensity': [], 'shuffled': [], 'support_only': []}
                for candidate in (plus, minus):
                    weight = weighted / weighted.sum().clamp_min(1e-8)
                    x = source_image - (weight * source_image).sum()
                    for arm, image in (('intensity', atlas_image[candidate]),
                                       ('shuffled', null_atlas[candidate])):
                        y = image - (weight * image).sum()
                        covariance = (weight * x * y).sum()
                        denominator = ((weight * x.square()).sum() *
                                       (weight * y.square()).sum()).sqrt().clamp_min(1e-8)
                        scores[arm].append(float((covariance / denominator).abs()))
                    scores['support_only'].append(float((source_q *
                        rendered[candidate, 1]).sum() / source_q.sum().clamp_min(1e-8)))
                truth_plus = bool(errors[plus] < errors[minus])
                wins = {arm: (.5 if abs(value[0] - value[1]) < 1e-10 else
                    float((value[0] > value[1]) == truth_plus)) for arm, value in scores.items()}
                row = {**case, 'axis': config['tangent_axes'][axis],
                    'plus_error_mm': float(errors[plus]), 'minus_error_mm': float(errors[minus]),
                    'physical_gap_mm': physical_gap, 'weighted_common_cells': mass,
                    'support_iou': iou, 'support_area_gap': area_gap, 'eligible': eligible,
                    'truth_prefers_plus': truth_plus, 'scores': scores, 'wins': wins}
                rows.append(row)
                pairs.write(json.dumps(row, allow_nan=False) + '\n')
            if len(cases) % 16 == 0:
                pairs.flush()
                draws.flush()
                print(json.dumps({'event': 'case_milestone', 'accepted': len(cases)}), flush=True)

eligible_rows = [row for row in rows if row['eligible']]
base_rows = {base: [row for row in eligible_rows if row['base'] == base]
             for base in range(56, 64)}
base_means = {arm: [float(np.mean([row['wins'][arm] for row in group]))
    for group in base_rows.values() if group] for arm in ('intensity', 'support_only', 'shuffled')}
equal_base = {arm: float(np.mean(values)) if values else None
              for arm, values in base_means.items()}
strata = {}
for name, select in (
        ('near_axis', lambda row: row['obliquity_deg'] < 20),
        ('oblique', lambda row: row['obliquity_deg'] >= 20),
        ('visible_lt_half', lambda row: row['visible_fraction'] < .5),
        ('visible_ge_half', lambda row: row['visible_fraction'] >= .5)):
    group = [row for row in eligible_rows if select(row)]
    strata[name] = {'pairs': len(group),
        'physical_sections': len({row['physical_section_id'] for row in group}),
        'intensity_win': float(np.mean([row['wins']['intensity'] for row in group])) if group else None}
gate = (len(eligible_rows) >= 40 and sum(bool(group) for group in base_rows.values()) >= 6
    and equal_base['intensity'] >= .70
    and equal_base['intensity'] - equal_base['support_only'] >= .15
    and equal_base['intensity'] - equal_base['shuffled'] >= .15)
summary = {'accepted_cases': len(cases), 'all_pairs': len(rows),
    'informative_pairs': len(eligible_rows),
    'informative_bases': sum(bool(group) for group in base_rows.values()),
    'by_base_pair_counts': {str(base): len(group) for base, group in base_rows.items()},
    'equal_base_directional_win': equal_base, 'strata': strata, 'gate_pass': gate,
    'mean_q_valid': float(np.mean([case['q_valid_mean'] for case in cases])),
    'mean_q_exterior': float(np.mean([case['q_exterior_mean'] for case in cases
        if case['q_exterior_mean'] is not None])),
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(output / 'completed.json').write_text(json.dumps({'summary': summary,
    'output_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ('config.json', 'draws.jsonl', 'pairs.jsonl', 'summary.json')}},
    indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'gate_pass': gate,
    'informative_pairs': len(eligible_rows), 'equal_base': equal_base}), flush=True)
