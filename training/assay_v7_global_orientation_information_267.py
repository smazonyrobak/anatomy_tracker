"""Privileged full-sphere normal-information assay on 64 frozen v7 DEV sections."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/v7_global_orientation_information_267_dev_001'
DOC = Path('docs/publication/V7_GLOBAL_ORIENTATION_INFORMATION_267_PROTOCOL_20261012.md')
AXES, SIDE = 112, 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


panel_done = json.loads((PANEL / 'completed.json').read_text(encoding='utf8'))
assert sha(PANEL / 'records.jsonl') == panel_done['records_sha256']
assert sha(PANEL / 'protocol.json') == panel_done['protocol_sha256']
panel_protocol = json.loads((PANEL / 'protocol.json').read_text(encoding='utf8'))
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
assert len(records) == 256
by_plan = {}
for record in records:
    if record['eligible'] and record['valid_pixels'] > 0:
        by_plan.setdefault(record['animal_id'], []).append(record)
assert len(by_plan) == 8
selected = [record for plan in sorted(by_plan) for record in by_plan[plan][:8]]
assert len(selected) == 64
assert len({row['synthetic_subject_plan_id'] for row in selected}) == 8
assert len({row['provenance']['physical_section_id'] for row in selected}) == 64
assert all(sha(PANEL / row['file']) == row['sha256'] for row in selected)

context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance']['atlas_binding']['normalized_array_receipt'] == \
    panel_protocol['atlas_normalized_receipt']
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'selected_records.jsonl').write_text(
    ''.join(json.dumps({'selected_index': i, **row}) + '\n' for i, row in enumerate(selected)),
    encoding='utf8')
config = {
    'scope': 'read-only synthetic DEV conditional global normal information; not blind inference',
    'selection': 'first eight eligible manifest records in stable order per animal_id, before any score',
    'selected_cases': 64, 'axes': AXES, 'minimum_antipodal_angle_deg': 30,
    'axis_grid': 'hemisphere Fibonacci z=(j+.5)/112, phi=j*pi*(3-sqrt(5))',
    'transport': 'minimal proper 3D rotation from true normal; true centre/basis/reflection/PSF fixed',
    'score': '257 absolute Pearson on 3x3-smoothed image/render, 5x5-eroded pairwise common support',
    'minimum_common_cells': 64, 'minimum_support_iou': .45,
    'maximum_support_area_fraction_gap': .15,
    'qualified_case_minimum_eligible_distant_axes': 16,
    'null': 'deterministic independent candidate-specific within-atlas-support intensity permutation',
    'observed_tissue_mask_used_for_score': False, 'learned_weights_used': False,
    'true_pose_and_reflection_used': True, 'calibrated': False,
    'biological_validation': False, 'public_benchmark_used': False,
    'panel_completed_sha256': sha(PANEL / 'completed.json'),
    'panel_protocol_sha256': sha(PANEL / 'protocol.json'),
    'panel_records_sha256': sha(PANEL / 'records.jsonl'),
    'selected_records_sha256': sha(OUT / 'selected_records.jsonl'),
    'protocol_sha256': sha(DOC),
    'source_sha256': {name: sha(Path('training') / name) for name in (
        Path(__file__).name, 'assay_v7_atlas_information_257.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_geometry.py')},
    'atlas_provenance': context['provenance'],
    'numpy': np.__version__, 'torch': torch.__version__,
}
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

j = torch.arange(AXES, device='cuda', dtype=torch.float32)
z = (j + .5) / AXES
phi = j * (math.pi * (3 - math.sqrt(5)))
radius = (1 - z.square()).sqrt()
axes = torch.stack((radius * phi.cos(), radius * phi.sin(), z), -1)
case_rows = []
with torch.inference_mode(), (OUT / 'candidates.jsonl').open('w', encoding='utf8') as candidates_file, \
     (OUT / 'cases.jsonl').open('w', encoding='utf8') as cases_file:
    for case_number, record in enumerate(selected):
        section_id = record['provenance']['physical_section_id']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][0:1][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        centre, frame, basis = full_frame_state_to_components(state)
        true_normal = frame[0, :, 2]
        signed_cosine = axes @ true_normal
        angles = torch.rad2deg(signed_cosine.abs().clamp_max(1).acos())
        distant = angles >= 30
        distant_indices = distant.nonzero().flatten()
        candidate_normal = axes[distant_indices] * torch.where(
            signed_cosine[distant_indices, None] < 0, -1., 1.)
        v = torch.linalg.cross(true_normal[None].expand_as(candidate_normal), candidate_normal)
        c = (true_normal[None] * candidate_normal).sum(-1)
        zero = torch.zeros_like(v[:, 0])
        vx = torch.stack((
            torch.stack((zero, -v[:, 2], v[:, 1]), -1),
            torch.stack((v[:, 2], zero, -v[:, 0]), -1),
            torch.stack((-v[:, 1], v[:, 0], zero), -1)), -2)
        rotation = torch.eye(3, device='cuda')[None] + vx + vx @ vx / (1 + c)[:, None, None]
        candidate_frame = rotation @ frame.expand(len(distant_indices), -1, -1)
        candidate_states = full_frame_state_from_components(
            centre.expand(len(distant_indices), -1), candidate_frame,
            basis.expand(len(distant_indices), -1, -1))
        states = torch.cat((state, candidate_states), 0)
        flags = torch.full((1, len(states)), reflection, dtype=torch.long, device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states[None], flags,
                                           offsets, weights, source_shape=(SIDE, SIDE))[0]
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        null_images = []
        for branch in range(len(states)):
            label = 'true' if branch == 0 else f'axis_{int(distant_indices[branch - 1]):03d}'
            seed = int.from_bytes(hashlib.sha256(f'{section_id}|{label}'.encode()).digest()[:8],
                                  'little') % (2**63)
            intensity = rendered[branch, 0].flatten().clone()
            inside = (rendered[branch, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(inside.numel(), device='cuda',
                                   generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            null_images.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(null_images)[:, None], 3, 1, 1)[:, 0]

        branch_by_axis = {int(axis): branch for branch, axis in enumerate(distant_indices, 1)}
        eligible_rows = []
        exclusions = {'angle_lt_30': 0, 'common_cells': 0, 'support_iou': 0, 'support_area_gap': 0}
        for axis in range(AXES):
            row = {'physical_section_id': section_id, 'animal_id': record['animal_id'],
                   'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                   'axis_index': axis, 'antipodal_angle_deg': float(angles[axis]),
                   'eligible': False, 'excluded_reason': None}
            if axis not in branch_by_axis:
                row['excluded_reason'] = 'angle_lt_30'
                exclusions['angle_lt_30'] += 1
            else:
                branch = branch_by_axis[axis]
                positive_support = rendered[0, 1] > .5
                wrong_support = rendered[branch, 1] > .5
                shared = positive_support & wrong_support
                common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
                cells = int(common.sum())
                union = int((positive_support | wrong_support).sum())
                iou = int(shared.sum()) / union if union else 0.
                area_gap = abs(int(positive_support.sum()) - int(wrong_support.sum())) / 96**2
                row.update({'common_cells': cells, 'support_iou': iou,
                            'support_area_fraction_gap': area_gap})
                if cells >= 2:
                    scores = []
                    for candidate in (atlas[0], atlas[branch], null_atlas[0], null_atlas[branch]):
                        a, b = observed[common], candidate[common]
                        a, b = a - a.mean(), b - b.mean()
                        scores.append(float((a * b).mean().abs() /
                                            (a.square().mean() * b.square().mean() + 1e-16).sqrt()))
                    row.update({'intact_true_score': scores[0], 'intact_wrong_score': scores[1],
                                'null_true_score': scores[2], 'null_wrong_score': scores[3],
                                'intact_margin': scores[0] - scores[1],
                                'null_margin': scores[2] - scores[3]})
                if cells < 64:
                    row['excluded_reason'] = 'common_cells'
                elif iou < .45:
                    row['excluded_reason'] = 'support_iou'
                elif area_gap > .15:
                    row['excluded_reason'] = 'support_area_gap'
                else:
                    row['eligible'] = True
                    eligible_rows.append(row)
                if row['excluded_reason']:
                    exclusions[row['excluded_reason']] += 1
            candidates_file.write(json.dumps(row, allow_nan=False) + '\n')
        qualified = len(eligible_rows) >= 16
        hardest = min(eligible_rows, key=lambda row: row['intact_margin']) if eligible_rows else None
        case = {'selected_index': case_number, 'physical_section_id': section_id,
                'animal_id': record['animal_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                'appearance_mode': record['appearance_mode'],
                'valid_fraction': record['valid_pixels'] / SIDE**2,
                'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
                'true_state': state[0].cpu().tolist(), 'reflection': reflection,
                'offsets_um': offsets[0].cpu().tolist(), 'weights': weights[0].cpu().tolist(),
                'axes': AXES, 'distant_axes': len(distant_indices),
                'eligible_distant_axes': len(eligible_rows), 'exclusions': exclusions,
                'qualified': qualified,
                'intact_all_wrong_success': all(row['intact_margin'] > 0 for row in eligible_rows)
                                            if qualified else None,
                'null_all_wrong_success': all(row['null_margin'] > 0 for row in eligible_rows)
                                          if qualified else None,
                'hardest_wrong_axis': hardest['axis_index'] if hardest else None,
                'hardest_wrong_angle_deg': hardest['antipodal_angle_deg'] if hardest else None,
                'minimum_intact_margin': hardest['intact_margin'] if hardest else None}
        case_rows.append(case)
        cases_file.write(json.dumps(case, allow_nan=False) + '\n')
        if (case_number + 1) % 8 == 0:
            candidates_file.flush()
            cases_file.flush()
            print(json.dumps({'cases': case_number + 1, 'total': len(selected)}), flush=True)

plans = {}
for plan in sorted({row['synthetic_subject_plan_id'] for row in case_rows}):
    rows = [row for row in case_rows if row['synthetic_subject_plan_id'] == plan]
    qualified = [row for row in rows if row['qualified']]
    plans[plan] = {'cases': len(rows), 'qualified': len(qualified),
                   'mean_eligible_distant_axes': float(np.mean([row['eligible_distant_axes'] for row in rows])),
                   'intact_all_wrong_success': float(np.mean([row['intact_all_wrong_success'] for row in qualified]))
                                               if qualified else None,
                   'null_all_wrong_success': float(np.mean([row['null_all_wrong_success'] for row in qualified]))
                                             if qualified else None}
covered = all(row['qualified'] >= 4 for row in plans.values())
intact = float(np.mean([row['intact_all_wrong_success'] for row in plans.values()])) if covered else None
null = float(np.mean([row['null_all_wrong_success'] for row in plans.values()])) if covered else None
gate = (covered and intact >= .75 and intact - null >= .20 and
        sum(row['intact_all_wrong_success'] >= .75 for row in plans.values()) >= 6)
summary = {'cases': len(case_rows), 'plans': plans,
           'qualified_cases': sum(row['qualified'] for row in case_rows),
           'eligible_comparisons': sum(row['eligible_distant_axes'] for row in case_rows),
           'coverage_gate': covered, 'equal_plan_intact_all_wrong_success': intact,
           'equal_plan_null_all_wrong_success': null, 'conditional_information_gate': gate,
           'interpretation': 'privileged nuisance and true-pose diagnostic only; no blind accuracy'}
summary['appearance'] = {}
for mode in sorted({row['appearance_mode'] for row in case_rows}):
    rows = [row for row in case_rows if row['appearance_mode'] == mode]
    qualified = [row for row in rows if row['qualified']]
    summary['appearance'][mode] = {
        'cases': len(rows), 'qualified': len(qualified),
        'intact_all_wrong_success': float(np.mean([row['intact_all_wrong_success'] for row in qualified]))
                                    if qualified else None,
        'null_all_wrong_success': float(np.mean([row['null_all_wrong_success'] for row in qualified]))
                                  if qualified else None}
summary['visible'] = {}
for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    rows = [row for row in case_rows if low <= row['valid_fraction'] < high]
    qualified = [row for row in rows if row['qualified']]
    summary['visible'][f'{low}_{high}'] = {
        'cases': len(rows), 'qualified': len(qualified),
        'intact_all_wrong_success': float(np.mean([row['intact_all_wrong_success'] for row in qualified]))
                                    if qualified else None,
        'null_all_wrong_success': float(np.mean([row['null_all_wrong_success'] for row in qualified]))
                                  if qualified else None}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(case_rows), 'axes_per_case': AXES,
    'config_sha256': sha(OUT / 'config.json'),
    'selected_records_sha256': sha(OUT / 'selected_records.jsonl'),
    'candidates_sha256': sha(OUT / 'candidates.jsonl'),
    'cases_sha256': sha(OUT / 'cases.jsonl'),
    'summary_sha256': sha(OUT / 'summary.json'),
    'coverage_gate': covered, 'conditional_information_gate': gate,
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(case_rows), 'gate': gate}), flush=True)
