"""Read-only independent verifier for the completed 267 conditional information assay."""
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/v7_global_orientation_information_267_dev_001'
SOURCE = Path(__file__).resolve().parent
SUBSET_CASES = (0, 9, 18, 27, 36, 45, 54, 63)
SUBSET_AXES = (0, 17, 53, 89, 111)
sys.path.insert(0, str(SOURCE.parent))
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


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


panel_done = json.loads((PANEL / 'completed.json').read_text(encoding='utf8'))
panel_protocol = json.loads((PANEL / 'protocol.json').read_text(encoding='utf8'))
manifest = rows(PANEL / 'records.jsonl')
assert sha(PANEL / 'protocol.json') == panel_done['protocol_sha256']
assert sha(PANEL / 'records.jsonl') == panel_done['records_sha256']
assert len(manifest) == 256
by_animal = {}
for record in manifest:
    if record['eligible'] and record['valid_pixels'] > 0:
        by_animal.setdefault(record['animal_id'], []).append(record)
assert len(by_animal) == 8
selected = [record for animal in sorted(by_animal) for record in by_animal[animal][:8]]
assert len(selected) == 64 and len({r['synthetic_subject_plan_id'] for r in selected}) == 8
assert len({r['provenance']['physical_section_id'] for r in selected}) == 64

completed = json.loads((OUT / 'completed.json').read_text(encoding='utf8'))
config = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
chosen = rows(OUT / 'selected_records.jsonl')
candidates = rows(OUT / 'candidates.jsonl')
cases = rows(OUT / 'cases.jsonl')
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf8'))
assert completed['cases'] == config['selected_cases'] == 64
assert completed['axes_per_case'] == config['axes'] == 112
for name in ('config', 'selected_records', 'candidates', 'cases', 'summary'):
    extension = '.json' if name in ('config', 'summary') else '.jsonl'
    assert sha(OUT / f'{name}{extension}') == completed[f'{name}_sha256'], name
assert config['panel_completed_sha256'] == sha(PANEL / 'completed.json')
assert config['panel_protocol_sha256'] == sha(PANEL / 'protocol.json')
assert config['panel_records_sha256'] == sha(PANEL / 'records.jsonl')
assert config['selected_records_sha256'] == sha(OUT / 'selected_records.jsonl')
assert config['protocol_sha256'] == sha(SOURCE.parent / 'docs/publication/V7_GLOBAL_ORIENTATION_INFORMATION_267_PROTOCOL_20261012.md')
for name, expected in config['source_sha256'].items():
    assert sha(SOURCE / name) == expected, name
assert config['minimum_antipodal_angle_deg'] == 30
assert config['minimum_common_cells'] == 64 and config['minimum_support_iou'] == .45
assert config['maximum_support_area_fraction_gap'] == .15
assert config['qualified_case_minimum_eligible_distant_axes'] == 16
assert config['true_pose_and_reflection_used'] and not config['observed_tissue_mask_used_for_score']
assert not any(config[k] for k in ('learned_weights_used', 'calibrated', 'biological_validation', 'public_benchmark_used'))
assert not any(completed[k] for k in ('calibrated', 'biological_validation', 'public_benchmark_used'))
assert chosen == [{'selected_index': i, **record} for i, record in enumerate(selected)]
assert len(candidates) == 64 * 112 and len(cases) == 64

axis_number = np.arange(112)
z = (axis_number + .5) / 112
phi = axis_number * (math.pi * (3 - math.sqrt(5)))
axis_grid = np.stack((np.sqrt(1 - z * z) * np.cos(phi),
                      np.sqrt(1 - z * z) * np.sin(phi), z), -1)
case_arrays = {}
for i, (record, case) in enumerate(zip(selected, cases, strict=True)):
    section_id = record['provenance']['physical_section_id']
    assert case['selected_index'] == i and case['physical_section_id'] == section_id
    assert case['animal_id'] == record['animal_id']
    assert case['synthetic_subject_plan_id'] == record['synthetic_subject_plan_id']
    assert case['specimen_id'] == record['specimen_id'] and case['experiment_id'] == record['experiment_id']
    assert case['appearance_mode'] == record['appearance_mode']
    assert case['valid_fraction'] == record['valid_pixels'] / 192**2
    assert case['panel_file'] == record['file'] and case['panel_file_sha256'] == record['sha256']
    assert sha(PANEL / record['file']) == record['sha256'], record['file']
    with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
        state = arrays['target_state'].copy()
        reflection = int(arrays['reflection'])
        offsets = arrays['offsets_um'].copy()
        weights = arrays['weights'].copy()
        if i in SUBSET_CASES:
            case_arrays[i] = {'image': arrays['inputs'][0:1].copy(), 'state': state,
                              'reflection': reflection, 'offsets': offsets, 'weights': weights}
    assert np.array_equal(state, np.asarray(case['true_state']))
    assert reflection == case['reflection']
    assert np.array_equal(offsets, np.asarray(case['offsets_um']))
    assert np.array_equal(weights, np.asarray(case['weights']))
    assert case['axes'] == 112
    frame = full_frame_state_to_components(torch.from_numpy(state[None]))[1][0].numpy()
    angle = np.rad2deg(np.arccos(np.clip(np.abs(axis_grid @ frame[:, 2]), 0, 1)))
    axis_rows = candidates[i * 112:(i + 1) * 112]
    exclusions = Counter()
    eligible = []
    for axis, row in enumerate(axis_rows):
        assert row['physical_section_id'] == section_id and row['axis_index'] == axis
        assert row['animal_id'] == record['animal_id']
        assert row['synthetic_subject_plan_id'] == record['synthetic_subject_plan_id']
        assert abs(row['antipodal_angle_deg'] - angle[axis]) < .001
        if row['antipodal_angle_deg'] < 30:
            reason = 'angle_lt_30'
            assert 'common_cells' not in row
        else:
            cells = row['common_cells']
            iou = row['support_iou']
            gap = row['support_area_fraction_gap']
            assert 0 <= cells <= 96**2 and 0 <= iou <= 1 and 0 <= gap <= 1
            reason = ('common_cells' if cells < 64 else 'support_iou' if iou < .45
                      else 'support_area_gap' if gap > .15 else None)
            if cells >= 2:
                assert abs(row['intact_margin'] -
                           (row['intact_true_score'] - row['intact_wrong_score'])) < 1e-10
                assert abs(row['null_margin'] -
                           (row['null_true_score'] - row['null_wrong_score'])) < 1e-10
            else:
                assert not any(k in row for k in ('intact_margin', 'null_margin'))
        assert row['excluded_reason'] == reason and row['eligible'] == (reason is None)
        if reason is None:
            eligible.append(row)
        else:
            exclusions[reason] += 1
    assert case['distant_axes'] == 112 - exclusions['angle_lt_30']
    assert case['eligible_distant_axes'] == len(eligible)
    assert case['exclusions'] == {key: exclusions[key] for key in
                                  ('angle_lt_30', 'common_cells', 'support_iou', 'support_area_gap')}
    qualified = len(eligible) >= 16
    hardest = min(eligible, key=lambda row: row['intact_margin']) if eligible else None
    assert case['qualified'] == qualified
    assert case['intact_all_wrong_success'] == (all(r['intact_margin'] > 0 for r in eligible) if qualified else None)
    assert case['null_all_wrong_success'] == (all(r['null_margin'] > 0 for r in eligible) if qualified else None)
    assert case['hardest_wrong_axis'] == (hardest['axis_index'] if hardest else None)
    assert case['hardest_wrong_angle_deg'] == (hardest['antipodal_angle_deg'] if hardest else None)
    assert case['minimum_intact_margin'] == (hardest['intact_margin'] if hardest else None)

plans = {}
for plan in sorted({r['synthetic_subject_plan_id'] for r in cases}):
    group = [r for r in cases if r['synthetic_subject_plan_id'] == plan]
    qualified = [r for r in group if r['qualified']]
    plans[plan] = {'cases': len(group), 'qualified': len(qualified),
        'mean_eligible_distant_axes': float(np.mean([r['eligible_distant_axes'] for r in group])),
        'intact_all_wrong_success': float(np.mean([r['intact_all_wrong_success'] for r in qualified])) if qualified else None,
        'null_all_wrong_success': float(np.mean([r['null_all_wrong_success'] for r in qualified])) if qualified else None}
covered = all(p['qualified'] >= 4 for p in plans.values())
intact = float(np.mean([p['intact_all_wrong_success'] for p in plans.values()])) if covered else None
null = float(np.mean([p['null_all_wrong_success'] for p in plans.values()])) if covered else None
gate = (covered and intact >= .75 and intact - null >= .20 and
        sum(p['intact_all_wrong_success'] >= .75 for p in plans.values()) >= 6)
expected = {'cases': 64, 'plans': plans,
    'qualified_cases': sum(r['qualified'] for r in cases),
    'eligible_comparisons': sum(r['eligible_distant_axes'] for r in cases),
    'coverage_gate': covered, 'equal_plan_intact_all_wrong_success': intact,
    'equal_plan_null_all_wrong_success': null, 'conditional_information_gate': gate,
    'interpretation': 'privileged nuisance and true-pose diagnostic only; no blind accuracy'}
expected['appearance'] = {}
for mode in sorted({r['appearance_mode'] for r in cases}):
    group = [r for r in cases if r['appearance_mode'] == mode]
    qualified = [r for r in group if r['qualified']]
    expected['appearance'][mode] = {'cases': len(group), 'qualified': len(qualified),
        'intact_all_wrong_success': float(np.mean([r['intact_all_wrong_success'] for r in qualified])) if qualified else None,
        'null_all_wrong_success': float(np.mean([r['null_all_wrong_success'] for r in qualified])) if qualified else None}
expected['visible'] = {}
for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    group = [r for r in cases if low <= r['valid_fraction'] < high]
    qualified = [r for r in group if r['qualified']]
    expected['visible'][f'{low}_{high}'] = {'cases': len(group), 'qualified': len(qualified),
        'intact_all_wrong_success': float(np.mean([r['intact_all_wrong_success'] for r in qualified])) if qualified else None,
        'null_all_wrong_success': float(np.mean([r['null_all_wrong_success'] for r in qualified])) if qualified else None}
assert summary == expected
assert completed['coverage_gate'] == covered and completed['conditional_information_gate'] == gate

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance']['atlas_binding']['normalized_array_receipt'] == panel_protocol['atlas_normalized_receipt']
assert json.dumps(context['provenance'], sort_keys=True) == json.dumps(config['atlas_provenance'], sort_keys=True)
recomputed_axes = 0
recomputed_scores = 0
with torch.inference_mode():
    for case_index in SUBSET_CASES:
        data = case_arrays[case_index]
        record = selected[case_index]
        image = torch.from_numpy(data['image'][None]).cuda()
        state = torch.from_numpy(data['state'][None]).cuda()
        offsets = torch.from_numpy(data['offsets'][None]).cuda()
        weights = torch.from_numpy(data['weights'][None]).cuda()
        centre, frame, basis = full_frame_state_to_components(state)
        normal = frame[0, :, 2]
        candidates_for_case = candidates[case_index * 112:(case_index + 1) * 112]
        distant_indices = [axis for axis in SUBSET_AXES if candidates_for_case[axis]['antipodal_angle_deg'] >= 30]
        if not distant_indices:
            continue
        atlas_axes = torch.tensor(axis_grid[distant_indices], dtype=state.dtype, device='cuda')
        atlas_axes *= torch.where((atlas_axes @ normal)[:, None] < 0, -1., 1.)
        cross = torch.linalg.cross(normal[None].expand_as(atlas_axes), atlas_axes)
        cosine = atlas_axes @ normal
        skew = torch.zeros((len(distant_indices), 3, 3), dtype=state.dtype, device='cuda')
        skew[:, 0, 1], skew[:, 0, 2] = -cross[:, 2], cross[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = cross[:, 2], -cross[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -cross[:, 1], cross[:, 0]
        rotation = torch.eye(3, dtype=state.dtype, device='cuda')[None] + skew + skew @ skew / (1 + cosine)[:, None, None]
        wrong = full_frame_state_from_components(centre.expand(len(distant_indices), -1),
                                                 rotation @ frame.expand(len(distant_indices), -1, -1),
                                                 basis.expand(len(distant_indices), -1, -1))
        states = torch.cat((state, wrong), 0)
        flags = torch.full((1, len(states)), data['reflection'], dtype=torch.long, device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states[None], flags,
                                           offsets, weights, source_shape=(192, 192))[0]
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        shuffled = []
        for branch, axis in enumerate((None, *distant_indices)):
            label = 'true' if axis is None else f'axis_{axis:03d}'
            section_id = record['provenance']['physical_section_id']
            seed = int.from_bytes(hashlib.sha256(f'{section_id}|{label}'.encode()).digest()[:8], 'little') % (2**63)
            intensity = rendered[branch, 0].flatten().clone()
            inside = (rendered[branch, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(inside.numel(), device='cuda',
                                   generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            shuffled.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(shuffled)[:, None], 3, 1, 1)[:, 0]
        for branch, axis in enumerate(distant_indices, 1):
            frozen = candidates_for_case[axis]
            shared = (rendered[0, 1] > .5) & (rendered[branch, 1] > .5)
            common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
            cells = int(common.sum())
            union = int(((rendered[0, 1] > .5) | (rendered[branch, 1] > .5)).sum())
            iou = int(shared.sum()) / union if union else 0.
            gap = abs(int((rendered[0, 1] > .5).sum()) - int((rendered[branch, 1] > .5).sum())) / 96**2
            assert cells == frozen['common_cells']
            assert abs(iou - frozen['support_iou']) < 1e-7
            assert abs(gap - frozen['support_area_fraction_gap']) < 1e-7
            recomputed_axes += 1
            if cells >= 2:
                for key, candidate in (('intact_true_score', atlas[0]),
                                       ('intact_wrong_score', atlas[branch]),
                                       ('null_true_score', null_atlas[0]),
                                       ('null_wrong_score', null_atlas[branch])):
                    a, b = observed[common], candidate[common]
                    a, b = a - a.mean(), b - b.mean()
                    value = float((a * b).mean().abs() /
                                  (a.square().mean() * b.square().mean() + 1e-16).sqrt())
                    assert abs(value - frozen[key]) < 1e-4, (case_index, axis, key, value, frozen[key])
                    recomputed_scores += 1

print(json.dumps({'verified': True, 'cases': 64, 'axes_per_case': 112,
    'predeclared_scoring_subset_cases': SUBSET_CASES, 'predeclared_scoring_subset_axes': SUBSET_AXES,
    'rendered_distant_axes_recomputed': recomputed_axes, 'scores_recomputed': recomputed_scores,
    'coverage_gate': covered, 'conditional_information_gate': gate}))
