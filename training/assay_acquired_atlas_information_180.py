"""Frozen TRAIN-only acquired-versus-synthetic atlas-information assay."""

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
    compose_full_frame_state, full_frame_state_from_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(x, y, mask):
    a, b = x[mask], y[mask]
    if a.numel() < 2:
        return None, None, None
    a, b = a - a.mean(), b - b.mean()
    aa, bb = a.square().mean().sqrt(), b.square().mean().sqrt()
    return float((a * b).mean().abs() / (aa * bb).clamp_min(1e-8)), float(aa), float(bb)


torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ACQUIRED_ATLAS_INFORMATION_180_PROTOCOL_20261011.md'
out = root / 'runs/acquired_atlas_information_180_train_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
real_sources = [
    ('coronal_train', root / 'data/allen_real_training_inputs_20260929',
     'raw_model_input.npy', 'image_geometry.jsonl', 'animal_id', 'model_pixel_to_ap_dv_ml_um', 96),
    ('sagittal_train', root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'model_input.npy', 'geometry.jsonl', 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um', 256),
]
cases = []
source_hashes = {}
for cohort, directory, image_name, geometry_name, donor_key, affine_key, side in real_sources:
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    source_hashes[cohort] = {name: sha(directory / name) for name in
                             (image_name, geometry_name, 'summary.json')}
    assert source_hashes[cohort][image_name] == summary['output_sha256'][image_name]
    assert source_hashes[cohort][geometry_name] == summary['output_sha256'][geometry_name]
    rows = [json.loads(line) for line in (directory / geometry_name).read_text().splitlines()]
    by_donor = {}
    for row in rows:
        assert row['split'] == ('development_train' if cohort == 'coronal_train' else 'train')
        by_donor.setdefault(row[donor_key], []).append(row)
    assert len(by_donor) == (58 if cohort == 'coronal_train' else 40)
    images = np.load(directory / image_name, mmap_mode='r')
    assert images.shape[0] == len(rows) and images.shape[1:] == (1, side, side)
    for donor, donor_rows in sorted(by_donor.items()):
        ordered = sorted(donor_rows, key=lambda row: (row['section_number'], row['section_id']))
        row = ordered[(len(ordered) - 1) // 2]
        affine = np.asarray(row[affine_key], dtype=np.float32)
        ouv = np.stack((affine[:, 2], side * affine[:, 0], side * affine[:, 1]))
        cases.append({'cohort': cohort, 'group_id': str(donor),
            'section_key': f"{cohort}:{row['section_id']}", 'source_shape': [side, side],
            'image': np.array(images[row['array_row_index']], copy=True),
            'physical_ouv_um': ouv.tolist(), 'thickness_um':
                float(row['section_thickness_um'] if cohort == 'coronal_train' else 25.),
            'reflection': 0, 'provenance': row, 'weak_affine_reference': True})

context = load_streaming_synthetic_v7_64('cuda')
out.mkdir(parents=True, exist_ok=False)
config = {'scope': 'TRAIN-only fixed image/reference information assay; no model fitting',
    'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'real_source_sha256': source_hashes, 'synthetic_provenance': context['provenance'],
    'coronal_preprocessing': 'TRAIN-only raw red / 255, acquisition-centred 12 mm, 96 px',
    'sagittal_preprocessing': 'TRAIN-only inverted red fixed gain, acquisition-centred 256 px; area downsample to 96 px for scoring',
    'real_reference': 'upstream weak Allen affine; not independently verified anatomical truth',
    'middle_section_rule': 'sort by (section_number, section_id), choose lower middle index (n-1)//2',
    'synthetic_draw_seed': 20261011180, 'synthetic_targets': 'two independent raw eligible planes per each of 64 TRAIN deformation bases',
    'candidate_bank': ['normal_plus_1.5_mm', 'normal_minus_1.5_mm',
        'normal_plus_3_mm', 'normal_minus_3_mm', 'u_plus_15_deg', 'u_minus_15_deg',
        'v_plus_15_deg', 'v_minus_15_deg'],
    'scoring': 'absolute Pearson after 3x3 box smoothing, pairwise 5x5 eroded common atlas support',
    'support_threshold': 0.5, 'minimum_common_cells': 64, 'minimum_support_iou': .45,
    'maximum_support_area_fraction_gap': .15, 'null': 'candidate-specific deterministic within-support atlas-intensity permutation',
    'observed_tissue_mask_used': False, 'dev_or_final_used': False,
    'public_benchmark_used': False, 'probabilities_calibrated': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')

rng = np.random.default_rng(20261011180)
with (out / 'attempts.jsonl').open('w') as attempts:
    for base_index in range(64):
        accepted = 0
        while accepted < 2:
            draw_seed = int(rng.integers(0, 2**63))
            geometry = sample_panel_geometry_train_166(context, [base_index], draw_seed, 256)
            record = geometry['provenance'][0]
            raw = record['mode'] == 'raw'
            sample = (sample_one_shot_slide_artifacts_v4(context, [base_index], draw_seed,
                      256, source_sample=geometry) if raw and bool(geometry['eligible'][0]) else None)
            eligible = bool(sample['eligible'][0]) if sample is not None else False
            attempt = {'base_index': base_index, 'draw_seed': draw_seed,
                'physical_section_id': record['physical_section_id'], 'raw': raw,
                'geometry_eligible': bool(geometry['eligible'][0]),
                'v4_eligible': eligible, 'selected': eligible, 'provenance':
                    sample['provenance'][0] if sample is not None else record}
            attempts.write(json.dumps(attempt, allow_nan=False) + '\n')
            if not eligible:
                continue
            provenance = sample['provenance'][0]
            cases.append({'cohort': 'synthetic_train', 'group_id': str(base_index),
                'section_key': provenance['physical_section_id'], 'source_shape': [256, 256],
                'image': sample['inputs'][0, :1].cpu().numpy().copy(),
                'state': sample['state'][0].cpu().tolist(),
                'offsets_um': sample['offsets'][0].cpu().tolist(),
                'weights': sample['weights'][0].cpu().tolist(),
                'reflection': int(sample['reflection'][0]), 'provenance': provenance,
                'weak_affine_reference': False})
            accepted += 1
assert len(cases) == 58 + 40 + 128

updates = torch.zeros((8, 9), device='cuda', dtype=torch.float32)
updates[:4, 5] = torch.tensor([1500., -1500., 3000., -3000.], device='cuda')
updates[4:6, 0] = torch.tensor([np.pi / 12, -np.pi / 12], device='cuda')
updates[6:8, 1] = torch.tensor([np.pi / 12, -np.pi / 12], device='cuda')
bank = config['candidate_bank']
pair_rows, section_rows = [], []
with torch.inference_mode(), (out / 'sections.jsonl').open('w') as sections_file, (out / 'pairs.jsonl').open('w') as pairs_file:
    for case in cases:
        image = torch.as_tensor(case['image'], device='cuda', dtype=torch.float32)[None]
        side = case['source_shape'][0]
        if case['cohort'] == 'synthetic_train':
            state = torch.tensor(case['state'], device='cuda', dtype=torch.float32)[None]
            offsets = torch.tensor(case['offsets_um'], device='cuda', dtype=torch.float32)[None]
            weights = torch.tensor(case['weights'], device='cuda', dtype=torch.float32)[None]
        else:
            ouv = torch.tensor(case['physical_ouv_um'], device='cuda', dtype=torch.float32)[None]
            state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * case['thickness_um']
            weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda', dtype=torch.float32)[None] / 16
        states = torch.cat((state, compose_full_frame_state(state.expand(8, -1), updates)), 0)
        reflection = torch.full((1, 9), case['reflection'], device='cuda', dtype=torch.long)
        rendered = render_atlas_planes_090(context['atlas'], states[None], reflection,
                                           offsets, weights, source_shape=(side, side))[0]
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        null_images, null_seeds = [], []
        for candidate in range(9):
            seed = int.from_bytes(hashlib.sha256(f"{case['section_key']}|{candidate}".encode()).digest()[:8], 'little') % (2**63)
            null_seeds.append(seed)
            intensity = rendered[candidate, 0].flatten().clone()
            inside = (rendered[candidate, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(inside.numel(), device='cuda',
                                   generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            null_images.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(null_images)[:, None], 3, 1, 1)[:, 0]
        section = {'cohort': case['cohort'], 'group_id': case['group_id'],
            'section_key': case['section_key'], 'source_shape': case['source_shape'],
            'model_input_sha256': hashlib.sha256(case['image'].tobytes()).hexdigest(),
            'positive_state': state[0].cpu().tolist(), 'positive_physical_ouv_um':
                case.get('physical_ouv_um'), 'reflection': case['reflection'],
            'offsets_um': offsets[0].cpu().tolist(), 'weights': weights[0].cpu().tolist(),
            'thickness_um': case.get('thickness_um', case['provenance'].get('thickness_um')),
            'weak_affine_reference': case['weak_affine_reference'],
            'provenance': case['provenance'], 'eligible_pairs': 0, 'intact_wins': 0,
            'null_wins': 0, 'bank_counts': {name: {'eligible': 0, 'intact_wins': 0,
                                                  'null_wins': 0} for name in ('offset', 'tilt')}}
        for index, name in enumerate(bank, 1):
            support_pos, support_wrong = rendered[0, 1] > .5, rendered[index, 1] > .5
            shared = support_pos & support_wrong
            common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
            count, intersection, union = int(common.sum()), int(shared.sum()), int((support_pos | support_wrong).sum())
            iou = intersection / union if union else 0.
            fraction_gap = abs(int(support_pos.sum()) - int(support_wrong.sum())) / (96 * 96)
            eligible = count >= 64 and iou >= .45 and fraction_gap <= .15
            exclusion_reasons = ([reason for failed, reason in ((count < 64, 'common_cells_below_64'),
                (iou < .45, 'support_iou_below_0.45'),
                (fraction_gap > .15, 'support_area_fraction_gap_above_0.15')) if failed])
            positive, observed_contrast, positive_contrast = score(observed, atlas[0], common)
            wrong, _, wrong_contrast = score(observed, atlas[index], common)
            null_positive, _, null_positive_contrast = score(observed, null_atlas[0], common)
            null_wrong, _, null_wrong_contrast = score(observed, null_atlas[index], common)
            kind = 'offset' if index <= 4 else 'tilt'
            pair = {'cohort': case['cohort'], 'group_id': case['group_id'],
                'section_key': case['section_key'], 'candidate': name, 'candidate_type': kind,
                'wrong_state': states[index].cpu().tolist(), 'local_update': updates[index-1].cpu().tolist(),
                'null_seed_positive': null_seeds[0], 'null_seed_wrong': null_seeds[index],
                'positive_support_cells': int(support_pos.sum()),
                'wrong_support_cells': int(support_wrong.sum()),
                'intersection_cells_before_erosion': intersection, 'union_cells': union,
                'common_cells_after_erosion': count, 'support_iou': iou,
                'support_area_fraction_gap': fraction_gap, 'eligible': eligible,
                'exclusion_reasons': exclusion_reasons,
                'observed_contrast': observed_contrast, 'positive_atlas_contrast': positive_contrast,
                'wrong_atlas_contrast': wrong_contrast,
                'null_positive_atlas_contrast': null_positive_contrast,
                'null_wrong_atlas_contrast': null_wrong_contrast,
                'intact_positive_score': positive, 'intact_wrong_score': wrong,
                'intact_margin': positive - wrong if positive is not None else None,
                'null_positive_score': null_positive, 'null_wrong_score': null_wrong,
                'null_margin': null_positive - null_wrong if null_positive is not None else None}
            pair['intact_win'] = bool(positive > wrong) if eligible else None
            pair['null_win'] = bool(null_positive > null_wrong) if eligible else None
            pairs_file.write(json.dumps(pair, allow_nan=False) + '\n')
            pair_rows.append(pair)
            if eligible:
                section['eligible_pairs'] += 1
                section['intact_wins'] += pair['intact_win']
                section['null_wins'] += pair['null_win']
                section['bank_counts'][kind]['eligible'] += 1
                section['bank_counts'][kind]['intact_wins'] += pair['intact_win']
                section['bank_counts'][kind]['null_wins'] += pair['null_win']
        sections_file.write(json.dumps(section, allow_nan=False) + '\n')
        section_rows.append(section)
        if len(section_rows) in (58, 98, 226):
            print(json.dumps({'event': 'sections', 'scored': len(section_rows)}), flush=True)

summary = {'sections_total': len(section_rows), 'pairs_total': len(pair_rows),
           'cohorts': {}, 'gate': None, 'caveat': 'Acquired positives are weak Allen affines, not expert truth; acquired planes are approximately cardinal only.'}
for cohort in ('coronal_train', 'sagittal_train', 'synthetic_train'):
    cohort_sections = [row for row in section_rows if row['cohort'] == cohort]
    cohort_pairs = [row for row in pair_rows if row['cohort'] == cohort]
    cohorts = {}
    for kind in ('all', 'offset', 'tilt'):
        grouped = {}
        for row in cohort_sections:
            stats = (row if kind == 'all' else row['bank_counts'][kind])
            if stats['eligible_pairs' if kind == 'all' else 'eligible']:
                count = stats['eligible_pairs' if kind == 'all' else 'eligible']
                grouped.setdefault(row['group_id'], []).append((stats['intact_wins'] / count,
                                                                  stats['null_wins'] / count))
        group_means = [np.mean(values, axis=0) for values in grouped.values()]
        bank_pairs = (cohort_pairs if kind == 'all' else
                      [row for row in cohort_pairs if row['candidate_type'] == kind])
        eligible_pairs = [row for row in bank_pairs if row['eligible']]
        cohorts[kind] = {'sections_total': len(cohort_sections),
            'sections_with_eligible_pairs': sum((row['eligible_pairs'] if kind == 'all'
                else row['bank_counts'][kind]['eligible']) > 0 for row in cohort_sections),
            'groups_total': len({row['group_id'] for row in cohort_sections}),
            'eligible_groups': len(group_means), 'pairs_total': len(bank_pairs),
            'eligible_pairs': len(eligible_pairs), 'excluded_pairs': len(bank_pairs) - len(eligible_pairs),
            'raw_intact_wins': sum(row['intact_win'] for row in eligible_pairs),
            'raw_null_wins': sum(row['null_win'] for row in eligible_pairs),
            'equal_group_intact_win_fraction': float(np.mean([x[0] for x in group_means])) if group_means else None,
            'equal_group_null_win_fraction': float(np.mean([x[1] for x in group_means])) if group_means else None}
    summary['cohorts'][cohort] = cohorts
s, c, g = (summary['cohorts'][name]['all'] for name in
           ('synthetic_train', 'coronal_train', 'sagittal_train'))
if s['eligible_groups'] < 32 or c['eligible_groups'] < 25 or g['eligible_groups'] < 25:
    summary['gate'] = 'inconclusive_insufficient_support_matched_groups'
elif s['equal_group_intact_win_fraction'] is not None and s['equal_group_intact_win_fraction'] < .8:
    summary['gate'] = 'synthetic_pair_score_or_pose_basin_failed'
elif (s['equal_group_intact_win_fraction'] is not None and s['equal_group_intact_win_fraction'] >= .8
      and all(r['equal_group_intact_win_fraction'] is not None
              and r['equal_group_intact_win_fraction'] <= .6
              and r['equal_group_intact_win_fraction'] - r['equal_group_null_win_fraction'] <= .05
              for r in (c, g))):
    summary['gate'] = 'combined_acquired_appearance_or_weak_reference_gap'
elif (s['equal_group_intact_win_fraction'] is not None
      and s['equal_group_intact_win_fraction'] >= .8
      and all(r['equal_group_intact_win_fraction'] is not None
         and r['equal_group_intact_win_fraction'] >= .7
         and r['equal_group_intact_win_fraction'] - r['equal_group_null_win_fraction'] >= .1
         for r in (c, g))):
    summary['gate'] = 'information_present_prioritize_blind_candidate_capture_selection'
else:
    summary['gate'] = 'inconclusive'
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
receipt = {'output_sha256': {name: sha(out / name) for name in
           ('config.json', 'attempts.jsonl', 'sections.jsonl', 'pairs.jsonl', 'summary.json')},
           'sections': len(section_rows), 'pairs': len(pair_rows), 'gate': summary['gate'],
           'dev_or_final_used': False, 'public_benchmark_used': False}
(out / 'completed.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'completed', 'output': str(out), 'gate': summary['gate'],
                  'eligible_pairs_by_cohort': {name: data['all']['eligible_pairs']
                  for name, data in summary['cohorts'].items()}}, allow_nan=False), flush=True)
