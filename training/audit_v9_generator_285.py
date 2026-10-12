"""Bounded TRAIN-only v9 generator coverage and affine-gauge audit."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import frame_to_physical_ouv
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import (
    sample_one_shot_slide_artifacts_v9,
)
from training.arbitrary_plane_streaming_synthetic_v7 import VARIANTS
from training.arbitrary_plane_streaming_synthetic_v7_64 import (
    load_streaming_synthetic_v7_64,
)


SEED = 202610285000
SIDE = 192
BASES = 64
DRAWS_PER_BASE = 8
RUN = ROOT / 'reports/v9_generator_audit_285_train_001'
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/V9_GENERATOR_AUDIT_285_PROTOCOL_20261012.md'
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE_NAMES = (
    'audit_v9_generator_285.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_subject_sampling_v6.py',
    'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'subject_deformed_slab_multiresolution_bundle_v2.py',
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {name: float(np.quantile(values, q)) for name, q in
            (('min', 0), ('q10', .1), ('q25', .25), ('median', .5),
             ('q75', .75), ('q90', .9), ('max', 1))} if len(values) else None


def cohort(rows):
    return {
        'sections': len(rows),
        'sampling_branch_counts': {name: sum(r['sampling_branch'] == name for r in rows)
                                   for name in ('through_support_point', 'uniform_box_offset')},
        'nearest_cardinal_angle_deg': quantiles([r['nearest_cardinal_angle_deg'] for r in rows]),
        'absolute_normal_components_mean_ap_dv_ml': np.mean(
            [r['absolute_unit_normal_ap_dv_ml'] for r in rows], axis=0).tolist() if rows else None,
        'visible_support_mass': quantiles([r['visible_support_mass'] for r in rows]),
        'source_visible_support_mass': quantiles([r['source_visible_support_mass'] for r in rows]),
        'valid_pixels': quantiles([r['valid_pixels'] for r in rows]),
        'sites_available_for_2048': quantiles([r['sites_available_for_2048'] for r in rows]),
        'fraction_with_at_least_128_sites': float(np.mean(
            [r['valid_pixels'] >= 128 for r in rows])) if rows else None,
        'fraction_with_full_2048_sites': float(np.mean(
            [r['valid_pixels'] >= 2048 for r in rows])) if rows else None,
        'artifact_rates': {event: {
            'event_rate': float(np.mean([r['artifacts'][event]['event'] for r in rows])) if rows else None,
            'on_tissue_rate_all_sections': float(np.mean(
                [r['artifacts'][event]['on_tissue'] for r in rows])) if rows else None,
            'on_tissue_fraction_given_event': (sum(r['artifacts'][event]['on_tissue'] for r in rows) /
                sum(r['artifacts'][event]['event'] for r in rows))
                if any(r['artifacts'][event]['event'] for r in rows) else None,
            'affected_pixels': quantiles([r['artifacts'][event]['affected_pixels'] for r in rows
                                          if r['artifacts'][event]['event'] and
                                          r['artifacts'][event]['affected_pixels'] is not None]),
        } for event in EVENTS},
    }


torch.set_num_threads(4)
assert RUN.drive.upper() == 'I:' and not RUN.exists()
assert torch.cuda.is_available()
assert BASES == 64 and VARIANTS >= DRAWS_PER_BASE
config = {
    'seed': SEED,
    'side': SIDE,
    'bases': BASES,
    'draws_per_base': DRAWS_PER_BASE,
    'proposals': BASES * DRAWS_PER_BASE,
    'split': 'TRAIN-only',
    'draw_policy': 'eight variants per base without replacement; one draw and one appearance per physical section; no retries',
    'accepted_definition': 'generator eligible flag',
    'dense_site_count': 'min(valid_mask pixels, 2048)',
    'intact_fit_mask': 'valid_mask AND NOT fragment_displaced_mask',
    'robust_fit': 'OLS followed by three Huber IRLS updates; scale=max(1.4826*MAD(norm residual),25 um); cutoff=1.345*scale',
    'affine_energy_decomposition': 'valid-intact support OLS projection of target-minus-saved displacement onto [1,x,y], the affine subspace removed by mapper projection',
    'pose_label_stage': 'pre-artifact source affine gauge; v9 performs no post-artifact state refit',
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'development_used': False,
    'calibration_used': False,
    'final_animals_used': False,
    'real_data_used': False,
    'public_benchmark_used': False,
    'model_loaded_or_updated': False,
}
RUN.mkdir(parents=True)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

context = load_streaming_synthetic_v7_64(device='cuda')
(RUN / 'generator_provenance.json').write_text(
    json.dumps(context['provenance'], indent=2, allow_nan=False), encoding='utf8')
rows = []
with torch.no_grad(), (RUN / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for base in range(BASES):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, base, 285]))
        variants = rng.choice(VARIANTS, DRAWS_PER_BASE, replace=False).tolist()
        subject_indices = [base * VARIANTS + variant for variant in variants]
        draw_seed = SEED + base * 100003
        sample = sample_one_shot_slide_artifacts_v9(
            context, subject_indices, draw_seed, SIDE)
        centers = sample['centre'].cpu().numpy()
        valid_masks = sample['valid_mask'].cpu().numpy()
        displaced_masks = sample['fragment_displaced_mask'].cpu().numpy()
        visible = sample['visible'].cpu().numpy()
        states = sample['state'].cpu()
        reflections = sample['reflection'].cpu().numpy().astype(bool)
        saved_ouv = frame_to_physical_ouv(
            *full_frame_state_to_components(states)).reshape(-1, 3, 3).numpy()

        for local, (subject_index, variant) in enumerate(zip(subject_indices, variants)):
            provenance = sample['provenance'][local]
            normal = np.asarray(provenance['virtual_unit_normal'], dtype=np.float64)
            intact = valid_masks[local] & ~displaced_masks[local]
            yy, xx = np.nonzero(intact)
            x = xx.astype(np.float64) / SIDE
            if reflections[local]:
                x = (SIDE - 1) / SIDE - x
            design = np.column_stack((np.ones(len(x)), x, yy.astype(np.float64) / SIDE))
            target = centers[local][intact].astype(np.float64)
            gauge = None
            if len(design) >= 3 and np.linalg.matrix_rank(design) == 3:
                ordinary_coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
                coefficients = ordinary_coefficients.copy()
                weights = np.ones(len(design), dtype=np.float64)
                for _ in range(3):
                    residual = np.linalg.norm(target - design @ coefficients, axis=1)
                    median = np.median(residual)
                    scale = max(1.4826 * np.median(np.abs(residual - median)), 25.)
                    cutoff = 1.345 * scale
                    weights = np.minimum(1., cutoff / np.maximum(
                        residual, np.finfo(np.float64).eps))
                    root_weight = np.sqrt(weights)
                    coefficients = np.linalg.lstsq(
                        design * root_weight[:, None],
                        target * root_weight[:, None], rcond=None)[0]

                saved_prediction = design @ saved_ouv[local]
                ordinary_prediction = design @ ordinary_coefficients
                fitted_prediction = design @ coefficients
                total_displacement = target - saved_prediction
                affine_component = ordinary_prediction - saved_prediction
                local_component = target - ordinary_prediction
                total_squared = float(np.square(total_displacement).sum())
                saved_residual = np.linalg.norm(total_displacement, axis=1)
                fitted_residual = np.linalg.norm(target - fitted_prediction, axis=1)
                separation = np.linalg.norm(saved_prediction - fitted_prediction, axis=1)
                saved_normal = np.cross(saved_ouv[local, 1], saved_ouv[local, 2])
                fitted_normal = np.cross(coefficients[1], coefficients[2])
                saved_normal /= np.linalg.norm(saved_normal)
                fitted_normal /= np.linalg.norm(fitted_normal)
                normal_angle = math.degrees(math.acos(float(np.clip(
                    abs(saved_normal @ fitted_normal), -1, 1))))
                center_chart = np.array([1., .5, .5])
                gauge = {
                    'fit_condition_number': float(np.linalg.cond(
                        design * np.sqrt(weights)[:, None])),
                    'saved_residual_mean_mm': float(saved_residual.mean() / 1000),
                    'saved_residual_median_mm': float(np.median(saved_residual) / 1000),
                    'robust_fit_residual_mean_mm': float(fitted_residual.mean() / 1000),
                    'robust_fit_residual_median_mm': float(np.median(fitted_residual) / 1000),
                    'saved_to_fit_plane_separation_mean_mm': float(separation.mean() / 1000),
                    'saved_to_fit_plane_separation_median_mm': float(np.median(separation) / 1000),
                    'saved_to_fit_center_separation_mm': float(np.linalg.norm(
                        center_chart @ saved_ouv[local] - center_chart @ coefficients) / 1000),
                    'saved_to_fit_unsigned_normal_angle_deg': normal_angle,
                    'total_target_displacement_rms_mm': float(np.sqrt(
                        np.square(total_displacement).sum(axis=1).mean()) / 1000),
                    'affine_component_rms_mm': float(np.sqrt(
                        np.square(affine_component).sum(axis=1).mean()) / 1000),
                    'representable_local_residual_rms_mm': float(np.sqrt(
                        np.square(local_component).sum(axis=1).mean()) / 1000),
                    'affine_component_fraction_squared_error': (
                        float(np.square(affine_component).sum() / total_squared)
                        if total_squared > 0 else 0.),
                    'ols_orthogonality_relative_error': float(abs(
                        total_squared - np.square(affine_component).sum() -
                        np.square(local_component).sum()) / max(total_squared, 1e-12)),
                    'saved_ouv_um': saved_ouv[local].tolist(),
                    'support_ols_ouv_um': ordinary_coefficients.tolist(),
                    'robust_fit_ouv_um': coefficients.tolist(),
                }
            v3 = provenance['one_shot_slide_artifacts_v3']
            pixel_counts = v3['pixel_counts']
            affected = {
                'fragment': pixel_counts['fragment_displaced'],
                'fold': pixel_counts['fold_occlusion_invalid'],
                'bubble': pixel_counts['bubble_defocus_invalid'],
                'tile_seam': None,
            }
            artifacts = {event: {
                'event': bool(v3['events'][event]),
                'tissue_placed': bool(v3['tissue_placed'][event]),
                'on_tissue': bool(v3['events'][event] and v3['tissue_placed'][event]),
                'affected_pixels': (int(affected[event])
                                    if affected[event] is not None else None),
            } for event in EVENTS}
            record = {
                'proposal_index': len(rows),
                'base_index': base,
                'variant': variant,
                'subject_index': subject_index,
                'draw_seed': draw_seed,
                'physical_section_id': provenance['physical_section_id'],
                'split': provenance['split'],
                'eligible': bool(sample['eligible'][local]),
                'sampling_branch': provenance['sampling_branch'],
                'virtual_unit_normal_ap_dv_ml': normal.tolist(),
                'absolute_unit_normal_ap_dv_ml': np.abs(normal).tolist(),
                'nearest_cardinal_angle_deg': math.degrees(math.acos(float(np.abs(normal).max()))),
                'visible_support_mass': float(visible[local].sum()),
                'source_visible_support_mass': float(provenance['visible_support_mass']),
                'visible_pixel_count_gt_025': int((visible[local] > .25).sum()),
                'valid_pixels': int(valid_masks[local].sum()),
                'sites_available_for_2048': min(2048, int(valid_masks[local].sum())),
                'intact_fit_pixels': len(design),
                'artifacts': artifacts,
                'gauge': gauge,
                'pose_label_stage': v3['pose_gauge'],
                'generator_provenance': provenance,
            }
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        del sample
        if (base + 1) % 8 == 0:
            print(json.dumps({'event': 'progress', 'bases_complete': base + 1,
                              'proposals_complete': len(rows)}), flush=True)

assert len(rows) == BASES * DRAWS_PER_BASE
assert len({row['physical_section_id'] for row in rows}) == len(rows)
assert all(row['split'] == 'train' for row in rows)
assert all(sum(row['base_index'] == base for row in rows) == DRAWS_PER_BASE
           for base in range(BASES))
accepted = [row for row in rows if row['eligible']]
accepted_gauge = [row for row in accepted if row['gauge'] is not None]
summary = {
    'proposals': len(rows),
    'accepted': len(accepted),
    'acceptance_rate': len(accepted) / len(rows),
    'unique_physical_sections': len({row['physical_section_id'] for row in rows}),
    'proposal_coverage': cohort(rows),
    'accepted_coverage': cohort(accepted),
    'per_base_acceptance': {str(base): sum(
        row['eligible'] for row in rows if row['base_index'] == base)
        for base in range(BASES)},
    'accepted_gauge_fit_sections': len(accepted_gauge),
    'accepted_gauge': {key: quantiles([row['gauge'][key] for row in accepted_gauge]) for key in (
        'saved_residual_mean_mm',
        'robust_fit_residual_mean_mm',
        'saved_to_fit_plane_separation_mean_mm',
        'saved_to_fit_center_separation_mm',
        'saved_to_fit_unsigned_normal_angle_deg',
        'fit_condition_number',
        'total_target_displacement_rms_mm',
        'affine_component_rms_mm',
        'representable_local_residual_rms_mm',
        'affine_component_fraction_squared_error',
        'ols_orthogonality_relative_error',
    )},
    'pose_label_stage': {
        'classification': 'pre-artifact source affine gauge; not post-artifact refit',
        'all_generator_records_agree': len({row['pose_label_stage'] for row in rows}) == 1,
        'generator_statement': rows[0]['pose_label_stage'],
    },
    'interpretation': 'descriptive TRAIN-only generator/supervision audit; no model qualification or biological accuracy claim',
}
(RUN / 'summary.json').write_text(
    json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
completed = {
    'config_sha256': digest(RUN / 'config.json'),
    'generator_provenance_sha256': digest(RUN / 'generator_provenance.json'),
    'rows_sha256': digest(RUN / 'rows.jsonl'),
    'summary_sha256': digest(RUN / 'summary.json'),
    'protocol_sha256': config['protocol_sha256'],
    'source_sha256': config['source_sha256'],
    'proposals': len(rows),
    'accepted': len(accepted),
    'development_used': False,
    'calibration_used': False,
    'final_animals_used': False,
    'real_data_used': False,
    'public_benchmark_used': False,
    'model_loaded_or_updated': False,
}
(RUN / 'completed.json').write_text(
    json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'run': str(RUN),
                  'proposals': len(rows), 'accepted': len(accepted),
                  'summary_sha256': completed['summary_sha256']}), flush=True)
