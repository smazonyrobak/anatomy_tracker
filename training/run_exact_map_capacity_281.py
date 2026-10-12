"""TRAIN-only exact-plane coordinate-oracle capacity check; not a predictor."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_ribbon_v6 import (
    project_surface_affine_out, ribbon_derivative_frobenius_bound,
)
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.train_blind_fit_score_280 import mapped_sites, sites_from_sample


SEED, SIDE, GRID, SITES, UPDATES = 20261012281, 192, 64, 512, 300
STEPS = (0, 100, 200, 300)
PARAMETERIZATIONS = {'free64': 64, 'bilinear16': 16}
LIMIT = .35
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/EXACT_MAP_CAPACITY_281_PROTOCOL_20261012.md'
RUN = ROOT / 'runs/exact_map_capacity_281_001'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rigid_surface(state, reflection):
    center, frame, basis = full_frame_state_to_components(state)
    y = (torch.arange(GRID, device=state.device, dtype=state.dtype) + .5) / GRID - .5 / SIDE
    x = (torch.arange(GRID, device=state.device, dtype=state.dtype) + .5) / GRID - .5 / SIDE
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((sx.expand(len(state), -1, -1), yy.expand(len(state), -1, -1)), -1)
    surface = normalized_raster_to_ccf(
        center[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    return surface, frame, basis


def constrained_field(logits, reflection, basis, offsets):
    requested = 1000 * logits.tanh()
    if logits.shape[-1] == 16:
        requested = F.interpolate(requested, (GRID, GRID), mode='bilinear', align_corners=False)
    local, removed_affine = project_surface_affine_out(requested)
    canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
    before = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    scale = (LIMIT / before.clamp_min(LIMIT)).clamp_max(1.)
    return requested, local * scale[:, None, None, None], removed_affine, before, scale


def site_error(local, rigid, frame, indices, target):
    surface = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
    points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': surface[:, None]}, indices)[0, 0]
    return (points - target).norm(dim=-1).mean() / 1000, points


def evaluate(logits, reflection, basis, offsets, rigid, frame, indices, target, rigid_points):
    requested, local, removed_affine, before, scale = constrained_field(
        logits, reflection, basis, offsets)
    error, mapped_points = site_error(local, rigid, frame, indices, target)
    canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
    after = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    shift = (mapped_points - rigid_points).norm(dim=-1) / 1000
    row = {
        'fitted_error_mm': float(error),
        'prelimit_derivative_bound': float(before[0]),
        'deformation_rescale': float(scale[0]),
        'postlimit_derivative_bound': float(after[0]),
        'prelimit_times_rescale': float((before * scale)[0]),
        'removed_affine_local_um_3_by_intercept_s_t': removed_affine[0].tolist(),
        'requested_displacement_mean_mm': float(requested.norm(dim=1).mean() / 1000),
        'requested_displacement_rms_mm': float(requested.square().sum(1).mean().sqrt() / 1000),
        'field_displacement_mean_mm': float(local.norm(dim=1).mean() / 1000),
        'field_displacement_rms_mm': float(local.square().sum(1).mean().sqrt() / 1000),
        'surviving_site_displacement_mean_mm': float(shift.mean()),
        'surviving_site_displacement_rms_mm': float(shift.square().mean().sqrt()),
        'logit_absmax': float(logits.abs().amax()),
    }
    row['feasible'] = bool(np.isfinite(row['fitted_error_mm'])
        and np.isfinite(row['postlimit_derivative_bound'])
        and row['postlimit_derivative_bound'] <= LIMIT + 1e-5)
    return row


def subgroup(rows, parameterization):
    if not rows:
        return {'n': 0, 'no_warp_mean_mm': None, 'fitted_mean_mm': None, 'gain_mean_mm': None}
    return {'n': len(rows),
        'no_warp_mean_mm': float(np.mean([row['same_grid_no_warp_mm'] for row in rows])),
        'fitted_mean_mm': float(np.mean([
            row['oracles'][parameterization]['fitted_error_mm'] for row in rows])),
        'gain_mean_mm': float(np.mean([
            row['oracles'][parameterization]['gain_mm'] for row in rows]))}


def main():
    assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    context = load_streaming_synthetic_v7_64(device='cuda')
    synthetic_provenance = context['provenance']
    source_names = (
        'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py',
        'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_joint_model_v7.py',
        'arbitrary_plane_recurrent_model.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_one_shot_slide_artifacts_v7.py',
        'arbitrary_plane_one_shot_slide_artifacts_v6.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'arbitrary_plane_allen_atlas_binding_v6.py',
        'arbitrary_plane_geometry.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
    )
    source_hashes = {name: digest(SOURCE / name) for name in source_names}
    RUN.mkdir(parents=True, exist_ok=False)
    panel, attempted = [], 0
    rng = np.random.default_rng(SEED + 1)
    with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
        for base in range(56, 64):
            for slot in range(4):
                while True:
                    attempted += 1
                    subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                    draw_seed = SEED + 3_000_000_000 + attempted
                    with torch.no_grad():
                        sample = sample_one_shot_slide_artifacts_v9(
                            context, [subject_index], draw_seed, SIDE)
                    used = bool(sample['eligible'][0])
                    provenance = sample['provenance'][0]
                    draws.write(json.dumps({'base_index': base, 'slot': slot,
                        'attempt': attempted, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'used': used,
                        'provenance': provenance}, allow_nan=False) + '\n')
                    if used:
                        break
                    del sample
                assert provenance['split'] == 'train'
                assert provenance['base_lineage']['animal_index'] == base
                indices, chart, target = sites_from_sample(sample, SITES)
                normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
                artifacts = provenance['one_shot_slide_artifacts_v3']
                panel.append({'base_index': base, 'slot': slot,
                    'subject_index': subject_index, 'draw_seed': draw_seed,
                    'physical_section_id': provenance['physical_section_id'],
                    'state': sample['state'].cpu(),
                    'reflection': sample['reflection'].cpu(),
                    'offsets': sample['offsets'].cpu(),
                    'site_indices': indices.cpu(), 'site_chart_st': chart.cpu(),
                    'site_target_ccf_ap_dv_ml_um': target.cpu(),
                    'valid_fraction': float(sample['valid_mask'][0].float().mean()),
                    'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
                    'event_flags': {name: bool(artifacts['events'][name]) for name in
                        ('fragment', 'fold', 'bubble', 'tile_seam')},
                    'tissue_placed': artifacts['tissue_placed'],
                    'artifact_pixel_counts': artifacts['pixel_counts'],
                    'provenance': provenance})
                del sample
    assert len(panel) == 32 and len({row['physical_section_id'] for row in panel}) == 32
    torch.save(panel, RUN / 'inner_panel.pt')
    del context
    torch.cuda.empty_cache()
    config = {'candidate': 281, 'protocol_sha256': digest(PROTOCOL),
        'source_sha256': source_hashes, 'seed': SEED,
        'synthetic_provenance': synthetic_provenance,
        'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
        'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
        'attempted_draws': attempted, 'eligible_sections': len(panel),
        'train_base_indices': list(range(56, 64)), 'sections_per_base': 4,
        'source_shape': [SIDE, SIDE], 'map_grid_shape': [GRID, GRID],
        'surviving_sites_per_section': SITES, 'readout_steps': STEPS,
        'parameterizations': PARAMETERIZATIONS,
        'requested_field_um': '1000*tanh(logits); 16x16 requested controls bilinearly upsampled to 64x64 with align_corners=False',
        'optimizer': 'Adam, lr=0.1, 300 updates, zero initialization, logits only',
        'objective': 'mean Euclidean surviving-site CCF error in mm; no magnitude penalty',
        'map_derivative_limit': LIMIT,
        'geometry': '280 exact-plane 64-grid / 192-source no-warp, full-canvas affine projection, reflected canonical derivative bound, global rescale, bilinear site sampling',
        'best_step_rule': 'minimum exact fitted site error among feasible readouts at 0,100,200,300',
        'affine_diagnostic': 'independent least-squares CCF 3x[1,s,t] correction on identical sites; excluded from constrained local-map gain',
        'coordinate_oracle_only': True, 'atlas_used_only_for_synthetic_generation': True,
        'prior_model_or_weights_used': False, 'dev_cal_final_or_public_benchmark_used': False,
        'torch': torch.__version__, 'numpy': np.__version__}
    (RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')
    sections, best_fields = [], {}
    with (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
         (RUN / 'evaluations.jsonl').open('w', encoding='utf8') as evaluations, \
         (RUN / 'sections.jsonl').open('w', encoding='utf8') as output:
        for item in panel:
            section_id = item['physical_section_id']
            state = item['state'].cuda()
            reflection = item['reflection'].cuda()
            offsets = item['offsets'].cuda()
            indices = item['site_indices'].cuda()
            target = item['site_target_ccf_ap_dv_ml_um'].cuda()
            rigid, frame, basis = rigid_surface(state, reflection)
            rigid_points = mapped_sites(
                {'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
            no_warp = float((rigid_points - target).norm(dim=-1).mean() / 1000)
            design = torch.cat((torch.ones_like(item['site_chart_st'][:, :1]),
                item['site_chart_st']), -1).double()
            correction = torch.linalg.lstsq(design,
                (target - rigid_points).cpu().double()).solution
            affine_points = rigid_points + (design @ correction).to(rigid_points)
            affine_residual = (affine_points - target).norm(dim=-1) / 1000
            affine_error = float(affine_residual.mean())
            row = {'base_index': item['base_index'], 'slot': item['slot'],
                'subject_index': item['subject_index'], 'draw_seed': item['draw_seed'],
                'physical_section_id': section_id,
                'donor_animal_id': item['provenance']['base_lineage']['animal_id'],
                'base_lineage': item['provenance']['base_lineage'],
                'virtual_subject_id': item['provenance']['virtual_subject_id'],
                'virtual_index': item['provenance']['virtual_index'],
                'true_state': item['state'][0].tolist(),
                'true_reflection': int(item['reflection'][0]),
                'site_count': len(indices), 'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'sparse_support_under_15pct_visible': item['visible_fraction'] < .15,
                'obliquity_deg': item['obliquity_deg'],
                'event_flags': item['event_flags'],
                'tissue_placed': item['tissue_placed'],
                'artifact_pixel_counts': item['artifact_pixel_counts'],
                'same_grid_no_warp_mm': no_warp,
                'best_fit_affine_ccf_error_mm_diagnostic_only': affine_error,
                'best_fit_affine_ccf_residual_rms_mm_diagnostic_only': float(
                    affine_residual.square().mean().sqrt()),
                'best_fit_affine_ccf_gain_mm_diagnostic_only': no_warp - affine_error,
                'best_fit_affine_ccf_um_3_by_intercept_s_t_diagnostic_only': correction.T.tolist(),
                'oracles': {}, 'provenance': item['provenance']}
            for name, control_side in PARAMETERIZATIONS.items():
                logits = torch.nn.Parameter(torch.zeros(
                    (1, 3, control_side, control_side), device='cuda'))
                optimizer = torch.optim.Adam([logits], lr=.1)
                best, best_logits = None, None
                for step in range(UPDATES + 1):
                    if step in STEPS:
                        with torch.no_grad():
                            measured = evaluate(logits, reflection, basis, offsets, rigid,
                                frame, indices, target, rigid_points)
                        measured.update({'physical_section_id': section_id,
                            'base_index': item['base_index'], 'slot': item['slot'],
                            'parameterization': name, 'step': step,
                            'same_grid_no_warp_mm': no_warp,
                            'gain_mm': no_warp - measured['fitted_error_mm']})
                        evaluations.write(json.dumps(measured, allow_nan=False) + '\n')
                        if measured['feasible'] and (best is None or
                                measured['fitted_error_mm'] < best['fitted_error_mm']):
                            best = measured.copy()
                            best_logits = logits.detach().cpu().clone()
                    if step == UPDATES:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    _, local, _, _, _ = constrained_field(logits, reflection, basis, offsets)
                    loss, _ = site_error(local, rigid, frame, indices, target)
                    loss.backward()
                    objective, gradient = float(loss.detach()), float(logits.grad.norm())
                    optimizer.step()
                    training.write(json.dumps({'physical_section_id': section_id,
                        'parameterization': name, 'step': step + 1,
                        'objective_before_update_mm': objective,
                        'logit_gradient_norm': gradient,
                        'post_update_logit_absmax': float(logits.detach().abs().amax())},
                        allow_nan=False) + '\n')
                assert best is not None
                row['oracles'][name] = best
                best_fields[(section_id, name)] = {
                    'best_step': best['step'], 'logits': best_logits}
                print(json.dumps({'event': 'section_oracle', 'physical_section_id': section_id,
                    'parameterization': name, 'best_step': best['step'],
                    'gain_mm': best['gain_mm']}), flush=True)
            output.write(json.dumps(row, allow_nan=False) + '\n')
            sections.append(row)
            training.flush()
            evaluations.flush()
            output.flush()
    torch.save(best_fields, RUN / 'best_fields.pt')
    summary = {}
    for name in PARAMETERIZATIONS:
        values = [row['oracles'][name] for row in sections]
        overall = subgroup(sections, name)
        summary[name] = {**overall,
            'gain_at_least_0_05mm': overall['gain_mean_mm'] >= .05,
            'best_step_counts': {str(step): sum(v['step'] == step for v in values)
                for step in STEPS},
            'max_postlimit_derivative_bound': max(v['postlimit_derivative_bound'] for v in values),
            'min_deformation_rescale': min(v['deformation_rescale'] for v in values),
            'by_base': {str(base): subgroup(
                [row for row in sections if row['base_index'] == base], name)
                for base in range(56, 64)},
            'by_event': {event: {
                str(flag).lower(): subgroup([row for row in sections
                    if row['event_flags'][event] == flag], name)
                for flag in (False, True)}
                for event in ('fragment', 'fold', 'bubble', 'tile_seam')},
            'by_sparse_support': {str(flag).lower(): subgroup([row for row in sections
                if row['sparse_support_under_15pct_visible'] == flag], name)
                for flag in (False, True)}}
    output_names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
        'training.jsonl', 'evaluations.jsonl', 'sections.jsonl', 'best_fields.pt')
    completed = {'candidate': 281, 'eligible_sections': len(sections),
        'attempted_draws': attempted, 'summary': summary,
        'best_fit_affine_ccf_gain_mean_mm_diagnostic_only': float(np.mean([
            row['best_fit_affine_ccf_gain_mm_diagnostic_only'] for row in sections])),
        'best_fit_affine_ccf_error_mean_mm_diagnostic_only': float(np.mean([
            row['best_fit_affine_ccf_error_mm_diagnostic_only'] for row in sections])),
        'best_fit_affine_ccf_residual_rms_mean_mm_diagnostic_only': float(np.mean([
            row['best_fit_affine_ccf_residual_rms_mm_diagnostic_only'] for row in sections])),
        'either_local_parameterization_gain_at_least_0_05mm': any(
            value['gain_at_least_0_05mm'] for value in summary.values()),
        'interpretation': 'coordinate representability signal on TRAIN-only synthetic panel'
            if any(value['gain_at_least_0_05mm'] for value in summary.values())
            else 'finite-step coordinate-oracle result inconclusive about impossibility',
        'result_role': 'TRAIN-only bounded coordinate-oracle diagnostic; not a deployable predictor',
        'protocol_sha256': config['protocol_sha256'], 'source_sha256': source_hashes,
        'output_sha256': {name: digest(RUN / name) for name in output_names},
        'elapsed_seconds': time.perf_counter() - started,
        'dev_cal_final_or_public_benchmark_used': False}
    (RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'summary': summary,
        'elapsed_seconds': completed['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
