"""Read-only audit of 281. Run only after run_exact_map_capacity_281.py exits."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/exact_map_capacity_281_001'
PROTOCOL = SOURCE.parent / 'docs/publication/EXACT_MAP_CAPACITY_281_PROTOCOL_20261012.md'
COMMIT = '34c095c49dbfaba6786e0169503dfa94c7d0e593'
RUNNER_SHA256 = 'be5b52b806713b414a31e435a440accaf46bacead5910b1fc8615088438c723e'
PROTOCOL_SHA256 = '552ebd0b90a1aba72efd47bfffea361dea063a18dee0c4d633d620f14409a76c'
SEED, SIDE, GRID, SITES, VARIANTS, UPDATES, LIMIT = 20261012281, 192, 64, 512, 64, 300, .35
STEPS = (0, 100, 200, 300)
ORACLES = {'free64': 64, 'bilinear16': 16}
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE_NAMES = (
    'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'global_atlas_contrast_090.py',
)
OUTPUT_NAMES = (
    'config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'training.jsonl',
    'evaluations.jsonl', 'sections.jsonl', 'best_fields.pt',
)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def committed_sha256(path):
    return hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{COMMIT}:{path}'], cwd=SOURCE.parent)).hexdigest()


def jsonl(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def close(actual, expected, atol=1e-7):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=atol)


def identity_check(identity, subject_index, draw_seed, used, synthetic, stage_hashes):
    base = subject_index // VARIANTS
    subject = synthetic['virtual_subjects'][subject_index]
    assert identity['seed_prefix'] == [draw_seed, subject_index, 0]
    assert identity['physical_section_id'] == f'stream-v7-seed-{draw_seed}-row-0-virtual-{subject_index}'
    assert identity['virtual_index'] == subject_index
    assert identity['virtual_subject_id'] == subject['virtual_subject_id']
    assert identity['base_lineage'] == synthetic['base_subjects'][base]['lineage']
    assert identity['split'] == subject['base_lineage']['split'] == 'train'
    assert identity['base_lineage']['animal_index'] == base
    assert identity['shape_h_w'] == [SIDE, SIDE]
    assert identity['eligible'] == (identity['visible_support_mass'] >= SIDE**2 / 144)
    assert used == (identity['eligible'] and
                    identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] >= SIDE**2 / 144)
    for version in (3, 4, 6, 7, 9):
        assert identity[f'one_shot_slide_artifacts_v{version}']['code_sha256'] == stage_hashes[version]
    assert identity['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair'] is False
    assert identity['one_shot_slide_artifacts_v9']['seed_branch'] == 34
    assert identity['one_shot_slide_artifacts_v9']['sampling'] == 'one independent physical plane per row; no appearance pairing'


def site_points(surface, indices):
    xy = torch.stack((indices.remainder(SIDE), indices.div(SIDE, rounding_mode='floor')), -1).float()
    grid = ((xy + .5) * (2 / SIDE) - 1)[None, None]
    return F.grid_sample(surface.permute(0, 3, 1, 2), grid, mode='bilinear',
                         padding_mode='border', align_corners=False)[0, :, 0].T


def rigid_geometry(item, device):
    state = item['state'].to(device)
    reflection = item['reflection'].to(device)
    centre, frame, basis = full_frame_state_to_components(state)
    y = (torch.arange(GRID, device=device, dtype=state.dtype) + .5) / GRID - .5 / SIDE
    x = (torch.arange(GRID, device=device, dtype=state.dtype) + .5) / GRID - .5 / SIDE
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((sx.expand(len(state), -1, -1), yy.expand(len(state), -1, -1)), -1)
    rigid = normalized_raster_to_ccf(
        centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    return rigid, frame, basis


def rerender(logits, item, rigid, frame, basis, rigid_points, device):
    logits = logits.to(device)
    requested = 1000 * logits.tanh()
    if logits.shape[-1] == 16:
        requested = F.interpolate(requested, (GRID, GRID), mode='bilinear', align_corners=False)
    local, removed = project_surface_affine_out(requested)
    canonical = local.flip(-1) if bool(item['reflection'][0]) else local
    offsets = item['offsets'].to(device)
    before = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    scale = (LIMIT / before.clamp_min(LIMIT)).clamp_max(1.)
    local = local * scale[:, None, None, None]
    canonical = local.flip(-1) if bool(item['reflection'][0]) else local
    after = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    surface = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
    points = site_points(surface, item['site_indices'].to(device))
    target = item['site_target_ccf_ap_dv_ml_um'].to(device)
    shift = (points - rigid_points).norm(dim=-1) / 1000
    return {
        'fitted_error_mm': float((points - target).norm(dim=-1).mean() / 1000),
        'prelimit_derivative_bound': float(before[0]),
        'deformation_rescale': float(scale[0]),
        'postlimit_derivative_bound': float(after[0]),
        'prelimit_times_rescale': float((before * scale)[0]),
        'removed_affine_local_um_3_by_intercept_s_t': removed[0].tolist(),
        'requested_displacement_mean_mm': float(requested.norm(dim=1).mean() / 1000),
        'requested_displacement_rms_mm': float(requested.square().sum(1).mean().sqrt() / 1000),
        'field_displacement_mean_mm': float(local.norm(dim=1).mean() / 1000),
        'field_displacement_rms_mm': float(local.square().sum(1).mean().sqrt() / 1000),
        'surviving_site_displacement_mean_mm': float(shift.mean()),
        'surviving_site_displacement_rms_mm': float(shift.square().mean().sqrt()),
        'logit_absmax': float(logits.abs().amax()),
    }


def subgroup(rows, oracle):
    if not rows:
        return {'n': 0, 'no_warp_mean_mm': None, 'fitted_mean_mm': None, 'gain_mean_mm': None}
    return {'n': len(rows),
        'no_warp_mean_mm': float(np.mean([r['same_grid_no_warp_mm'] for r in rows])),
        'fitted_mean_mm': float(np.mean([r['oracles'][oracle]['fitted_error_mm'] for r in rows])),
        'gain_mean_mm': float(np.mean([r['oracles'][oracle]['gain_mm'] for r in rows]))}


def main():
    if sys.flags.optimize:
        raise RuntimeError('Run without -O: assertions are the audit.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), 'Faithful generator replay requires CUDA.'
    device = 'cuda'

    assert sha256(SOURCE / SOURCE_NAMES[0]) == committed_sha256('training/' + SOURCE_NAMES[0]) == RUNNER_SHA256
    assert sha256(PROTOCOL) == committed_sha256('docs/publication/' + PROTOCOL.name) == PROTOCOL_SHA256
    config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
    receipt = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
    assert config['candidate'] == receipt['candidate'] == 281
    assert config['seed'] == SEED and config['attempted_draws'] >= 32
    assert config['eligible_sections'] == receipt['eligible_sections'] == 32
    assert config['train_base_indices'] == list(range(56, 64)) and config['sections_per_base'] == 4
    assert config['source_shape'] == [SIDE, SIDE] and config['map_grid_shape'] == [GRID, GRID]
    assert config['surviving_sites_per_section'] == SITES
    assert config['readout_steps'] == list(STEPS) and config['parameterizations'] == ORACLES
    assert config['map_derivative_limit'] == LIMIT
    assert config['optimizer'] == 'Adam, lr=0.1, 300 updates, zero initialization, logits only'
    assert config['objective'] == 'mean Euclidean surviving-site CCF error in mm; no magnitude penalty'
    assert config['best_step_rule'] == 'minimum exact fitted site error among feasible readouts at 0,100,200,300'
    assert config['coordinate_oracle_only'] is True
    assert config['atlas_used_only_for_synthetic_generation'] is True
    assert config['prior_model_or_weights_used'] is False
    assert config['dev_cal_final_or_public_benchmark_used'] is False
    assert receipt['dev_cal_final_or_public_benchmark_used'] is False
    assert receipt['result_role'] == 'TRAIN-only bounded coordinate-oracle diagnostic; not a deployable predictor'
    assert receipt['protocol_sha256'] == config['protocol_sha256'] == PROTOCOL_SHA256
    assert set(config['source_sha256']) == set(SOURCE_NAMES)
    assert receipt['source_sha256'] == config['source_sha256']
    for name in SOURCE_NAMES:
        expected = committed_sha256('training/' + name)
        assert sha256(SOURCE / name) == config['source_sha256'][name] == expected

    global full_frame_state_to_components, normalized_raster_to_ccf
    global sample_one_shot_slide_artifacts_v9, project_surface_affine_out
    global ribbon_derivative_frobenius_bound, load_streaming_synthetic_v7_64
    from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
    from training.arbitrary_plane_geometry import normalized_raster_to_ccf
    from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
    from training.arbitrary_plane_ribbon_v6 import (
        project_surface_affine_out, ribbon_derivative_frobenius_bound,
    )
    from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

    synthetic = config['synthetic_provenance']
    assert synthetic['new_independent_train_bases'] == 56 and synthetic['all_train_bases'] == 64
    assert synthetic['calibrated'] is False
    assert len(synthetic['base_subjects']) == 64
    assert len(synthetic['virtual_subjects']) == 64 * VARIANTS
    for name, expected in synthetic['source_sha256'].items():
        assert Path(name).name == name
        assert sha256(SOURCE / name) == expected == committed_sha256('training/' + name)
    for path, expected in synthetic['input_sha256'].items():
        bound = Path(path).resolve()
        assert bound.is_relative_to(ROOT / 'data')
        assert sha256(bound) == expected
    for key in ('template', 'annotation'):
        bound = Path(synthetic['atlas_binding'][key + '_path']).resolve()
        assert bound.is_relative_to(ROOT / 'data')
        assert sha256(bound) == synthetic['atlas_binding'][key + '_sha256']
    for index, subject in enumerate(synthetic['virtual_subjects']):
        assert subject['virtual_index'] == index
        assert (subject['base_index'], subject['variant']) == divmod(index, VARIANTS)
        assert subject['base_lineage'] == synthetic['base_subjects'][index // VARIANTS]['lineage']
        assert subject['base_lineage']['split'] == 'train'

    assert config['inner_draws_sha256'] == sha256(RUN / 'inner_draws.jsonl')
    assert config['inner_panel_sha256'] == sha256(RUN / 'inner_panel.pt')
    assert set(receipt['output_sha256']) == set(OUTPUT_NAMES)
    for name in OUTPUT_NAMES:
        assert sha256(RUN / name) == receipt['output_sha256'][name]
    draws = jsonl(RUN / 'inner_draws.jsonl')
    sections = jsonl(RUN / 'sections.jsonl')
    evaluations = jsonl(RUN / 'evaluations.jsonl')
    training = jsonl(RUN / 'training.jsonl')
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype,
                                           np.dtypes.Float64DType]):
        panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=True)
        best_fields = torch.load(RUN / 'best_fields.pt', map_location='cpu', weights_only=True)
    assert len(draws) == config['attempted_draws'] == receipt['attempted_draws']
    assert len(panel) == len(sections) == 32
    assert len(evaluations) == 32 * len(ORACLES) * len(STEPS)
    assert len(training) == 32 * len(ORACLES) * UPDATES
    assert len(best_fields) == 32 * len(ORACLES)

    stage_hashes = {version: config['source_sha256'][f'arbitrary_plane_one_shot_slide_artifacts_v{version}.py']
                    for version in (3, 4, 6, 7, 9)}
    rng = np.random.default_rng(SEED + 1)
    accepted, seen = [], set()
    for attempt, draw in enumerate(draws, 1):
        assert len(accepted) < 32
        base, slot = 56 + len(accepted) // 4, len(accepted) % 4
        subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
        draw_seed = SEED + 3_000_000_000 + attempt
        assert (draw['base_index'], draw['slot'], draw['attempt'],
                draw['subject_index'], draw['draw_seed']) == (
                base, slot, attempt, subject_index, draw_seed)
        assert type(draw['used']) is bool
        identity_check(draw['provenance'], subject_index, draw_seed,
                       draw['used'], synthetic, stage_hashes)
        section_id = draw['provenance']['physical_section_id']
        assert section_id not in seen
        seen.add(section_id)
        if draw['used']:
            accepted.append(draw)
    assert len(accepted) == 32
    assert {d['provenance']['physical_section_id'] for d in accepted} == {
        item['physical_section_id'] for item in panel}
    assert set(best_fields) == {(item['physical_section_id'], name)
                                for item in panel for name in ORACLES}

    train_pos = eval_pos = 0
    for item, draw, row in zip(panel, accepted, sections):
        section_id = item['physical_section_id']
        identity = draw['provenance']
        assert (item['base_index'], item['slot'], item['subject_index'], item['draw_seed']) == (
            draw['base_index'], draw['slot'], draw['subject_index'], draw['draw_seed'])
        assert section_id == row['physical_section_id'] == identity['physical_section_id']
        assert json.loads(json.dumps(item['provenance'])) == row['provenance'] == identity
        assert item['state'].shape == (1, 12) and item['reflection'].shape == (1,)
        assert item['offsets'].shape == (1, 9)
        indices = item['site_indices']
        valid_count = identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid']
        assert indices.ndim == 1 and len(indices) == min(SITES, valid_count) == row['site_count']
        assert bool(((indices >= 0) & (indices < SIDE**2)).all())
        assert bool((indices[1:] > indices[:-1]).all())
        assert item['site_chart_st'].shape == (len(indices), 2)
        assert item['site_target_ccf_ap_dv_ml_um'].shape == (len(indices), 3)
        assert bool(torch.isfinite(item['site_target_ccf_ap_dv_ml_um']).all())
        chart = torch.stack((indices.remainder(SIDE), indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        close(item['site_chart_st'], chart, 1e-7)
        close(item['valid_fraction'], valid_count / SIDE**2, 5e-8)
        assert 0 <= item['visible_fraction'] <= 1 and 0 <= item['obliquity_deg'] <= 90
        assert item['event_flags'] == {event: bool(identity['one_shot_slide_artifacts_v3']['events'][event])
                                       for event in EVENTS}
        assert item['artifact_pixel_counts'] == identity['one_shot_slide_artifacts_v3']['pixel_counts']
        assert item['tissue_placed'] == identity['one_shot_slide_artifacts_v3']['tissue_placed']
        assert (row['base_index'], row['slot'], row['subject_index'], row['draw_seed']) == (
            item['base_index'], item['slot'], item['subject_index'], item['draw_seed'])
        assert row['donor_animal_id'] == identity['base_lineage']['animal_id']
        assert row['base_lineage'] == identity['base_lineage']
        assert row['virtual_subject_id'] == identity['virtual_subject_id']
        assert row['virtual_index'] == identity['virtual_index']
        assert row['true_state'] == item['state'][0].tolist()
        assert row['true_reflection'] == int(item['reflection'][0])
        assert row['event_flags'] == item['event_flags']
        assert row['artifact_pixel_counts'] == item['artifact_pixel_counts']
        assert row['tissue_placed'] == item['tissue_placed']
        close(row['valid_fraction'], item['valid_fraction'])
        close(row['visible_fraction'], item['visible_fraction'])
        close(row['obliquity_deg'], item['obliquity_deg'])
        assert row['sparse_support_under_15pct_visible'] == (item['visible_fraction'] < .15)
        assert set(row['oracles']) == set(ORACLES)

        with torch.no_grad():
            rigid, frame, basis = rigid_geometry(item, device)
            rigid_points = site_points(rigid, indices.to(device))
            target = item['site_target_ccf_ap_dv_ml_um'].to(device)
            no_warp = float((rigid_points - target).norm(dim=-1).mean() / 1000)
            design = torch.cat((torch.ones_like(chart[:, :1]), chart), -1).double()
            correction = torch.linalg.lstsq(design, (target - rigid_points).cpu().double()).solution
            affine_points = rigid_points + (design @ correction).to(rigid_points)
            affine_residual = (affine_points - target).norm(dim=-1) / 1000
            affine_error = float(affine_residual.mean())
            close(row['same_grid_no_warp_mm'], no_warp, 1e-5)
            close(row['best_fit_affine_ccf_error_mm_diagnostic_only'], affine_error, 1e-5)
            close(row['best_fit_affine_ccf_residual_rms_mm_diagnostic_only'],
                  float(affine_residual.square().mean().sqrt()), 1e-5)
            close(row['best_fit_affine_ccf_gain_mm_diagnostic_only'], no_warp - affine_error, 1e-5)
            close(row['best_fit_affine_ccf_um_3_by_intercept_s_t_diagnostic_only'],
                  correction.T.tolist(), .1)

            for name, control_side in ORACLES.items():
                trace = training[train_pos:train_pos + UPDATES]
                readouts = evaluations[eval_pos:eval_pos + len(STEPS)]
                train_pos += UPDATES
                eval_pos += len(STEPS)
                assert [(t['physical_section_id'], t['parameterization'], t['step']) for t in trace] == [
                    (section_id, name, step) for step in range(1, UPDATES + 1)]
                assert [(e['physical_section_id'], e['parameterization'], e['step']) for e in readouts] == [
                    (section_id, name, step) for step in STEPS]
                for t in trace:
                    assert all(math.isfinite(t[key]) for key in (
                        'objective_before_update_mm', 'logit_gradient_norm', 'post_update_logit_absmax'))
                    assert t['objective_before_update_mm'] >= 0 and t['logit_gradient_norm'] >= 0
                    assert t['post_update_logit_absmax'] >= 0
                assert trace[0]['logit_gradient_norm'] > 0
                for readout in readouts:
                    assert all(math.isfinite(readout[key]) for key in (
                        'fitted_error_mm', 'prelimit_derivative_bound', 'deformation_rescale',
                        'postlimit_derivative_bound', 'prelimit_times_rescale', 'logit_absmax'))
                    assert readout['base_index'] == item['base_index'] and readout['slot'] == item['slot']
                    close(readout['same_grid_no_warp_mm'], row['same_grid_no_warp_mm'])
                    close(readout['gain_mm'], no_warp - readout['fitted_error_mm'], 1e-5)
                    close(readout['prelimit_times_rescale'],
                          readout['prelimit_derivative_bound'] * readout['deformation_rescale'], 1e-5)
                    assert 0 < readout['deformation_rescale'] <= 1
                    assert readout['feasible'] == (
                        math.isfinite(readout['fitted_error_mm']) and
                        math.isfinite(readout['postlimit_derivative_bound']) and
                        readout['postlimit_derivative_bound'] <= LIMIT + 1e-5)
                    assert readout['postlimit_derivative_bound'] <= LIMIT + 1e-5
                initial = readouts[0]
                close(initial['fitted_error_mm'], no_warp, 1e-5)
                close(initial['logit_absmax'], 0)
                close(initial['prelimit_derivative_bound'], 0)
                close(initial['postlimit_derivative_bound'], 0)
                close(initial['deformation_rescale'], 1)
                close(initial['field_displacement_rms_mm'], 0)
                for step in STEPS[:-1]:
                    close(trace[step]['objective_before_update_mm'],
                          readouts[STEPS.index(step)]['fitted_error_mm'], 1e-5)
                for step in STEPS[1:]:
                    close(trace[step - 1]['post_update_logit_absmax'],
                          readouts[STEPS.index(step)]['logit_absmax'], 1e-5)
                feasible = [e for e in readouts if e['feasible']]
                assert feasible
                best = min(feasible, key=lambda e: e['fitted_error_mm'])
                assert row['oracles'][name] == best
                checkpoint = best_fields[(section_id, name)]
                assert checkpoint['best_step'] == best['step']
                logits = checkpoint['logits']
                assert logits.shape == (1, 3, control_side, control_side)
                assert bool(torch.isfinite(logits).all())
                close(float(logits.abs().amax()), best['logit_absmax'], 1e-5)
                rendered = rerender(logits, item, rigid, frame, basis, rigid_points, device)
                for key, value in rendered.items():
                    close(best[key], value, .1 if key == 'removed_affine_local_um_3_by_intercept_s_t' else 1e-4)
                close(best['gain_mm'], no_warp - rendered['fitted_error_mm'], 1e-4)
                assert rendered['postlimit_derivative_bound'] <= LIMIT + 1e-5
    assert train_pos == len(training) and eval_pos == len(evaluations)

    summary = {}
    for name in ORACLES:
        values = [row['oracles'][name] for row in sections]
        overall = subgroup(sections, name)
        summary[name] = {**overall,
            'gain_at_least_0_05mm': overall['gain_mean_mm'] >= .05,
            'best_step_counts': {str(step): sum(v['step'] == step for v in values) for step in STEPS},
            'max_postlimit_derivative_bound': max(v['postlimit_derivative_bound'] for v in values),
            'min_deformation_rescale': min(v['deformation_rescale'] for v in values),
            'by_base': {str(base): subgroup([r for r in sections if r['base_index'] == base], name)
                        for base in range(56, 64)},
            'by_event': {event: {str(flag).lower(): subgroup(
                [r for r in sections if r['event_flags'][event] == flag], name)
                for flag in (False, True)} for event in EVENTS},
            'by_sparse_support': {str(flag).lower(): subgroup(
                [r for r in sections if r['sparse_support_under_15pct_visible'] == flag], name)
                for flag in (False, True)}}
    assert receipt['summary'] == summary
    close(receipt['best_fit_affine_ccf_gain_mean_mm_diagnostic_only'], np.mean([
        r['best_fit_affine_ccf_gain_mm_diagnostic_only'] for r in sections]))
    close(receipt['best_fit_affine_ccf_error_mean_mm_diagnostic_only'], np.mean([
        r['best_fit_affine_ccf_error_mm_diagnostic_only'] for r in sections]))
    close(receipt['best_fit_affine_ccf_residual_rms_mean_mm_diagnostic_only'], np.mean([
        r['best_fit_affine_ccf_residual_rms_mm_diagnostic_only'] for r in sections]))
    gate = any(value['gain_at_least_0_05mm'] for value in summary.values())
    assert receipt['either_local_parameterization_gain_at_least_0_05mm'] == gate
    assert receipt['interpretation'] == ('coordinate representability signal on TRAIN-only synthetic panel'
        if gate else 'finite-step coordinate-oracle result inconclusive about impossibility')
    assert math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0

    # Replay every attempted physical draw, including rejected ones, from frozen TRAIN inputs.
    context = load_streaming_synthetic_v7_64(device=device)
    assert context['provenance'] == synthetic
    used_index = 0
    for draw in draws:
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(
                context, [draw['subject_index']], draw['draw_seed'], SIDE)
        assert bool(sample['eligible'][0]) == draw['used']
        assert json.loads(json.dumps(sample['provenance'][0])) == draw['provenance']
        if not draw['used']:
            continue
        item = panel[used_index]
        used_index += 1
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        expected_indices = valid[torch.linspace(0, len(valid) - 1, min(SITES, len(valid)),
                                                device=valid.device).long()]
        assert torch.equal(item['site_indices'], expected_indices.cpu())
        close(item['site_target_ccf_ap_dv_ml_um'],
              sample['centre'][0].reshape(-1, 3)[expected_indices].cpu(), .1)
        close(item['state'], sample['state'].cpu(), .01)
        assert torch.equal(item['reflection'], sample['reflection'].cpu())
        close(item['offsets'], sample['offsets'].cpu(), .01)
        close(item['valid_fraction'], float(sample['valid_mask'][0].float().mean()), 5e-8)
        close(item['visible_fraction'], float((sample['visible'][0] > .25).float().mean()), 5e-8)
        normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
        obliquity = float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1))))
        close(item['obliquity_deg'], obliquity, 1e-4)
    assert used_index == 32
    print(json.dumps({'audit': 'pass', 'candidate': 281, 'runner_commit': COMMIT,
        'attempted_draws': len(draws), 'eligible_sections': len(panel),
        'oracle_gain_mean_mm': {name: summary[name]['gain_mean_mm'] for name in ORACLES},
        'either_local_parameterization_gain_at_least_0_05mm': gate,
        'affine_diagnostic_gain_mean_mm': receipt['best_fit_affine_ccf_gain_mean_mm_diagnostic_only']},
        allow_nan=False))


if __name__ == '__main__':
    main()
