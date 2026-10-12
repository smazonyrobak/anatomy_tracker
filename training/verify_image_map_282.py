"""Read-only public audit of 282. Run only after train_image_map_282.py exits."""
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

ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
RUN = ROOT / 'runs/image_map_282_002'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
PROTOCOL = SOURCE.parent / 'docs/publication/IMAGE_MAP_282_PROTOCOL_20261012.md'
COMMIT = '237b435d6fc11d9855ee644965e913112649f84c'
RUNNER_SHA256 = '80c189e40f7be1c1bbea687c73ba904fad1103ab8f2c05201cbc15ff07977e85'
PROTOCOL_SHA256 = '5c1acc4099d87a534af1ede305ac2eaea56a8107326931ee1cb799d979fa222a'
SEED, SIDE, GRID, UPDATES, VARIANTS = 20261012282, 192, 64, 1500, 64
TRAIN_SITES, READOUT_SITES, LIMIT = 128, 512, .35
STEPS = (0, 500, 1500)
ARMS = {'smooth16_full': (16, 'full'), 'free64_full': (64, 'full'),
        'smooth16_support': (16, 'support'), 'smooth16_zero': (16, 'zero')}
POSE_MODULES = ('encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global')
POSE_BUFFERS = ('normal_anchor_frames', 'center_origin', 'center_scale',
                'euclidean_scale', 'rotation_grid')
MAP_MODULES = ('warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair')
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE_NAMES = (
    'train_image_map_282.py', 'run_exact_map_capacity_281.py',
    'train_blind_fit_score_280.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
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
    'arbitrary_plane_geometry.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
)
OUTPUT_NAMES = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
    'training_draws.jsonl', 'training_schedule.jsonl', 'training.jsonl', 'readouts.jsonl')


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


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def mapper_digest(model):
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.split('.')[0] in MAP_MODULES:
            sha.update(name.encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def identity_check(identity, subject_index, draw_seed, synthetic, stage_hashes):
    base = subject_index // VARIANTS
    subject = synthetic['virtual_subjects'][subject_index]
    assert identity['seed_prefix'] == [draw_seed, subject_index, 0]
    assert identity['physical_section_id'] == f'stream-v7-seed-{draw_seed}-row-0-virtual-{subject_index}'
    assert identity['virtual_index'] == subject_index
    assert identity['virtual_subject_id'] == subject['virtual_subject_id']
    assert identity['base_lineage'] == synthetic['base_subjects'][base]['lineage']
    assert identity['split'] == subject['base_lineage']['split'] == 'train'
    assert all(identity['base_lineage'][key] == synthetic['base_subjects'][base]['lineage'][key]
               for key in ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
    assert identity['shape_h_w'] == [SIDE, SIDE]
    assert identity['eligible'] == (identity['visible_support_mass'] >= SIDE**2 / 144)
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


def rigid_points(item):
    state = item['state'][None].cuda()
    reflection = item['reflection'][None].cuda()
    centre, frame, basis = full_frame_state_to_components(state)
    y = (torch.arange(GRID, device='cuda', dtype=state.dtype) + .5) / GRID - .5 / SIDE
    x = (torch.arange(GRID, device='cuda', dtype=state.dtype) + .5) / GRID - .5 / SIDE
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((sx.expand(1, -1, -1), yy.expand(1, -1, -1)), -1)
    rigid = normalized_raster_to_ccf(
        centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    return site_points(rigid, item['site_indices'].cuda())


def subgroup(rows):
    if not rows:
        return {'n': 0, 'same_grid_no_warp_mean_mm': None,
                'mapped_mean_mm': None, 'gain_mean_mm': None}
    return {'n': len(rows),
        'same_grid_no_warp_mean_mm': float(np.mean([r['same_grid_no_warp_mm'] for r in rows])),
        'mapped_mean_mm': float(np.mean([r['exact_mapped_mm'] for r in rows])),
        'gain_mean_mm': float(np.mean([r['gain_mm'] for r in rows])),
        'field_displacement_mean_mm': float(np.mean([r['field_displacement_mean_mm'] for r in rows])),
        'field_displacement_rms_mm': float(np.mean([r['field_displacement_rms_mm'] for r in rows])),
        'surviving_site_displacement_mean_mm': float(np.mean([
            r['surviving_site_displacement_mean_mm'] for r in rows])),
        'surviving_site_displacement_rms_mm': float(np.mean([
            r['surviving_site_displacement_rms_mm'] for r in rows])),
        'max_postlimit_derivative_bound': max(r['postlimit_derivative_bound'] for r in rows)}


def summary(rows):
    value = {**subgroup(rows),
        'by_base': {str(base): subgroup([r for r in rows if r['base_index'] == base])
                    for base in range(56, 64)},
        'by_sparse_support': {str(flag).lower(): subgroup([r for r in rows
            if r['sparse_support_under_15pct_visible'] == flag]) for flag in (False, True)},
        'by_event': {event: {str(flag).lower(): subgroup([r for r in rows
            if r['event_flags'][event] == flag]) for flag in (False, True)}
            for event in EVENTS}}
    value['nonnegative_gain_bases'] = sum(
        value['by_base'][str(base)]['gain_mean_mm'] >= 0 for base in range(56, 64))
    return value


def make_model(pose_weights, pose_buffers, control_side):
    torch.manual_seed(SEED + 2)
    torch.cuda.manual_seed_all(SEED + 2)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, warp_control_side=control_side).cuda()
    for name in POSE_MODULES:
        prefix = name + '.'
        getattr(model, name).load_state_dict({key[len(prefix):]: value
            for key, value in pose_weights.items() if key.startswith(prefix)}, strict=True)
    with torch.no_grad():
        for name in POSE_BUFFERS:
            getattr(model, name).copy_(pose_buffers[name])
    model.eval()
    return model


def rerender(model, atlas, item):
    inputs = item['inputs'][None].cuda()
    state = item['state'][None].cuda()
    reflection = item['reflection'][None].cuda()
    offsets = item['offsets'][None].cuda()
    weights = item['weights'][None].cuda()
    indices = item['site_indices'].cuda()
    target = item['site_target'].cuda()
    rigid = item['rigid_points'].cuda()
    prediction = model.predict(inputs)
    prediction['state'] = state[:, None]
    mapped = model.map(prediction, offsets,
        torch.zeros((1, 1), device='cuda', dtype=torch.long), reflection[:, None],
        (GRID, GRID), atlas, weights, feature_side=GRID, source_shape=(SIDE, SIDE),
        max_derivative=LIMIT)
    surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0, 0][None]
    points = site_points(surface, indices)
    local = mapped['local_displacement_um'][:, 0]
    basis = full_frame_state_to_components(state)[2]
    canonical = local.flip(-1) if bool(reflection[0]) else local
    derivative = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    shift = (points - rigid).norm(dim=-1) / 1000
    oracle = (target - rigid).norm(dim=-1) / 1000
    magnitude = local.norm(dim=1) / 1000
    error = float((points - target).norm(dim=-1).mean() / 1000)
    return {'exact_mapped_mm': error,
        'gain_mm': item['same_grid_no_warp_mm'] - error,
        'field_displacement_mean_mm': float(magnitude.mean()),
        'field_displacement_rms_mm': float(magnitude.square().mean().sqrt()),
        'surviving_site_displacement_mean_mm': float(shift.mean()),
        'surviving_site_displacement_rms_mm': float(shift.square().mean().sqrt()),
        'oracle_surviving_site_displacement_mean_mm': float(oracle.mean()),
        'oracle_surviving_site_displacement_rms_mm': float(oracle.square().mean().sqrt()),
        'prelimit_derivative_bound': float(mapped['prelimit_derivative_bound'][0, 0]),
        'deformation_rescale': float(mapped['deformation_rescale'][0, 0]),
        'postlimit_derivative_bound': float(derivative[0])}


def main():
    if sys.flags.optimize:
        raise RuntimeError('Run without -O: assertions are the audit.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), 'Faithful checkpoint rerender requires CUDA.'

    assert sha256(SOURCE / SOURCE_NAMES[0]) == committed_sha256('training/' + SOURCE_NAMES[0]) == RUNNER_SHA256
    assert sha256(PROTOCOL) == committed_sha256('docs/publication/' + PROTOCOL.name) == PROTOCOL_SHA256
    config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
    receipt = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
    assert config['candidate'] == receipt['candidate'] == 282
    assert config['seed'] == receipt['seed'] == SEED
    assert config['protocol_sha256'] == receipt['protocol_sha256'] == PROTOCOL_SHA256
    assert set(config['source_sha256']) == set(SOURCE_NAMES)
    assert receipt['source_sha256'] == config['source_sha256']
    for name in SOURCE_NAMES:
        expected = committed_sha256('training/' + name)
        assert sha256(SOURCE / name) == config['source_sha256'][name] == expected
    assert config['parent_candidate'] == 275
    assert config['parent_pose_modules'] == list(POSE_MODULES)
    assert config['parent_pose_buffers'] == list(POSE_BUFFERS)
    assert config['arms'] == {name: list(spec) for name, spec in ARMS.items()}
    assert config['updates_per_arm'] == UPDATES and config['readout_steps'] == list(STEPS)
    assert receipt['steps'] == list(STEPS)
    assert config['train_base_indices'] == list(range(56))
    assert config['inner_base_indices'] == list(range(56, 64))
    assert config['sections_per_inner_base'] == 8
    assert config['source_shape'] == [SIDE, SIDE] and config['map_grid_shape'] == [GRID, GRID]
    assert config['train_surviving_sites'] == TRAIN_SITES
    assert config['readout_surviving_sites'] == READOUT_SITES
    assert config['map_derivative_limit'] == LIMIT
    assert config['checkpoint_selection'] == 'none; fixed 0,500,1500 readouts; only 1500 eligible for gate'
    assert config['optimizer'] == 'fresh matched AdamW, lr=1e-4, weight_decay=1e-4, clip=2'
    assert config['objective'] == 'mean exact-plane surviving-site Euclidean CCF error in mm; no rank, reliability, or deformation penalty'
    assert config['result_role'] == receipt['result_role'] == 'TRAIN-inner mechanism diagnostic, not release model or real-animal location validation'
    assert config['dev_cal_final_or_public_benchmark_used'] is False
    assert receipt['dev_cal_final_or_public_benchmark_used'] is False and receipt['calibrated'] is False

    parent_receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
    assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
    assert config['parent_completed_sha256'] == sha256(PARENT / 'completed.json')
    assert config['parent_config_sha256'] == parent_receipt['config_sha256'] == sha256(PARENT / 'config.json')
    parent_path = PARENT / 'pose_step_08000.pt'
    assert config['parent_checkpoint_sha256'] == parent_receipt['checkpoint_sha256']['8000'] == sha256(parent_path)

    synthetic = config['synthetic_provenance']
    assert synthetic['new_independent_train_bases'] == 56 and synthetic['all_train_bases'] == 64
    assert synthetic['calibrated'] is False
    assert len(synthetic['base_subjects']) == 64
    assert len(synthetic['virtual_subjects']) == 64 * VARIANTS
    assert receipt['input_sha256'] == synthetic['input_sha256']
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
    assert set(receipt['checkpoint_sha256']) == {
        f'{arm}_{step}' for arm in ARMS for step in STEPS}
    for arm in ARMS:
        for step in STEPS:
            assert sha256(RUN / f'{arm}_step_{step:05d}.pt') == receipt['checkpoint_sha256'][f'{arm}_{step}']

    global OneShotJointSliceModel, full_frame_state_to_components
    global normalized_raster_to_ccf, ribbon_derivative_frobenius_bound
    global sample_one_shot_slide_artifacts_v9, load_streaming_synthetic_v7_64
    from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
    from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
    from training.arbitrary_plane_geometry import normalized_raster_to_ccf
    from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
    from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
    from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

    safe = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType,
            torch.torch_version.TorchVersion]
    with torch.serialization.safe_globals(safe):
        panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=True)
    draws = jsonl(RUN / 'inner_draws.jsonl')
    training_draws = jsonl(RUN / 'training_draws.jsonl')
    schedule = jsonl(RUN / 'training_schedule.jsonl')
    training = jsonl(RUN / 'training.jsonl')
    readouts = jsonl(RUN / 'readouts.jsonl')
    assert len(panel) == receipt['panel_sections'] == 64
    assert len(draws) == config['panel_attempts'] == receipt['panel_attempts']
    assert len(schedule) == receipt['training_schedule_sections'] == UPDATES
    assert len(training) == len(ARMS) * UPDATES
    assert len(readouts) == len(ARMS) * len(STEPS) * len(panel)

    stage_hashes = {version: config['source_sha256'][f'arbitrary_plane_one_shot_slide_artifacts_v{version}.py']
                    for version in (3, 4, 6, 7, 9)}
    panel_rng = np.random.default_rng(SEED + 1)
    accepted, seen = [], set()
    for attempt, draw in enumerate(draws, 1):
        assert len(accepted) < 64
        base, slot = 56 + len(accepted) // 8, len(accepted) % 8
        subject_index = base * VARIANTS + int(panel_rng.integers(VARIANTS))
        draw_seed = SEED + 1_000_000_000 + attempt
        assert (draw['base_index'], draw['slot'], draw['attempt'],
                draw['subject_index'], draw['draw_seed']) == (
                base, slot, attempt, subject_index, draw_seed)
        identity = draw['provenance']
        identity_check(identity, subject_index, draw_seed, synthetic, stage_hashes)
        assert draw['valid_pixels'] == identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid']
        assert type(draw['used']) is bool
        assert draw['used'] == (identity['eligible'] and draw['valid_pixels'] >= READOUT_SITES)
        assert identity['physical_section_id'] not in seen
        seen.add(identity['physical_section_id'])
        if draw['used']:
            accepted.append(draw)
    assert len(accepted) == 64

    train_rng = np.random.default_rng(SEED + 3)
    accepted_train, train_seen = [], set()
    for attempt, draw in enumerate(training_draws, 1):
        assert len(accepted_train) < UPDATES
        base = int(train_rng.integers(56))
        subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
        draw_seed = SEED + 2_000_000_000 + attempt
        assert (draw['step'], draw['attempt'], draw['base_index'],
                draw['subject_index'], draw['draw_seed']) == (
                len(accepted_train) + 1, attempt, base, subject_index, draw_seed)
        identity = draw['provenance']
        identity_check(identity, subject_index, draw_seed, synthetic, stage_hashes)
        assert draw['used'] == (identity['eligible'] and
            identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] >= SIDE**2 / 144)
        if draw['used']:
            assert identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] >= TRAIN_SITES
        assert identity['physical_section_id'] not in seen and identity['physical_section_id'] not in train_seen
        train_seen.add(identity['physical_section_id'])
        if draw['used']:
            accepted_train.append(draw)
    assert len(accepted_train) == UPDATES
    for step, (entry, draw) in enumerate(zip(schedule, accepted_train), 1):
        assert (entry['step'], entry['base_index'], entry['subject_index'], entry['draw_seed'],
                entry['physical_section_id']) == (
                step, draw['base_index'], draw['subject_index'], draw['draw_seed'],
                draw['provenance']['physical_section_id'])
        for key in ('input_sha256', 'site_target_sha256', 'state_psf_sha256'):
            assert len(entry[key]) == 64 and all(char in '0123456789abcdef' for char in entry[key])
    assert len({entry['physical_section_id'] for entry in schedule}) == UPDATES
    for arm_index, arm in enumerate(ARMS):
        block = training[arm_index * UPDATES:(arm_index + 1) * UPDATES]
        assert [(r['arm'], r['step'], r['physical_section_id']) for r in block] == [
            (arm, entry['step'], entry['physical_section_id']) for entry in schedule]
        for row, entry in zip(block, schedule):
            assert row['input_sha256'] == entry['input_sha256']
            assert row['site_target_sha256'] == entry['site_target_sha256']
            assert all(math.isfinite(row[key]) for key in (
                'loss_mm_before_update', 'gradient_norm', 'elapsed_seconds'))
            assert row['loss_mm_before_update'] >= 0 and row['gradient_norm'] >= 0
            assert row['elapsed_seconds'] > 0
    assert receipt['counters'] == {arm: {
        'attempted_draws_in_first_arm': len(training_draws) if arm == 'smooth16_full' else None,
        'accepted_sections': UPDATES, 'schedule_replayed': arm != 'smooth16_full'} for arm in ARMS}

    context = load_streaming_synthetic_v7_64(device='cuda')
    assert context['provenance'] == synthetic
    used_index = 0
    for draw in draws:
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(
                context, [draw['subject_index']], draw['draw_seed'], SIDE)
        assert json.loads(json.dumps(sample['provenance'][0])) == draw['provenance']
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        assert len(valid) == draw['valid_pixels']
        assert (bool(sample['eligible'][0]) and len(valid) >= READOUT_SITES) == draw['used']
        if not draw['used']:
            continue
        item = panel[used_index]
        used_index += 1
        expected_indices = valid[torch.linspace(0, len(valid) - 1, READOUT_SITES,
                                                device=valid.device).long()]
        assert item['site_indices'].shape == (READOUT_SITES,)
        assert torch.equal(item['site_indices'], expected_indices.cpu())
        assert item['inputs'].shape == (5, SIDE, SIDE) and item['state'].shape == (12,)
        assert item['offsets'].shape == item['weights'].shape == (9,)
        assert item['site_target'].shape == item['rigid_points'].shape == (READOUT_SITES, 3)
        close(item['inputs'], sample['inputs'][0].cpu(), 1e-5)
        close(item['state'], sample['state'][0].cpu(), .01)
        assert torch.equal(item['reflection'], sample['reflection'][0].cpu())
        close(item['offsets'], sample['offsets'][0].cpu(), .01)
        close(item['weights'], sample['weights'][0].cpu(), 1e-5)
        close(item['site_target'], sample['centre'][0].reshape(-1, 3)[expected_indices].cpu(), .1)
        assert (item['base_index'], item['slot'], item['subject_index'], item['draw_seed']) == (
            draw['base_index'], draw['slot'], draw['subject_index'], draw['draw_seed'])
        assert item['physical_section_id'] == draw['provenance']['physical_section_id']
        assert json.loads(json.dumps(item['provenance'])) == draw['provenance']
        close(item['valid_fraction'], len(valid) / SIDE**2, 5e-8)
        close(item['visible_fraction'], float((sample['visible'][0] > .25).float().mean()), 5e-8)
        normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        close(item['obliquity_deg'],
              float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))), 1e-4)
        artifact = draw['provenance']['one_shot_slide_artifacts_v3']
        assert item['event_flags'] == {event: bool(artifact['events'][event]) for event in EVENTS}
        with torch.no_grad():
            rigid = rigid_points(item)
        close(item['rigid_points'], rigid.cpu(), .02)
        no_warp = float((rigid - item['site_target'].cuda()).norm(dim=-1).mean() / 1000)
        close(item['same_grid_no_warp_mm'], no_warp, 1e-5)
    assert used_index == 64

    # Independent digest spot checks of the fixed first-arm training schedule.
    for step in (1, 500, 1000, 1500):
        entry = schedule[step - 1]
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(
                context, [entry['subject_index']], entry['draw_seed'], SIDE)
        assert sample['provenance'][0]['physical_section_id'] == entry['physical_section_id']
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        indices = valid[torch.linspace(0, len(valid) - 1, TRAIN_SITES, device=valid.device).long()]
        target = sample['centre'][0].reshape(-1, 3)[indices]
        assert tensor_digest(sample['inputs']) == entry['input_sha256']
        assert tensor_digest(torch.cat((indices[:, None].float(), target), -1)) == entry['site_target_sha256']
        assert tensor_digest(torch.cat((sample['state'], sample['reflection'][:, None].float(),
            sample['offsets'], sample['weights']), -1)) == entry['state_psf_sha256']

    # The hash-pinned parent tensor dictionary is the frozen feature source.
    with torch.serialization.safe_globals(safe):
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    assert parent['step'] == 8000 and parent['config']['candidate'] == 275
    pose_weights = {name: value for name, value in parent['model'].items()
                    if name.split('.')[0] in POSE_MODULES}
    pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
    del parent
    assert set(receipt['initial_mapper_sha256']) == set(ARMS)
    assert len(set(receipt['initial_mapper_sha256'].values())) == 1
    assert set(receipt['readouts']) == set(ARMS)
    readout_position = 0
    all_summaries = {}
    for arm, (control_side, atlas_mode) in ARMS.items():
        model = make_model(pose_weights, pose_buffers, control_side)
        assert mapper_digest(model) == receipt['initial_mapper_sha256'][arm]
        if atlas_mode == 'full':
            atlas = context['atlas']
        else:
            atlas = context['atlas'].clone()
            atlas[0].zero_()
            if atlas_mode == 'zero':
                atlas[1].zero_()
        all_summaries[arm] = {}
        for step in STEPS:
            checkpoint_path = RUN / f'{arm}_step_{step:05d}.pt'
            with torch.serialization.safe_globals(safe):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert (checkpoint['candidate'], checkpoint['arm'], checkpoint['step']) == (282, arm, step)
            assert checkpoint['config_sha256'] == sha256(RUN / 'config.json')
            assert checkpoint['parent_checkpoint_sha256'] == config['parent_checkpoint_sha256']
            assert checkpoint['initial_mapper_sha256'] == receipt['initial_mapper_sha256'][arm]
            assert checkpoint['calibrated'] is False
            expected_keys = {name for name in model.state_dict() if name.split('.')[0] in MAP_MODULES}
            assert set(checkpoint['mapper']) == expected_keys
            for name in MAP_MODULES:
                prefix = name + '.'
                getattr(model, name).load_state_dict({key[len(prefix):]: value
                    for key, value in checkpoint['mapper'].items() if key.startswith(prefix)}, strict=True)
            if step == 0:
                assert mapper_digest(model) == receipt['initial_mapper_sha256'][arm]
                assert not checkpoint['optimizer']['state']
            else:
                states = checkpoint['optimizer']['state']
                assert states and all(int(value['step']) == step for value in states.values())
            group = checkpoint['optimizer']['param_groups']
            assert len(group) == 1 and group[0]['lr'] == 1e-4 and group[0]['weight_decay'] == 1e-4
            rows = readouts[readout_position:readout_position + 64]
            readout_position += 64
            assert [(r['arm'], r['step'], r['physical_section_id']) for r in rows] == [
                (arm, step, item['physical_section_id']) for item in panel]
            with torch.inference_mode():
                for row, item in zip(rows, panel):
                    assert row['panel'] == 'new_v9_train_inner'
                    assert (row['base_index'], row['slot'], row['subject_index'], row['draw_seed']) == (
                        item['base_index'], item['slot'], item['subject_index'], item['draw_seed'])
                    assert row['true_state'] == item['state'].tolist()
                    assert row['true_reflection'] == int(item['reflection'])
                    assert row['site_count'] == READOUT_SITES
                    assert row['event_flags'] == item['event_flags']
                    assert row['provenance'] == json.loads(json.dumps(item['provenance']))
                    artifact = item['provenance']['one_shot_slide_artifacts_v3']
                    assert row['tissue_placed'] == artifact['tissue_placed']
                    assert row['artifact_pixel_counts'] == artifact['pixel_counts']
                    assert row['sparse_support_under_15pct_visible'] == (item['visible_fraction'] < .15)
                    for key in ('valid_fraction', 'visible_fraction', 'obliquity_deg',
                                'same_grid_no_warp_mm'):
                        close(row[key], item[key])
                    measured = rerender(model, atlas, item)
                    for key, value in measured.items():
                        close(row[key], value, 1e-4)
                    assert row['postlimit_derivative_bound'] <= LIMIT + 1e-5
                    close(row['prelimit_derivative_bound'] * row['deformation_rescale'],
                          row['postlimit_derivative_bound'], 1e-5)
                    assert row['gain_mm'] <= row['surviving_site_displacement_mean_mm'] + 1e-5
            calculated = summary(rows)
            assert receipt['readouts'][arm][str(step)] == calculated
            all_summaries[arm][step] = calculated
        del model
        if atlas_mode != 'full':
            del atlas
        torch.cuda.empty_cache()
    assert readout_position == len(readouts)

    baseline = all_summaries['smooth16_full'][0]['same_grid_no_warp_mean_mm']
    for arm in ARMS:
        for step in STEPS:
            close(all_summaries[arm][step]['same_grid_no_warp_mean_mm'], baseline)
    a, b, c, d = (all_summaries[name][UPDATES] for name in ARMS)
    differences = {'A_minus_B': a['gain_mean_mm'] - b['gain_mean_mm'],
        'A_minus_C': a['gain_mean_mm'] - c['gain_mean_mm'],
        'C_minus_D': c['gain_mean_mm'] - d['gain_mean_mm'],
        'A_minus_D': a['gain_mean_mm'] - d['gain_mean_mm']}
    assert receipt['step_1500_gain_differences_mm'] == differences
    gate = {'smooth_architecture_mean_gain_at_least_0_05mm': a['gain_mean_mm'] >= .05,
        'smooth_architecture_at_least_six_nonnegative_bases': a['nonnegative_gain_bases'] >= 6,
        'smooth_architecture_derivative_at_most_0_35':
            a['max_postlimit_derivative_bound'] <= LIMIT + 1e-5,
        'directional_smooth_advantage_over_free_at_least_0_03mm':
            differences['A_minus_B'] >= .03,
        'atlas_intensity_value_at_least_0_03mm': differences['A_minus_C'] >= .03,
        'atlas_support_value_at_least_0_03mm': differences['C_minus_D'] >= .03,
        'all_arms_derivative_at_most_0_35': all(
            all_summaries[arm][step]['max_postlimit_derivative_bound'] <= LIMIT + 1e-5
            for arm in ARMS for step in STEPS)}
    assert receipt['gate'] == gate
    assert receipt['smooth_architecture_learns'] == all(gate[key] for key in (
        'smooth_architecture_mean_gain_at_least_0_05mm',
        'smooth_architecture_at_least_six_nonnegative_bases',
        'smooth_architecture_derivative_at_most_0_35'))
    assert math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0
    print(json.dumps({'audit': 'pass', 'candidate': 282, 'runner_commit': COMMIT,
        'panel_sections': len(panel), 'panel_attempts': len(draws),
        'training_sections': len(schedule),
        'step_1500_gain_mm': {arm: all_summaries[arm][UPDATES]['gain_mean_mm'] for arm in ARMS},
        'gate': gate, 'smooth_architecture_learns': receipt['smooth_architecture_learns']},
        allow_nan=False))


if __name__ == '__main__':
    main()
