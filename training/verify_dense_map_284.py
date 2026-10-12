"""Read-only public audit of 284. Run only after train_dense_map_284.py exits."""
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
RUN = ROOT / 'runs/dense_map_supervision_284_001'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
PROTOCOL = SOURCE.parent / 'docs/publication/DENSE_MAP_SUPERVISION_284_PROTOCOL_20261012.md'
COMMIT = '7a8cdb4716459e2cbafd0c3262f8884a0ff78e15'
RUNNER_SHA256 = '81ccc9ac32ae7a4b14573609d761c0ce7a8e8f9c7a61bf807ceee850f133ac9c'
PROTOCOL_SHA256 = '62ceee2f08a0374eb9782f54b578c37fab134050850af93b9cda4fa7694deedd'
SEED, SIDE, GRID, UPDATES, VARIANTS = 20261012284, 192, 64, 3000, 64
READOUT_SITES, LIMIT = 512, .35
STEPS = (0, 750, 1500, 3000)
ARMS = {'sparse128': 128, 'dense2048': 2048}
POSE_MODULES = ('encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global')
POSE_BUFFERS = ('normal_anchor_frames', 'center_origin', 'center_scale',
                'euclidean_scale', 'rotation_grid')
MAP_MODULES = ('warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair')
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE_NAMES = (
    'train_dense_map_284.py', 'train_image_map_282.py',
    'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_joint_model_v7.py',
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
    'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'global_atlas_contrast_090.py',
)
SYNTHETIC_SOURCE_NAMES = {
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_subject_sampling_v6.py',
    'subject_deformed_slab_multiresolution_bundle_v2.py',
}
EXTRA_SOURCE_NAMES = (
    'arbitrary_plane_deformation_primitives.py',
    'arbitrary_plane_catalogue_v3.py',
    'arbitrary_plane_catalogue_binding_v3.py',
    'arbitrary_plane_support.py',
    'arbitrary_plane_acquisition_v2.py',
    'arbitrary_plane_manifest.py',
    'arbitrary_plane_rendered_generator.py',
    'arbitrary_plane_subject_deformation_v2.py',
    'arbitrary_plane_subject_section_v2.py',
    'arbitrary_plane_subject_slab_v2.py',
    'arbitrary_plane_support_resolution_v2.py',
    'arbitrary_plane_synthetic_generator_v2.py',
    'slab_refinement_gate_status_v2.py',
    'subject_deformed_slab_multiresolution_assessment_v2.py',
    'subject_deformed_slab_qualification_v2.py',
)
OUTPUT_NAMES = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
    'training_draws.jsonl', 'training_schedule.jsonl', 'training.jsonl',
    'readouts.jsonl')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def committed_sha256(path):
    return hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{COMMIT}:{path}'], cwd=SOURCE.parent)).hexdigest()


def reject_constant(value):
    raise ValueError(f'non-finite JSON constant: {value}')


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'), parse_constant=reject_constant)


def read_jsonl(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line, parse_constant=reject_constant) for line in stream]


def close(actual, expected, atol=1e-7):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=atol)


def compare_tree(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            compare_tree(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_tree(left, right)
    elif isinstance(actual, float):
        close(actual, expected, 1e-5)
    else:
        assert actual == expected


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def mapper_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.split('.')[0] in MAP_MODULES:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_identity(identity, subject_index, draw_seed, synthetic, stage_hashes):
    base = subject_index // VARIANTS
    subject = synthetic['virtual_subjects'][subject_index]
    assert identity['seed_prefix'] == [draw_seed, subject_index, 0]
    assert identity['physical_section_id'] == f'stream-v7-seed-{draw_seed}-row-0-virtual-{subject_index}'
    assert identity['virtual_index'] == subject_index
    assert identity['virtual_subject_id'] == subject['virtual_subject_id']
    assert identity['base_lineage'] == synthetic['base_subjects'][base]['lineage']
    assert identity['split'] == subject['base_lineage']['split'] == 'train'
    assert identity['shape_h_w'] == [SIDE, SIDE]
    assert identity['eligible'] == (identity['visible_support_mass'] >= SIDE**2 / 144)
    for version in (3, 4, 6, 7, 9):
        assert identity[f'one_shot_slide_artifacts_v{version}']['code_sha256'] == stage_hashes[version]
    assert identity['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair'] is False
    assert identity['one_shot_slide_artifacts_v9']['seed_branch'] == 34
    assert identity['one_shot_slide_artifacts_v9']['sampling'] == \
        'one independent physical plane per row; no appearance pairing'


def select_sites(sample, budget):
    valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    count = min(budget, len(valid))
    indices = valid[torch.linspace(0, len(valid) - 1, count,
                                   device=valid.device).long()]
    assert len(indices.unique()) == count
    target = sample['centre'][0].reshape(-1, 3)[indices]
    return valid, indices, target


def site_points(surface, indices):
    xy = torch.stack((indices.remainder(SIDE),
                      indices.div(SIDE, rounding_mode='floor')), -1).float()
    grid = ((xy + .5) * (2 / SIDE) - 1)[None, None]
    return F.grid_sample(surface.permute(0, 3, 1, 2), grid, mode='bilinear',
                         padding_mode='border', align_corners=False)[0, :, 0].T


def rigid_points(item):
    state = item['state'][None].cuda()
    reflection = item['reflection'][None].cuda()
    centre, frame, basis = full_frame_state_to_components(state)
    axis = (torch.arange(GRID, device='cuda', dtype=state.dtype) + .5) / GRID - .5 / SIDE
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((sx.expand(1, -1, -1), yy.expand(1, -1, -1)), -1)
    surface = normalized_raster_to_ccf(
        centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    return site_points(surface, item['site_indices'].cuda())


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


def summarize(rows):
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


def make_model(pose_weights, pose_buffers):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, warp_control_side=16).cuda()
    for name in POSE_MODULES:
        prefix = name + '.'
        getattr(model, name).load_state_dict({key[len(prefix):]: value
            for key, value in pose_weights.items() if key.startswith(prefix)}, strict=True)
    with torch.no_grad():
        for name in POSE_BUFFERS:
            getattr(model, name).copy_(pose_buffers[name])
    model.requires_grad_(False)
    for name in MAP_MODULES:
        getattr(model, name).requires_grad_(True)
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
    prediction = model.predict(inputs)
    prediction['state'] = state[:, None]
    mapped = model.map(prediction, offsets,
        torch.zeros((1, 1), device='cuda', dtype=torch.long), reflection[:, None],
        (GRID, GRID), atlas, weights, feature_side=GRID, source_shape=(SIDE, SIDE),
        max_derivative=LIMIT)
    points = site_points(mapped['centre_surface_ccf_ap_dv_ml_um'][0, 0][None], indices)
    rigid = rigid_points(item)
    local = mapped['local_displacement_um'][:, 0]
    basis = full_frame_state_to_components(state)[2]
    canonical = local.flip(-1) if bool(reflection[0]) else local
    derivative = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    shift = (points - rigid).norm(dim=-1) / 1000
    oracle = (target - rigid).norm(dim=-1) / 1000
    magnitude = local.norm(dim=1) / 1000
    no_warp = float((rigid - target).norm(dim=-1).mean() / 1000)
    error = float((points - target).norm(dim=-1).mean() / 1000)
    return {'same_grid_no_warp_mm': no_warp, 'exact_mapped_mm': error,
        'gain_mm': no_warp - error,
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
    assert SOURCE.drive.upper() == RUN.drive.upper() == PARENT.drive.upper() == 'I:'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), 'Faithful checkpoint rerender requires CUDA.'

    assert sha256(SOURCE / SOURCE_NAMES[0]) == \
        committed_sha256('training/' + SOURCE_NAMES[0]) == RUNNER_SHA256
    assert sha256(PROTOCOL) == \
        committed_sha256('docs/publication/' + PROTOCOL.name) == PROTOCOL_SHA256
    for name in EXTRA_SOURCE_NAMES:
        assert sha256(SOURCE / name) == committed_sha256('training/' + name)

    config = read_json(RUN / 'config.json')
    receipt = read_json(RUN / 'completed.json')
    assert config['candidate'] == receipt['candidate'] == 284
    assert config['seed'] == receipt['seed'] == SEED
    assert config['protocol_sha256'] == receipt['protocol_sha256'] == PROTOCOL_SHA256
    assert set(config['source_sha256']) == set(SOURCE_NAMES)
    assert receipt['source_sha256'] == config['source_sha256']
    for name in SOURCE_NAMES:
        expected = committed_sha256('training/' + name)
        assert sha256(SOURCE / name) == config['source_sha256'][name] == expected
    assert config['mapper_initialization_seed'] == SEED
    assert config['parent_candidate'] == 275
    assert config['arms'] == ARMS
    assert config['updates_per_arm'] == UPDATES
    assert config['readout_steps'] == receipt['steps'] == list(STEPS)
    assert config['train_base_indices'] == list(range(56))
    assert config['inner_base_indices'] == list(range(56, 64))
    assert config['sections_per_inner_base'] == 8
    assert config['source_shape'] == [SIDE, SIDE] and config['map_grid_shape'] == [GRID, GRID]
    assert config['readout_sites'] == READOUT_SITES and config['map_derivative_limit'] == LIMIT
    assert config['optimizer'] == 'matched AdamW lr=1e-4 weight_decay=1e-4 clip=2 batch=1'
    assert config['objective'] == 'exact-plane mean Euclidean surviving-site CCF error in mm'
    assert config['feature_lineage'] == \
        'frozen 275 step-8000 scratch pose encoder/lateral; true state/reflection overrides pose'
    assert config['checkpoint_selection'] == \
        'none; fixed 0,750,1500,3000 readouts; only 3000 eligible for gate'
    assert config['result_role'] == receipt['result_role'] == \
        'TRAIN-inner supervision-density diagnostic, not a release model or acquired location validation'
    assert config['dev_cal_final_or_public_benchmark_used'] is False
    assert receipt['dev_cal_final_or_public_benchmark_used'] is False
    assert receipt['calibrated'] is False
    assert config['torch'] == torch.__version__ and config['numpy'] == np.__version__

    parent_receipt = read_json(PARENT / 'completed.json')
    assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
    assert config['parent_completed_sha256'] == sha256(PARENT / 'completed.json')
    assert parent_receipt['config_sha256'] == sha256(PARENT / 'config.json')
    parent_path = PARENT / 'pose_step_08000.pt'
    assert config['parent_checkpoint_sha256'] == \
        parent_receipt['checkpoint_sha256']['8000'] == sha256(parent_path)

    synthetic = config['synthetic_provenance']
    assert synthetic['new_independent_train_bases'] == 56
    assert synthetic['all_train_bases'] == 64 and synthetic['calibrated'] is False
    assert len(synthetic['base_subjects']) == 64
    assert len(synthetic['virtual_subjects']) == 64 * VARIANTS
    assert set(synthetic['source_sha256']) == SYNTHETIC_SOURCE_NAMES
    assert receipt['input_sha256'] == synthetic['input_sha256']
    for name, expected in synthetic['source_sha256'].items():
        assert Path(name).name == name
        assert sha256(SOURCE / name) == expected == committed_sha256('training/' + name)
    for path, expected in synthetic['input_sha256'].items():
        bound = Path(path).resolve()
        assert bound.is_relative_to((ROOT / 'data').resolve())
        assert sha256(bound) == expected
    for key in ('template', 'annotation'):
        bound = Path(synthetic['atlas_binding'][key + '_path']).resolve()
        assert bound.is_relative_to((ROOT / 'data').resolve())
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
            path = RUN / f'{arm}_step_{step:05d}.pt'
            assert sha256(path) == receipt['checkpoint_sha256'][f'{arm}_{step}']

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
    draws = read_jsonl(RUN / 'inner_draws.jsonl')
    training_draws = read_jsonl(RUN / 'training_draws.jsonl')
    schedule = read_jsonl(RUN / 'training_schedule.jsonl')
    training = read_jsonl(RUN / 'training.jsonl')
    readouts = read_jsonl(RUN / 'readouts.jsonl')
    assert len(panel) == receipt['panel_sections'] == 64
    assert len(draws) == config['panel_attempts'] == receipt['panel_attempts']
    assert len(training_draws) == receipt['training_attempts']
    assert len(schedule) == receipt['training_schedule_sections'] == UPDATES
    assert len(training) == len(ARMS) * UPDATES
    assert len(readouts) == len(ARMS) * len(STEPS) * len(panel)

    context = load_streaming_synthetic_v7_64(device='cuda')
    assert context['provenance'] == synthetic
    stage_hashes = {version: config['source_sha256'][
        f'arbitrary_plane_one_shot_slide_artifacts_v{version}.py']
        for version in (3, 4, 6, 7, 9)}

    panel_rng = np.random.default_rng(SEED + 1)
    accepted_panel, seen_ids = 0, set()
    for attempt, draw in enumerate(draws, 1):
        assert accepted_panel < 64
        base, slot = 56 + accepted_panel // 8, accepted_panel % 8
        subject_index = base * VARIANTS + int(panel_rng.integers(VARIANTS))
        draw_seed = SEED + 1_000_000_000 + attempt
        assert (draw['base_index'], draw['slot'], draw['attempt'],
                draw['subject_index'], draw['draw_seed']) == (
                base, slot, attempt, subject_index, draw_seed)
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(
                context, [subject_index], draw_seed, SIDE)
        identity = sample['provenance'][0]
        assert json.loads(json.dumps(identity)) == draw['provenance']
        check_identity(identity, subject_index, draw_seed, synthetic, stage_hashes)
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        used = bool(sample['eligible'][0]) and len(valid) >= READOUT_SITES
        assert draw['valid_pixels'] == len(valid)
        assert draw['used'] is used
        assert identity['physical_section_id'] not in seen_ids
        seen_ids.add(identity['physical_section_id'])
        if not used:
            del sample
            continue
        item = panel[accepted_panel]
        _, indices, target = select_sites(sample, READOUT_SITES)
        assert len(indices) == READOUT_SITES
        assert torch.equal(item['site_indices'], indices.cpu())
        assert item['inputs'].shape == (5, SIDE, SIDE) and item['state'].shape == (12,)
        assert item['offsets'].shape == item['weights'].shape == (9,)
        assert item['site_target'].shape == item['rigid_points'].shape == (READOUT_SITES, 3)
        close(item['inputs'], sample['inputs'][0].cpu(), 1e-5)
        close(item['state'], sample['state'][0].cpu(), .01)
        assert torch.equal(item['reflection'], sample['reflection'][0].cpu())
        close(item['offsets'], sample['offsets'][0].cpu(), .01)
        close(item['weights'], sample['weights'][0].cpu(), 1e-5)
        close(item['site_target'], target.cpu(), .1)
        assert (item['base_index'], item['slot'], item['subject_index'], item['draw_seed']) == (
            base, slot, subject_index, draw_seed)
        assert item['physical_section_id'] == identity['physical_section_id']
        assert item['provenance'] == json.loads(json.dumps(identity))
        close(item['valid_fraction'], len(valid) / SIDE**2, 5e-8)
        close(item['visible_fraction'], float((sample['visible'][0] > .25).float().mean()), 5e-8)
        normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        close(item['obliquity_deg'],
              float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))), 1e-4)
        artifact = identity['one_shot_slide_artifacts_v3']
        assert item['event_flags'] == {event: bool(artifact['events'][event]) for event in EVENTS}
        with torch.no_grad():
            rigid = rigid_points(item)
        close(item['rigid_points'], rigid.cpu(), .02)
        close(item['same_grid_no_warp_mm'],
              float((rigid - target).norm(dim=-1).mean() / 1000), 1e-5)
        accepted_panel += 1
        del sample
    assert accepted_panel == 64
    panel_ids = {item['physical_section_id'] for item in panel}
    assert len(panel_ids) == 64

    train_rng = np.random.default_rng(SEED + 3)
    accepted_train, train_seen = 0, set()
    for attempt, draw in enumerate(training_draws, 1):
        assert accepted_train < UPDATES
        base = int(train_rng.integers(56))
        subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
        draw_seed = SEED + 2_000_000_000 + attempt
        assert (draw['step'], draw['attempt'], draw['base_index'],
                draw['subject_index'], draw['draw_seed']) == (
                accepted_train + 1, attempt, base, subject_index, draw_seed)
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(
                context, [subject_index], draw_seed, SIDE)
        identity = sample['provenance'][0]
        assert json.loads(json.dumps(identity)) == draw['provenance']
        check_identity(identity, subject_index, draw_seed, synthetic, stage_hashes)
        used = bool(sample['eligible'][0])
        assert draw['used'] is used
        assert identity['physical_section_id'] not in train_seen
        train_seen.add(identity['physical_section_id'])
        if not used:
            del sample
            continue
        entry = schedule[accepted_train]
        assert set(entry) == {'step', 'base_index', 'subject_index', 'draw_seed',
            'physical_section_id', 'input_sha256', 'sparse_target_sha256',
            'dense_target_sha256', 'state_psf_sha256', 'dense_site_count'}
        assert (entry['step'], entry['base_index'], entry['subject_index'], entry['draw_seed'],
                entry['physical_section_id']) == (
                accepted_train + 1, base, subject_index, draw_seed,
                identity['physical_section_id'])
        valid, sparse_indices, sparse_target = select_sites(sample, ARMS['sparse128'])
        _, dense_indices, dense_target = select_sites(sample, ARMS['dense2048'])
        assert len(valid) >= 128 and len(sparse_indices) == 128
        assert len(dense_indices) == min(2048, len(valid))
        sparse_sha = tensor_digest(torch.cat((sparse_indices[:, None].float(), sparse_target), -1))
        dense_sha = tensor_digest(torch.cat((dense_indices[:, None].float(), dense_target), -1))
        state_sha = tensor_digest(torch.cat((sample['state'],
            sample['reflection'][:, None].float(), sample['offsets'], sample['weights']), -1))
        assert entry['input_sha256'] == tensor_digest(sample['inputs'])
        assert entry['sparse_target_sha256'] == sparse_sha
        assert entry['dense_target_sha256'] == dense_sha
        assert entry['state_psf_sha256'] == state_sha
        assert entry['dense_site_count'] == len(dense_indices)
        accepted_train += 1
        del sample
    assert accepted_train == UPDATES
    schedule_ids = {entry['physical_section_id'] for entry in schedule}
    assert len(schedule_ids) == UPDATES and panel_ids.isdisjoint(schedule_ids)

    train_counts = {arm: [] for arm in ARMS}
    elapsed = -math.inf
    for arm_index, arm in enumerate(ARMS):
        block = training[arm_index * UPDATES:(arm_index + 1) * UPDATES]
        assert [(row['arm'], row['step'], row['physical_section_id']) for row in block] == [
            (arm, entry['step'], entry['physical_section_id']) for entry in schedule]
        for row, entry in zip(block, schedule):
            target_sha = entry['sparse_target_sha256'] if arm == 'sparse128' \
                else entry['dense_target_sha256']
            site_count = 128 if arm == 'sparse128' else entry['dense_site_count']
            assert row['input_sha256'] == entry['input_sha256']
            assert row['site_target_sha256'] == target_sha
            assert row['site_count'] == site_count
            assert all(math.isfinite(row[key]) for key in
                       ('loss_mm_before_update', 'gradient_norm', 'elapsed_seconds'))
            assert row['loss_mm_before_update'] >= 0 and row['gradient_norm'] >= 0
            assert row['elapsed_seconds'] >= elapsed
            elapsed = row['elapsed_seconds']
            train_counts[arm].append(site_count)
        reported = receipt['train_site_count'][arm]
        assert reported['min'] == min(train_counts[arm])
        assert reported['max'] == max(train_counts[arm])
        close(reported['mean'], float(np.mean(train_counts[arm])))

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
    for arm in ARMS:
        model = make_model(pose_weights, pose_buffers)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable and all(name.split('.')[0] in MAP_MODULES for name in trainable)
        assert mapper_digest(model) == receipt['initial_mapper_sha256'][arm]
        all_summaries[arm] = {}
        for step in STEPS:
            path = RUN / f'{arm}_step_{step:05d}.pt'
            with torch.serialization.safe_globals(safe):
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert set(checkpoint) == {'candidate', 'arm', 'step', 'mapper', 'optimizer',
                'config_sha256', 'parent_checkpoint_sha256', 'initial_mapper_sha256',
                'calibrated'}
            assert (checkpoint['candidate'], checkpoint['arm'], checkpoint['step']) == (284, arm, step)
            assert checkpoint['config_sha256'] == sha256(RUN / 'config.json')
            assert checkpoint['parent_checkpoint_sha256'] == config['parent_checkpoint_sha256']
            assert checkpoint['initial_mapper_sha256'] == receipt['initial_mapper_sha256'][arm]
            assert checkpoint['calibrated'] is False
            expected_keys = {name for name in model.state_dict()
                             if name.split('.')[0] in MAP_MODULES}
            assert set(checkpoint['mapper']) == expected_keys
            assert all(not value.is_floating_point() or torch.isfinite(value).all()
                       for value in checkpoint['mapper'].values())
            for name in MAP_MODULES:
                prefix = name + '.'
                getattr(model, name).load_state_dict({key[len(prefix):]: value
                    for key, value in checkpoint['mapper'].items() if key.startswith(prefix)},
                    strict=True)
            states = checkpoint['optimizer']['state']
            if step == 0:
                assert mapper_digest(model) == receipt['initial_mapper_sha256'][arm]
                assert not states
            else:
                assert states and all(int(value['step']) == step for value in states.values())
            groups = checkpoint['optimizer']['param_groups']
            assert len(groups) == 1 and len(groups[0]['params']) == len(trainable)
            assert groups[0]['lr'] == 1e-4 and groups[0]['weight_decay'] == 1e-4
            rows = readouts[readout_position:readout_position + 64]
            readout_position += 64
            assert [(row['arm'], row['step'], row['physical_section_id']) for row in rows] == [
                (arm, step, item['physical_section_id']) for item in panel]
            measured_rows = []
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
                    for key in ('valid_fraction', 'visible_fraction', 'obliquity_deg'):
                        close(row[key], item[key])
                    measured = rerender(model, context['atlas'], item)
                    for key, value in measured.items():
                        close(row[key], value, 1e-4)
                    assert measured['postlimit_derivative_bound'] <= LIMIT + 1e-5
                    close(measured['prelimit_derivative_bound'] * measured['deformation_rescale'],
                          measured['postlimit_derivative_bound'], 1e-5)
                    assert measured['gain_mm'] <= \
                        measured['surviving_site_displacement_mean_mm'] + 1e-5
                    measured_rows.append({**measured, 'base_index': item['base_index'],
                        'sparse_support_under_15pct_visible': item['visible_fraction'] < .15,
                        'event_flags': item['event_flags']})
            calculated = summarize(measured_rows)
            compare_tree(receipt['readouts'][arm][str(step)], calculated)
            all_summaries[arm][step] = calculated
        del model
        torch.cuda.empty_cache()
    assert readout_position == len(readouts)

    baseline = all_summaries['sparse128'][0]['same_grid_no_warp_mean_mm']
    for arm in ARMS:
        for step in STEPS:
            close(all_summaries[arm][step]['same_grid_no_warp_mean_mm'], baseline)
    sparse = all_summaries['sparse128'][UPDATES]
    dense = all_summaries['dense2048'][UPDATES]
    advantage = dense['gain_mean_mm'] - sparse['gain_mean_mm']
    close(receipt['step_3000_dense_minus_sparse_gain_mm'], advantage)
    gate = {'dense_gain_at_least_0_02mm': dense['gain_mean_mm'] >= .02,
        'dense_at_least_six_nonnegative_bases': dense['nonnegative_gain_bases'] >= 6,
        'dense_derivative_at_most_0_35':
            dense['max_postlimit_derivative_bound'] <= LIMIT + 1e-5,
        'dense_advantage_over_sparse_at_least_0_01mm': advantage >= .01,
        'all_arms_derivative_at_most_0_35': all(
            all_summaries[arm][step]['max_postlimit_derivative_bound'] <= LIMIT + 1e-5
            for arm in ARMS for step in STEPS)}
    assert receipt['gate'] == gate
    assert receipt['dense_supervision_signal'] == all(gate[key] for key in (
        'dense_gain_at_least_0_02mm', 'dense_at_least_six_nonnegative_bases',
        'dense_derivative_at_most_0_35',
        'dense_advantage_over_sparse_at_least_0_01mm'))
    assert math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0
    print(json.dumps({'audit': 'pass', 'candidate': 284, 'runner_commit': COMMIT,
        'panel_sections': len(panel), 'panel_attempts': len(draws),
        'training_sections': len(schedule), 'training_attempts': len(training_draws),
        'step_3000_gain_mm': {arm: all_summaries[arm][UPDATES]['gain_mean_mm']
                              for arm in ARMS},
        'step_3000_dense_minus_sparse_gain_mm': advantage,
        'gate': gate, 'dense_supervision_signal': receipt['dense_supervision_signal'],
        'optimizer_trajectory_replayed': False}, allow_nan=False))


if __name__ == '__main__':
    main()
