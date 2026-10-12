"""Read-only post-exit audit of the bounded 270 pose-capture continuation."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
PARENT = ROOT / 'runs/pose_capture_268_v8_train_inner_002'
RUN = ROOT / 'runs/pose_capture_270_v9_continuation_001'
DOCUMENT = SOURCE.parent / 'docs/publication/POSE_CAPTURE_270_PROTOCOL_20261012.md'
STEPS = (0, 2000, 4000, 8000)
PANELS = ('fresh_v9', 'old_v8')
REPLAY_CASES = (0, 18, 36, 54)
PARENT_CHECKPOINT_SHA256 = 'bbbe06174064d95e426b02bd9106dcece1ee16eb851c76f2238229bce885d065'
sys.path.insert(0, str(SOURCE.parent))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def close(actual, expected, tolerance=1e-6):
    return math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)


# Do not invoke this script until the 270 trainer has exited and completed.json is final.
done = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf8'))
parent_done = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf8'))
assert RUN.name == 'pose_capture_270_v9_continuation_001'
assert PARENT.name == 'pose_capture_268_v8_train_inner_002'
assert parent_done['candidate'] == 268 and parent_done['selected_step'] == 8000
assert parent_done['checkpoint_sha256']['8000'] == PARENT_CHECKPOINT_SHA256
assert sha(PARENT / 'pose_step_08000.pt') == PARENT_CHECKPOINT_SHA256
assert sha(PARENT / 'completed.json') == config['parent_receipt_sha256']
assert sha(PARENT / 'inner_panel.pt') == parent_done['inner_panel_sha256'] == config['parent_inner_panel_sha256']
assert sha(PARENT / 'config.json') == parent_done['config_sha256']
assert sha(PARENT / 'protocol.json') == parent_done['protocol_sha256']
assert sha(PARENT / 'inner_draws.jsonl') == parent_done['inner_draws_sha256']
assert sha(PARENT / 'inner_scores.jsonl') == parent_done['inner_scores_sha256']
assert config['parent_checkpoint_sha256'] == PARENT_CHECKPOINT_SHA256
assert parent_config['source_sha256'] == parent_done['source_sha256']
assert parent_config['protocol'] == json.loads((PARENT / 'protocol.json').read_text(encoding='utf8'))
for name, suffix in (('protocol', '.json'), ('config', '.json'), ('inner_panel', '.pt'),
                     ('inner_draws', '.jsonl'), ('draws', '.jsonl'),
                     ('training', '.jsonl'), ('inner_scores', '.jsonl')):
    assert sha(RUN / f'{name}{suffix}') == done[f'{name}_sha256'], name
assert config['protocol'] == protocol
assert config['protocol_document_sha256'] == sha(DOCUMENT)
assert config['protocol_sha256'] == done['protocol_sha256']
assert config['source_sha256'] == done['source_sha256']
assert list(config['source_sha256']) == [
    'train_pose_capture_270.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    *parent_config['source_sha256']]
for name, digest in config['source_sha256'].items():
    assert sha(SOURCE / name) == digest, name
for step in STEPS:
    assert sha(RUN / f'pose_step_{step:05d}.pt') == done['checkpoint_sha256'][str(step)]
assert set(done['checkpoint_sha256']) == {str(step) for step in STEPS}
assert protocol['candidate'] == done['candidate'] == 270 and protocol['parent'] == 268
assert protocol['seed'] == 20261012270 and protocol['side'] == 192
assert protocol['batch'] == 4 and protocol['added_optimizer_batches'] == 8000
assert protocol['readouts'] == list(STEPS)
assert protocol['gradient_base_indices'] == list(range(56))
assert protocol['inner_base_indices'] == list(range(56, 64))
assert protocol['inner_eligible_per_base'] == 8 and protocol['inner_site_limit'] == 1024
assert not any(protocol[key] for key in ('calibrated',
    'external_prior_weights_features_pseudolabels_used', 'acquired_images_used',
    'dev_cal_final_or_public_benchmark_used'))
assert not any(done[key] for key in ('calibrated', 'acquired_images_used',
                                    'dev_cal_final_or_public_benchmark_used'))
base_subjects = config['synthetic_provenance']['base_subjects']
assert len(base_subjects) == 64
assert all(base['lineage']['split'] == 'train' for base in base_subjects)
assert config['synthetic_provenance'] == parent_config['synthetic_provenance']

fresh_panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
old_panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(fresh_panel) == len(old_panel) == 64
panel_draws = rows(RUN / 'inner_draws.jsonl')
assert len(panel_draws) == done['panel_attempted_draws']
accepted_panel = []
for attempt, draw in enumerate(panel_draws, 1):
    base, slot = 56 + len(accepted_panel) // 8, len(accepted_panel) % 8
    assert (draw['base_index'], draw['slot']) == (base, slot)
    assert 0 <= draw['variant'] < 64
    assert draw['subject_index'] == 64 * base + draw['variant']
    assert draw['draw_seed'] == protocol['seed'] + 1_000_000_000 + attempt
    provenance = draw['provenance']
    assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
    assert provenance['virtual_index'] == draw['subject_index']
    assert provenance['physical_section_id'] == (
        f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{draw['subject_index']}")
    assert not provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
    v9 = provenance['one_shot_slide_artifacts_v9']
    assert v9['source_mode'] == provenance['mode']
    assert v9['sampling'] == 'one independent physical plane per row; no appearance pairing'
    assert (v9['exterior_style'] == 'exact_black_brush') == v9['brush_available']
    if draw['used']:
        accepted_panel.append(draw)
assert len(accepted_panel) == 64

fresh_ids = set()
for index, (item, draw) in enumerate(zip(fresh_panel, accepted_panel, strict=True)):
    base, slot = 56 + index // 8, index % 8
    provenance = draw['provenance']
    assert (item['base_index'], item['slot']) == (base, slot)
    assert item['physical_section_id'] == provenance['physical_section_id']
    assert json.loads(json.dumps(item['provenance'])) == provenance
    assert item['source_mode'] == provenance['mode']
    assert item['exterior_style'] == provenance['one_shot_slide_artifacts_v9']['exterior_style']
    assert item['inputs'].shape == (5, 192, 192) and item['state'].shape == (12,)
    assert item['reflection'].numel() == 1 and item['chart'].shape[-1] == 2
    assert 1 <= len(item['chart']) <= 1024
    assert torch.isfinite(item['inputs']).all() and torch.isfinite(item['state']).all()
    assert torch.isfinite(item['chart']).all()
    pixels = item['chart'] * 192
    assert torch.allclose(pixels, pixels.round(), atol=1e-5)
    assert ((pixels >= 0) & (pixels <= 191)).all()
    assert 0 < item['valid_fraction'] <= 1 and 0 <= item['visible_fraction'] <= 1
    assert item['nearest_axis'] in ('AP', 'DV', 'ML') and 0 <= item['obliquity_deg'] <= 90
    fresh_ids.add(item['physical_section_id'])
assert len(fresh_ids) == 64
old_ids = set()
old_draws = rows(PARENT / 'inner_draws.jsonl')
assert len(old_draws) == parent_done['panel_attempted_draws']
old_accepted = []
for attempt, draw in enumerate(old_draws, 1):
    base, slot = 56 + len(old_accepted) // 8, len(old_accepted) % 8
    assert (draw['base_index'], draw['slot']) == (base, slot)
    assert 0 <= draw['variant'] < 64
    assert draw['subject_index'] == 64 * base + draw['variant']
    assert draw['draw_seed'] == 20261012268 + 1_000_000_000 + attempt
    provenance = draw['provenance']
    assert provenance['virtual_index'] == draw['subject_index']
    assert provenance['physical_section_id'] == (
        f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{draw['subject_index']}")
    assert not provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
    assert provenance['one_shot_slide_artifacts_v8']['sampling'].endswith('no appearance pairing')
    if draw['used']:
        old_accepted.append(draw)
assert len(old_accepted) == 64
for index, (item, draw) in enumerate(zip(old_panel, old_accepted, strict=True)):
    assert (item['base_index'], item['slot']) == (56 + index // 8, index % 8)
    assert (draw['base_index'], draw['slot']) == (item['base_index'], item['slot'])
    assert item['physical_section_id'] == draw['provenance']['physical_section_id']
    assert json.loads(json.dumps(item['provenance'])) == draw['provenance']
    assert item['provenance']['split'] == item['provenance']['base_lineage']['split'] == 'train'
    assert item['physical_section_id'] == item['provenance']['physical_section_id']
    assert item['exterior_style'] == item['provenance']['one_shot_slide_artifacts_v8']['exterior_style']
    assert item['inputs'].shape == (5, 192, 192) and item['state'].shape == (12,)
    assert item['reflection'].numel() == 1 and item['chart'].shape[-1] == 2
    assert 1 <= len(item['chart']) <= 1024
    old_ids.add(item['physical_section_id'])
assert len(old_ids) == 64 and not (fresh_ids & old_ids)

training = rows(RUN / 'training.jsonl')
assert len(training) == done['added_optimizer_batches'] == 8000
for step, row in enumerate(training, 1):
    assert row['step'] == step and row['batch_size'] == 4
    assert row['accepted'] == 4 * step
    assert all(math.isfinite(row[key]) for key in
               ('mean_valid_fraction', 'direct_loss', 'gradient_norm', 'elapsed_seconds'))
    assert 0 < row['mean_valid_fraction'] <= 1

attempts_by_step = [0] * 8001
used_by_step = [0] * 8001
accepted_ids = set()
previous_step = 1
attempt = 0
with (RUN / 'draws.jsonl').open(encoding='utf8') as stream:
    for attempt, line in enumerate(stream, 1):
        draw = json.loads(line)
        step, base, variant = draw['step'], draw['base_index'], draw['variant']
        assert previous_step <= step <= 8000 and draw['attempt'] == attempt
        previous_step = step
        assert 0 <= base < 56 and 0 <= variant < 64
        assert draw['subject_index'] == 64 * base + variant
        assert draw['draw_seed'] == protocol['seed'] + 2_000_000_000 + attempt
        provenance = draw['provenance']
        assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
        assert provenance['virtual_index'] == draw['subject_index']
        assert draw['physical_section_id'] == provenance['physical_section_id']
        assert draw['physical_section_id'] == (
            f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{draw['subject_index']}")
        assert not provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
        v9 = provenance['one_shot_slide_artifacts_v9']
        assert v9['source_mode'] == provenance['mode']
        assert v9['sampling'] == 'one independent physical plane per row; no appearance pairing'
        assert (v9['exterior_style'] == 'exact_black_brush') == v9['brush_available']
        assert 0 <= draw['valid_fraction'] <= 1
        attempts_by_step[step] += 1
        if draw['used']:
            assert draw['physical_section_id'] not in accepted_ids
            assert draw['physical_section_id'] not in fresh_ids | old_ids
            accepted_ids.add(draw['physical_section_id'])
            used_by_step[step] += 1
assert attempt == done['attempted_draws']
assert len(accepted_ids) == done['accepted_presentations'] == 32000
cumulative_attempts = 0
for step in range(1, 8001):
    cumulative_attempts += attempts_by_step[step]
    assert used_by_step[step] == 4
    assert training[step - 1]['attempted'] == cumulative_attempts
assert cumulative_attempts == done['attempted_draws']

score_rows = rows(RUN / 'inner_scores.jsonl')
assert len(score_rows) == 2 * len(STEPS)
assert [(score['step'], score['panel']) for score in score_rows] == [
    (step, panel_name) for step in STEPS for panel_name in PANELS]
readouts = {}
for score in score_rows:
    step, panel_name = score['step'], score['panel']
    panel = fresh_panel if panel_name == 'fresh_v9' else old_panel
    metrics, cases = score['metrics'], score['rows']
    assert len(cases) == metrics['n'] == 64
    for item, case in zip(panel, cases, strict=True):
        for key in ('base_index', 'slot', 'physical_section_id', 'valid_fraction',
                    'visible_fraction', 'obliquity_deg', 'nearest_axis',
                    'exterior_style', 'source_mode'):
            assert case[key] == item[key], (step, panel_name, key)
        assert all(math.isfinite(case[key]) for key in ('oracle_mm', 'top1_mm',
            'normal_axis_error_deg', 'selected_minor_fov_mm', 'true_minor_fov_mm',
            'minor_fov_ratio'))
        assert 0 <= case['oracle_mm'] <= case['top1_mm']
        assert 0 <= case['normal_axis_error_deg'] <= 90
        assert case['selected_minor_fov_mm'] > 0 and case['true_minor_fov_mm'] > 0
        assert close(case['minor_fov_ratio'],
                     case['selected_minor_fov_mm'] / case['true_minor_fov_mm'])
        assert not case['top2_hit'] or case['top16_hit']
        assert not case['top16_hit'] or case['oracle_hit']
        assert case['oracle_hit'] == (case['oracle_mm'] <= 1.5)
    for key in ('top16', 'top2', 'oracle'):
        assert metrics[f'{key}_hits'] == sum(case[f'{key}_hit'] for case in cases)
    assert close(metrics['mean_top1_mm'], np.mean([case['top1_mm'] for case in cases]))
    assert close(metrics['mean_normal_error_deg'], np.mean([
        case['normal_axis_error_deg'] for case in cases]))
    assert close(metrics['median_minor_fov_ratio'],
                 np.median([case['minor_fov_ratio'] for case in cases]))
    assert metrics['selected_minor_fov_le_1.2mm'] == sum(
        case['selected_minor_fov_mm'] <= 1.2 for case in cases)
    assert set(metrics['base']) == {str(base) for base in range(56, 64)}
    for base in range(56, 64):
        group = [case for case in cases if case['base_index'] == base]
        reported = metrics['base'][str(base)]
        assert len(group) == 8 and reported['top16_hits'] == sum(case['top16_hit'] for case in group)
        assert close(reported['mean_top1_mm'], np.mean([case['top1_mm'] for case in group]))
    assert close(metrics['equal_base_top1_mm'], np.mean([
        metrics['base'][str(base)]['mean_top1_mm'] for base in range(56, 64)]))
    strata = {
        'visible_lt_0.15': [case for case in cases if case['visible_fraction'] < .15],
        'visible_0.15_to_0.40': [case for case in cases if .15 <= case['visible_fraction'] < .40],
        'visible_ge_0.40': [case for case in cases if case['visible_fraction'] >= .40],
        'axis_le_20deg': [case for case in cases if case['obliquity_deg'] <= 20],
        'axis_20_to_45deg': [case for case in cases if 20 < case['obliquity_deg'] <= 45],
        'axis_gt_45deg': [case for case in cases if case['obliquity_deg'] > 45],
    }
    assert set(metrics['strata']) == set(strata)
    for label, group in strata.items():
        reported = metrics['strata'][label]
        assert reported['n'] == len(group)
        assert reported['top16_hits'] == sum(case['top16_hit'] for case in group)
        expected = float(np.mean([case['top1_mm'] for case in group])) if group else None
        assert reported['mean_top1_mm'] is None if expected is None else close(
            reported['mean_top1_mm'], expected)
    readouts[step, panel_name] = metrics

parent_scores = rows(PARENT / 'inner_scores.jsonl')
assert [row['step'] for row in parent_scores] == list(STEPS)
for original, retained in zip(parent_scores[-1]['rows'], score_rows[1]['rows'], strict=True):
    assert original['physical_section_id'] == retained['physical_section_id']
    for key in ('top16_hit', 'top2_hit', 'oracle_hit'):
        assert original[key] == retained[key]
    for key in ('oracle_mm', 'top1_mm', 'normal_axis_error_deg',
                'selected_minor_fov_mm', 'true_minor_fov_mm', 'minor_fov_ratio'):
        assert close(original[key], retained[key], 1e-3)

best = max(STEPS[1:], key=lambda step: (
    readouts[step, 'fresh_v9']['top16_hits'],
    -readouts[step, 'fresh_v9']['equal_base_top1_mm']))
fresh, old = readouts[best, 'fresh_v9'], readouts[best, 'old_v8']
gate = {
    'fresh_top16_at_least_45': fresh['top16_hits'] >= 45,
    'fresh_top2_at_least_16': fresh['top2_hits'] >= 16,
    'fresh_mean_top1_at_most_3mm': fresh['mean_top1_mm'] <= 3,
    'fresh_median_minor_fov_ratio_at_least_0.5': fresh['median_minor_fov_ratio'] >= .5,
    'fresh_at_most_four_minor_fov_le_1.2mm': fresh['selected_minor_fov_le_1.2mm'] <= 4,
    'old_top16_at_least_40': old['top16_hits'] >= 40,
}
assert done['selected_step'] == best and done['gate'] == gate
assert done['gate_pass'] == all(gate.values())

# Check inherited weights and AdamW moments before replaying saved images.
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_checkpoint = torch.load(PARENT / 'pose_step_08000.pt', map_location='cpu', weights_only=False)
assert parent_checkpoint['step'] == 8000
assert json.loads(json.dumps(parent_checkpoint['config'])) == parent_config
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).to(device).eval()
for step in STEPS:
    checkpoint = torch.load(RUN / f'pose_step_{step:05d}.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == step
    assert json.loads(json.dumps(checkpoint['config'])) == config
    assert checkpoint['accepted'] == 4 * step and not checkpoint['calibrated']
    assert checkpoint['attempted'] == (training[step - 1]['attempted'] if step else 0)
    if step == 0:
        assert checkpoint['model'].keys() == parent_checkpoint['model'].keys()
        assert all(torch.equal(value, parent_checkpoint['model'][name])
                   for name, value in checkpoint['model'].items())
        assert checkpoint['optimizer']['param_groups'] == parent_checkpoint['optimizer']['param_groups']
        assert checkpoint['optimizer']['state'].keys() == parent_checkpoint['optimizer']['state'].keys()
        for index, state in checkpoint['optimizer']['state'].items():
            for key, value in state.items():
                original = parent_checkpoint['optimizer']['state'][index][key]
                if torch.is_tensor(value):
                    assert torch.equal(value, original)
                else:
                    assert value == original
    else:
        assert all(torch.equal(value, parent_checkpoint['model'][name]) for name, value in
                   checkpoint['model'].items() if not name.startswith(
                       ('encoder.', 'lateral.', 'pose.', 'anchor_pose.', 'anchor_global.')))
        assert close(checkpoint['optimizer']['param_groups'][0]['lr'],
                     5e-5 if step <= 4000 else 3e-5)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    with torch.inference_mode():
        for panel_name, panel in (('fresh_v9', fresh_panel), ('old_v8', old_panel)):
            reported_cases = next(score['rows'] for score in score_rows
                                  if score['step'] == step and score['panel'] == panel_name)
            for index in REPLAY_CASES:
                item, reported = panel[index], reported_cases[index]
                image = item['inputs'][None].to(device)
                state = item['state'][None].to(device)
                reflection = item['reflection'][None].to(device)
                chart = item['chart'].to(device)
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                truth = rigid_points_090(state, reflection, chart, (192, 192))
                errors = []
                for start in range(0, 80, 8):
                    candidate = prediction['state'][:, start:start + 8, None].expand(-1, -1, 2, -1)
                    flags = torch.arange(2, device=device)[None, None].expand(1, 8, 2)
                    points = rigid_points_090(candidate, flags, chart[None, None], (192, 192))
                    errors.append(((points - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten())
                errors = torch.cat(errors)
                order = prior.argsort(descending=True)
                top = int(order[0])
                selected = prediction['state'][0, top // 2]
                true_normal = full_frame_state_to_components(state)[1][0, :, 2]
                selected_normal = full_frame_state_to_components(selected)[1][:, 2]
                selected_basis = full_frame_state_to_components(selected)[2]
                true_basis = full_frame_state_to_components(state)[2][0]
                normal_angle = float(torch.rad2deg(torch.acos(
                    (selected_normal * true_normal).sum().abs().clamp(0, 1))))
                minor = float(torch.linalg.svdvals(selected_basis).min() / 1000)
                true_minor = float(torch.linalg.svdvals(true_basis).min() / 1000)
                assert close(float(errors.min()), reported['oracle_mm'], 1e-3)
                assert close(float(errors[top]), reported['top1_mm'], 1e-3)
                assert bool((errors[order[:16]] <= 1.5).any()) == reported['top16_hit']
                assert bool((errors[order[:2]] <= 1.5).any()) == reported['top2_hit']
                assert bool((errors <= 1.5).any()) == reported['oracle_hit']
                assert close(normal_angle, reported['normal_axis_error_deg'], 1e-3)
                assert close(minor, reported['selected_minor_fov_mm'], 1e-3)
                assert close(true_minor, reported['true_minor_fov_mm'], 1e-3)
                assert close(minor / true_minor, reported['minor_fov_ratio'], 1e-3)

print(json.dumps({'verified': True, 'run': RUN.name, 'fresh_panel_cases': 64,
                  'old_panel_cases': 64, 'accepted_presentations': len(accepted_ids),
                  'replayed_cases': len(STEPS) * len(PANELS) * len(REPLAY_CASES),
                  'selected_step': best, 'gate': gate, 'gate_pass': all(gate.values())}))
