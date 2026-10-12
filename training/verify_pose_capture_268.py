"""Read-only post-exit audit of the completed 268 TRAIN-inner scratch run."""
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
RUN = ROOT / 'runs/pose_capture_268_v8_train_inner_002'
DOCUMENT = SOURCE.parent / 'docs/publication/POSE_CAPTURE_268_PROTOCOL_20261012.md'
STEPS = (0, 2000, 4000, 8000)
REPLAY_CASES = (0, 9, 18, 27, 36, 45, 54, 63)
sys.path.insert(0, str(SOURCE.parent))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def close(actual, expected, tolerance=1e-6):
    return math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)


# Do not invoke this script until the trainer has exited and completed.json is final.
done = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf8'))
assert RUN.name == 'pose_capture_268_v8_train_inner_002'
for name, suffix in (('protocol', '.json'), ('config', '.json'), ('inner_panel', '.pt'),
                     ('inner_draws', '.jsonl'), ('draws', '.jsonl'),
                     ('training', '.jsonl'), ('inner_scores', '.jsonl')):
    assert sha(RUN / f'{name}{suffix}') == done[f'{name}_sha256'], name
assert config['protocol'] == protocol
assert config['protocol_document_sha256'] == sha(DOCUMENT)
assert config['protocol_sha256'] == done['protocol_sha256']
assert config['source_sha256'] == done['source_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(SOURCE / name) == digest, name
for step in STEPS:
    assert sha(RUN / f'pose_step_{step:05d}.pt') == done['checkpoint_sha256'][str(step)]
assert set(done['checkpoint_sha256']) == {str(step) for step in STEPS}
assert protocol['candidate'] == done['candidate'] == 268
assert protocol['seed'] == 20261012268 and protocol['side'] == 192
assert protocol['batch'] == 4 and protocol['max_optimizer_batches'] == 8000
assert protocol['readouts'] == list(STEPS)
assert protocol['gradient_base_indices'] == list(range(56))
assert protocol['inner_base_indices'] == list(range(56, 64))
assert protocol['inner_eligible_per_base'] == 8 and protocol['inner_site_limit'] == 1024
assert protocol['stages'] == [[1, 2000, .12, 1e-4], [2001, 4000, .08, 7e-5],
                              [4001, 8000, 0., 5e-5]]
assert not any(protocol[key] for key in ('calibrated',
    'prior_model_weights_features_pseudolabels_used', 'acquired_images_used',
    'dev_cal_final_or_public_benchmark_used'))
assert not any(done[key] for key in ('calibrated', 'acquired_images_used',
                                    'dev_cal_final_or_public_benchmark_used'))
base_subjects = config['synthetic_provenance']['base_subjects']
assert len(base_subjects) == 64
assert all(base['lineage']['split'] == 'train' for base in base_subjects)

panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
panel_draws = rows(RUN / 'inner_draws.jsonl')
assert len(panel) == 64 and len(panel_draws) == done['panel_attempted_draws']
accepted_panel = []
for attempt, draw in enumerate(panel_draws, 1):
    base, slot = 56 + len(accepted_panel) // 8, len(accepted_panel) % 8
    assert (draw['base_index'], draw['slot'], draw['attempt']) == (base, slot, attempt)
    assert 0 <= draw['variant'] < 64
    assert draw['subject_index'] == 64 * base + draw['variant']
    assert draw['draw_seed'] == protocol['seed'] + 1_000_000_000 + attempt
    provenance = draw['provenance']
    assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
    assert provenance['virtual_index'] == draw['subject_index']
    assert provenance['physical_section_id'] == (
        f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{draw['subject_index']}")
    assert not provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
    assert provenance['one_shot_slide_artifacts_v8']['sampling'].endswith('no appearance pairing')
    if draw['used']:
        accepted_panel.append(draw)
assert len(accepted_panel) == 64
panel_ids = set()
for index, (item, draw) in enumerate(zip(panel, accepted_panel, strict=True)):
    base, slot = 56 + index // 8, index % 8
    provenance = draw['provenance']
    assert (item['base_index'], item['slot']) == (base, slot)
    assert item['physical_section_id'] == provenance['physical_section_id']
    assert item['provenance'] == provenance
    assert item['source_mode'] == provenance['mode']
    assert item['exterior_style'] == provenance['one_shot_slide_artifacts_v8']['exterior_style']
    assert item['inputs'].shape == (5, 192, 192) and item['state'].shape == (12,)
    assert item['reflection'].numel() == 1 and item['chart'].shape[-1] == 2
    assert 1 <= len(item['chart']) <= 1024
    assert torch.isfinite(item['inputs']).all() and torch.isfinite(item['state']).all()
    assert torch.isfinite(item['chart']).all()
    pixels = item['chart'] * 192
    assert torch.allclose(pixels, pixels.round(), atol=1e-5)
    assert ((pixels >= 0) & (pixels <= 191)).all()
    assert 0 < item['valid_fraction'] <= 1 and 0 <= item['visible_fraction'] <= 1
    assert item['nearest_axis'] in ('AP', 'DV', 'ML')
    assert 0 <= item['obliquity_deg'] <= 90
    panel_ids.add(item['physical_section_id'])
assert len(panel_ids) == 64

training = rows(RUN / 'training.jsonl')
assert len(training) == done['optimizer_batches'] == 8000
for step, row in enumerate(training, 1):
    minimum, stage = (.12, 1) if step <= 2000 else (.08, 2) if step <= 4000 else (0., 3)
    assert row['step'] == step and row['stage'] == stage
    assert row['batch_size'] == 4 and row['accepted'] == 4 * step
    assert row['mean_valid_fraction'] >= minimum
    assert all(math.isfinite(row[key]) for key in
               ('mean_valid_fraction', 'direct_loss', 'gradient_norm', 'elapsed_seconds'))

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
        assert draw['stage'] == (1 if step <= 2000 else 2 if step <= 4000 else 3)
        provenance = draw['provenance']
        assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
        assert provenance['virtual_index'] == draw['subject_index']
        assert draw['physical_section_id'] == provenance['physical_section_id']
        assert draw['physical_section_id'] == (
            f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{draw['subject_index']}")
        assert not provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
        assert provenance['one_shot_slide_artifacts_v8']['sampling'].endswith('no appearance pairing')
        assert 0 <= draw['valid_fraction'] <= 1
        attempts_by_step[step] += 1
        if draw['used']:
            assert draw['valid_fraction'] >= (.12 if step <= 2000 else .08 if step <= 4000 else 0.)
            assert draw['physical_section_id'] not in accepted_ids
            assert draw['physical_section_id'] not in panel_ids
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

readouts = {}
scores = rows(RUN / 'inner_scores.jsonl')
assert len(scores) == 4 and [score['step'] for score in scores] == list(STEPS)
for score in scores:
    step, metrics, cases = score['step'], score['metrics'], score['rows']
    assert len(cases) == metrics['n'] == 64
    for item, case in zip(panel, cases, strict=True):
        for key in ('base_index', 'slot', 'physical_section_id', 'valid_fraction',
                    'visible_fraction', 'obliquity_deg', 'nearest_axis',
                    'exterior_style', 'source_mode'):
            assert case[key] == item[key], (step, key)
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
    readouts[step] = metrics

best = max((2000, 4000, 8000), key=lambda step:
           (readouts[step]['top16_hits'], -readouts[step]['equal_base_top1_mm']))
gate = {
    'top16_at_least_45': readouts[8000]['top16_hits'] >= 45,
    'improved_in_six_bases': sum(readouts[8000]['base'][str(base)]['top16_hits'] >
        readouts[0]['base'][str(base)]['top16_hits'] for base in range(56, 64)) >= 6,
    'median_minor_fov_ratio_at_least_0.5': readouts[8000]['median_minor_fov_ratio'] >= .5,
    'at_most_four_minor_fov_le_1.2mm': readouts[8000]['selected_minor_fov_le_1.2mm'] <= 4,
}
assert done['selected_step'] == best and done['gate'] == gate
assert done['gate_pass'] == all(gate.values())

# Recompute physical errors from saved images, targets and model weights for one case per base.
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).to(device).eval()
baseline = None
for step, score in zip(STEPS, scores, strict=True):
    checkpoint = torch.load(RUN / f'pose_step_{step:05d}.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == step and checkpoint['config'] == config
    assert checkpoint['accepted'] == 4 * step and not checkpoint['calibrated']
    assert checkpoint['attempted'] == (training[step - 1]['attempted'] if step else 0)
    if step == 0:
        baseline = checkpoint['model']
    else:
        assert all(torch.equal(value, baseline[name]) for name, value in
                   checkpoint['model'].items() if not name.startswith(
                       ('encoder.', 'lateral.', 'pose.', 'anchor_pose.', 'anchor_global.')))
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    with torch.inference_mode():
        for index in REPLAY_CASES:
            item, reported = panel[index], score['rows'][index]
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

print(json.dumps({'verified': True, 'run': RUN.name, 'panel_cases': 64,
                  'accepted_presentations': len(accepted_ids), 'replayed_cases': 32,
                  'selected_step': best, 'gate': gate, 'gate_pass': all(gate.values())}))
