"""Read-only hash, lineage, count and gate audit of the frozen 275 run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/pose_capture_275_v9_continuation_001'
SOURCE = Path(__file__).resolve().parent
receipt = json.loads((RUN / 'completed.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
assert receipt['candidate'] == config['candidate'] == 275
for name, extension in (('config', 'json'), ('inner_panel', 'pt'),
                        ('inner_draws', 'jsonl'), ('draws', 'jsonl'),
                        ('training', 'jsonl'), ('readouts', 'jsonl')):
    with (RUN / f'{name}.{extension}').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt[f'{name}_sha256']
for step, digest in receipt['checkpoint_sha256'].items():
    with (RUN / f'pose_step_{int(step):05d}.pt').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
for name, digest in config['source_sha256'].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((SOURCE.parent / 'docs/publication/POSE_CAPTURE_275_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
for path, key in ((ROOT / 'runs/pose_style_274_train_inner_001/control_v9_step_02000.pt',
                   'parent_checkpoint_sha256'),
                  (ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
                   'panel_270_sha256'),
                  (ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt',
                   'panel_268_sha256'),
                  (ROOT / 'runs/acquired_train_transfer_272_001/cases.jsonl',
                   'transfer_cases_sha256')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == config[key]

panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
inner_draws = [json.loads(line) for line in (RUN / 'inner_draws.jsonl').open()]
draws = [json.loads(line) for line in (RUN / 'draws.jsonl').open()]
training = [json.loads(line) for line in (RUN / 'training.jsonl').open()]
readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open()]
assert len(panel) == 64 and len(inner_draws) == receipt['panel_attempts']
assert sum(row['used'] for row in inner_draws) == 64
assert all(sum(item['base_index'] == base for item in panel) == 8 for base in range(56, 64))
assert all(item['physical_section_id'] == item['provenance']['physical_section_id'] for item in panel)
assert len(training) == 8000 and [row['step'] for row in training] == list(range(1, 8001))
assert sum(row['used'] for row in draws) == receipt['accepted_presentations'] == 32000
assert len(draws) == receipt['attempted_draws']
assert [row['attempt'] for row in draws] == list(range(1, len(draws) + 1))
assert all(0 <= row['base_index'] < 56 for row in draws)
assert len(readouts) == 20
for step in (0, 2000, 4000, 8000):
    for panel_name in ('new_v9', 'frozen_270_v9', 'frozen_268_v8',
                       'coronal', 'sagittal'):
        row, = [row for row in readouts if (row['step'], row['panel']) ==
                (step, panel_name)]
        records, metrics = row['rows'], row['metrics']
        assert metrics == receipt['readouts'][str(step)][panel_name]
        assert len(records) == metrics['n'] == (64 if panel_name not in ('coronal', 'sagittal')
                                                else 28 if panel_name == 'coronal' else 22)
        if panel_name in ('coronal', 'sagittal'):
            assert metrics['first_axis_correct'] == sum(
                item['first_axis_correct'] for item in records)
            assert abs(metrics['median_weak_normal_angle_deg'] - np.median([
                item['normal_gap_to_weak_affine_deg'] for item in records])) < 1e-9
            assert all(hashlib.sha256(f'{panel_name}|{item["donor_id"]}'.encode()).digest()[0]
                       % 2 == 1 for item in records)
        else:
            assert metrics['top16_hits'] == sum(item['top16_hit'] for item in records)
            assert abs(metrics['mean_top1_mm'] - np.mean([
                item['top1_mm'] for item in records])) < 1e-9
            assert abs(metrics['mean_normal_axis_error_deg'] - np.mean([
                item['normal_axis_error_deg'] for item in records])) < 1e-9

scores = receipt['readouts']
best = max((0, 2000, 4000, 8000), key=lambda step: (
    scores[str(step)]['new_v9']['top16_hits'],
    -scores[str(step)]['new_v9']['mean_top1_mm']))
assert best == receipt['selected_step']
selected = scores[str(best)]
gate = {'new_top16_at_least_52': selected['new_v9']['top16_hits'] >= 52,
    'new_mean_top1_at_most_2.3mm': selected['new_v9']['mean_top1_mm'] <= 2.3,
    'frozen_270_top16_at_least_48': selected['frozen_270_v9']['top16_hits'] >= 48,
    'frozen_268_top16_at_least_50': selected['frozen_268_v8']['top16_hits'] >= 50,
    'acquired_coronal_at_least_26': selected['coronal']['first_axis_correct'] >= 26,
    'acquired_sagittal_at_least_16': selected['sagittal']['first_axis_correct'] >= 16}
assert gate == receipt['gate'] and all(gate.values()) == receipt['gate_pass']
panels = {'new_v9': panel,
    'frozen_270_v9': torch.load(ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
                                map_location='cpu', weights_only=False),
    'frozen_268_v8': torch.load(ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt',
                                map_location='cpu', weights_only=False)}
torch.set_num_threads(4)
replayed = 0
with torch.inference_mode():
    for step in (0, 2000, 4000, 8000):
        checkpoint = torch.load(RUN / f'pose_step_{step:05d}.pt',
                                map_location='cpu', weights_only=False)
        assert checkpoint['step'] == step
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        for panel_name, cases in panels.items():
            scored, = [row for row in readouts if (row['step'], row['panel']) ==
                       (step, panel_name)]
            for index in (0, 32):
                item = cases[index]
                prediction = model.predict(item['inputs'][None].cuda())
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                order = prior.argsort(descending=True)
                chart = item['chart'].cuda()
                truth = rigid_points_090(item['state'][None].cuda(),
                    item['reflection'][None].cuda(), chart, (192, 192))
                proposed = rigid_points_090(
                    prediction['state'][0, order[:16] // 2][None],
                    (order[:16] % 2)[None], chart[None], (192, 192))
                errors = ((proposed - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
                true_normal = full_frame_state_to_components(item['state'].cuda())[1][:, 2]
                chosen_normal = full_frame_state_to_components(
                    prediction['state'][0, order[0] // 2])[1][:, 2]
                angle = float(torch.rad2deg(torch.acos((true_normal * chosen_normal)
                                             .sum().abs().clamp(0, 1))))
                saved = scored['rows'][index]
                assert item['physical_section_id'] == saved['physical_section_id']
                assert bool((errors <= 1.5).any()) == saved['top16_hit']
                assert abs(float(errors[0]) - saved['top1_mm']) < 1e-4
                assert abs(angle - saved['normal_axis_error_deg']) < 1e-3
                replayed += 1
        del model, checkpoint
print(json.dumps({'verified': True, 'panel_cases': len(panel), 'attempted_draws': len(draws),
                  'accepted_presentations': sum(row['used'] for row in draws),
                  'optimizer_batches': len(training), 'readout_blocks': len(readouts),
                  'replayed_case_readouts': replayed,
                  'selected_step': best, 'gate': gate, 'gate_pass': all(gate.values())}))
