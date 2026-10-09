"""Paired 115/117 readout on the frozen 32 virtual-oblique TRAIN-subset cuts."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
runs = {'115': root / 'runs/v3_high_virtual_oblique_115',
    '117': root / 'runs/v3_virtual_tissue_pose_117'}
frozen = root / 'runs/v3_high_virtual_oblique_116_direct_atlas_eval'
out = root / 'runs/v3_virtual_tissue_pose_117_eval'
steps, side = (0, 500, 1000, 1306), 256
torch.set_num_threads(4)
assert not out.exists()

done = {name: json.loads((run / 'completed.json').read_text()) for name, run in runs.items()}
config = {name: json.loads((run / 'config.json').read_text()) for name, run in runs.items()}
for name, run in runs.items():
    assert done[name]['batches'] == config[name]['batches'] == 1306
    assert sha(run / 'config.json') == done[name]['config_sha256']
    assert sha(run / 'draws.jsonl') == done[name]['draws_sha256']
    assert sha(run / 'training.jsonl') == done[name]['training_sha256']
    assert all(sha(run / f'joint_step_{step:05d}.pt') ==
        done[name]['checkpoint_sha256'][str(step)] for step in steps)
    assert all(sha(source / file) == digest for file, digest in
        config[name]['source_sha256'].items())
    assert not any(done[name][key] for key in ('calibrated', 'public_benchmark_used',
        'expert_real_truth_used', 'external_pretrained_weights_used'))
assert done['115']['draws_sha256'] == done['117']['draws_sha256']
assert (runs['115'] / 'draws.jsonl').read_bytes() == (runs['117'] / 'draws.jsonl').read_bytes()
assert config['117']['comparison_115_completed_sha256'] == sha(runs['115'] / 'completed.json')
assert done['117']['comparison_115_completed_sha256'] == config['117']['comparison_115_completed_sha256']
assert config['117']['expected_115_draws_sha256'] == done['115']['draws_sha256']
assert config['117']['parent_checkpoint_sha256'] == config['115']['parent_checkpoint_sha256']
assert config['117']['teacher_checkpoint_sha256'] == config['115']['teacher_checkpoint_sha256']
assert sha(source.parent / 'docs/publication/V3_VIRTUAL_TISSUE_POSE_117_PROTOCOL_20261009.md') == config['117']['protocol_sha256']
for key in ('seed', 'side', 'virtual_train_animal_ids', 'virtual_holdout_animal_ids',
    'virtual_holdout_seeds', 'virtual_source_audit_sha256', 'synthetic_provenance',
    'coronal_bindings', 'sagittal_output_sha256'):
    assert config['117'][key] == config['115'][key], key
assert done['117']['source_sha256'] == config['117']['source_sha256']
assert done['115']['source_sha256'] == config['115']['source_sha256']
assert not set(config['117']['virtual_holdout_animal_ids']) & set(config['117']['virtual_train_animal_ids'])

frozen_done = json.loads((frozen / 'completed.json').read_text())
frozen_config = json.loads((frozen / 'config.json').read_text())
assert frozen_done['rows'] == 96 and frozen_done['sections'] == 32
assert sha(frozen / 'config.json') == frozen_done['config_sha256']
assert sha(frozen / 'rows.jsonl') == frozen_done['rows_sha256']
assert sha(frozen / 'summary.json') == frozen_done['summary_sha256']
assert sha(source / 'evaluate_v3_direct_atlas_116.py') == frozen_done['evaluator_sha256']
assert frozen_config['run_completion_sha256'] == sha(runs['115'] / 'completed.json')
assert frozen_config['run_config_sha256'] == done['115']['config_sha256']
assert frozen_config['checkpoint_sha256'] == done['115']['checkpoint_sha256']['1306']
assert not any(frozen_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used'))
frozen_rows = [json.loads(line) for line in (frozen / 'rows.jsonl').open()]
baseline = {row['section_id']: row for row in frozen_rows if row['source'] == 'donor_virtual'}
assert len(baseline) == 32

train = load_reserved_real_train()
assert train['manifest_sha256'] == frozen_config['virtual_source_manifest_sha256']
assert train['bindings'] == config['115']['coronal_bindings']
by_id = {str(donor['animal_id']): i for i, donor in enumerate(train['donors'])}
items, attempts = [], []
with torch.inference_mode():
    for animal_id in config['115']['virtual_holdout_animal_ids']:
        index = by_id[str(animal_id)]
        for base_seed in config['115']['virtual_holdout_seeds']:
            seed = int(base_seed)
            while True:
                sample = sample_reserved_real_oblique_v3(train, index, seed,
                    side=side, device='cuda')
                record = sample['provenance'][0]
                used = bool(sample['eligible'][0])
                attempts.append({'animal_id': animal_id, 'seed': seed,
                    'used': used, 'section_id': record['section_id']})
                if used:
                    break
                seed += 10000
            old = baseline[record['section_id']]
            image = sample['inputs'][0].cpu().numpy().copy()
            assert hashlib.sha256(image.tobytes()).hexdigest() == old['input_sha256']
            assert old['animal_id'] == animal_id and old['seed'] == seed
            valid16 = (F.interpolate(sample['valid_mask'][:, None].float(),
                (16, 16), mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
            assert bool(valid16.any())
            assert abs(float(valid16.float().mean()) - old['donor_valid_tissue_fraction16']) < 1e-7
            items.append({'animal_id': animal_id, 'section_id': record['section_id'],
                'image': image, 'state': sample['state'][0].cpu().numpy().copy(),
                'reflection': int(sample['reflection'][0]),
                'valid16': valid16.cpu().numpy().copy(), 'input_sha256': old['input_sha256']})
assert len(items) == 32 and attempts == frozen_config['virtual_attempts']
assert [item['section_id'] for item in items] == [row['section_id'] for row in frozen_rows
    if row['source'] == 'donor_virtual']

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(-1, 2)
flags = torch.tensor([0, 1], device='cuda')[None, None]
metrics = ('selected_error_mm', 'best14_error_mm', 'best160_oracle_error_mm',
    'tissue_selected_error_mm', 'tissue_best14_error_mm', 'tissue_best160_oracle_error_mm')
rows = []
with torch.inference_mode():
    for name, run in runs.items():
        for step in steps:
            checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
                map_location='cpu', weights_only=True)
            assert checkpoint['step'] == step and json.loads(json.dumps(checkpoint['config'])) == config[name]
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            for item in items:
                image = torch.from_numpy(item['image'][None]).cuda()
                state = torch.from_numpy(item['state'][None]).cuda()
                reflection = torch.tensor([item['reflection']], device='cuda')
                valid16 = torch.from_numpy(item['valid16']).cuda()
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(1, prediction['state'].shape[1], 2)
                five_error = ((rigid_points_090(states, reflections, five)
                    - rigid_points_090(state, reflection, five)[:, None, None])
                    .norm(dim=-1).mean(-1).flatten() / 1000)
                tissue_error = ((rigid_points_090(states, reflections, chart16)
                    - rigid_points_090(state, reflection, chart16)[:, None, None])
                    .norm(dim=-1)[..., valid16].mean(-1).flatten() / 1000)
                selected = int(prior[0].argmax())
                best14 = int(beam[five_error[beam].argmin()])
                best160 = int(five_error.argmin())
                tissue_best14 = int(beam[tissue_error[beam].argmin()])
                tissue_best160 = int(tissue_error.argmin())
                assert five_error.numel() == tissue_error.numel() == 160
                assert beam.numel() == 14 and selected in beam.tolist()
                row = {'model': name, 'step': step, 'animal_id': item['animal_id'],
                    'section_id': item['section_id'], 'input_sha256': item['input_sha256'],
                    'selected_branch': selected, 'best14_branch': best14,
                    'best160_branch': best160, 'tissue_best14_branch': tissue_best14,
                    'tissue_best160_branch': tissue_best160,
                    'selected_error_mm': float(five_error[selected]),
                    'best14_error_mm': float(five_error[best14]),
                    'best160_oracle_error_mm': float(five_error[best160]),
                    'tissue_selected_error_mm': float(tissue_error[selected]),
                    'tissue_best14_error_mm': float(tissue_error[tissue_best14]),
                    'tissue_best160_oracle_error_mm': float(tissue_error[tissue_best160])}
                if name == '115' and step == 1306:
                    old = baseline[item['section_id']]
                    for key in ('selected_branch', 'best14_branch', 'best160_branch',
                        'tissue_best14_branch', 'tissue_best160_branch'):
                        assert row[key] == old[key]
                    for key in metrics:
                        assert abs(row[key] - old[key]) < 1e-4
                rows.append(row)
            print(json.dumps({'model': name, 'step': step, 'sections': len(items)}), flush=True)
assert len(rows) == 2 * len(steps) * len(items)

donors = sorted({item['animal_id'] for item in items})
assert len(donors) == 8
by_section = {(row['model'], row['step'], row['section_id']): row for row in rows}
summary = {'role': 'paired virtual-tissue pose-cost diagnostic on 32 frozen TRAIN-subset cuts',
    'caveat': 'Non-final weak-Allen-affine serial reslices; no acquired-oblique truth, independent-animal final test, calibration or GUI selection.',
    'sections': len(items), 'donors': len(donors), 'models': {}, 'paired_117_minus_115': {}}
for name in runs:
    summary['models'][name] = {}
    for step in steps:
        group = [row for row in rows if row['model'] == name and row['step'] == step]
        summary['models'][name][str(step)] = {'donor_equal': {metric: float(np.mean([
            np.mean([row[metric] for row in group if row['animal_id'] == donor])
            for donor in donors])) for metric in metrics}}
for step in steps:
    group = [row for row in rows if row['model'] == '117' and row['step'] == step]
    summary['paired_117_minus_115'][str(step)] = {'donor_equal': {metric: float(np.mean([
        np.mean([row[metric] - by_section['115', step, row['section_id']][metric]
            for row in group if row['animal_id'] == donor]) for donor in donors]))
        for metric in metrics}}

output_config = {'steps': steps,
    'run_completion_sha256': {name: sha(run / 'completed.json') for name, run in runs.items()},
    'run_config_sha256': {name: done[name]['config_sha256'] for name in runs},
    'run_draws_sha256': done['117']['draws_sha256'],
    'checkpoint_sha256': {name: done[name]['checkpoint_sha256'] for name in runs},
    'frozen_116_completion_sha256': sha(frozen / 'completed.json'),
    'frozen_116_rows_sha256': frozen_done['rows_sha256'],
    'virtual_source_manifest_sha256': train['manifest_sha256'],
    'virtual_attempts': attempts, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'sections': len(items), 'donors': len(donors),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
