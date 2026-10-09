"""Paired, non-final 115/113 readout on the frozen 113 development panels."""
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
runs = {'113': root / 'runs/v3_mixed_virtual_oblique_113',
    '115': root / 'runs/v3_high_virtual_oblique_115'}
old_eval = root / 'runs/v3_mixed_virtual_oblique_113_eval'
out = root / 'runs/v3_high_virtual_oblique_115_eval'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
steps = {'113': (0, 1000), '115': (0, 500, 1000, 1306)}
side = 256
torch.set_num_threads(4)
assert not out.exists()

done = {name: json.loads((run / 'completed.json').read_text()) for name, run in runs.items()}
config = {name: json.loads((run / 'config.json').read_text()) for name, run in runs.items()}
assert done['113']['batches'] == config['113']['batches'] == 4000
assert done['115']['batches'] == config['115']['batches'] == 1306
for name, run in runs.items():
    assert sha(run / 'config.json') == done[name]['config_sha256']
    assert sha(run / 'draws.jsonl') == done[name]['draws_sha256']
    assert sha(run / 'training.jsonl') == done[name]['training_sha256']
    assert done[name]['source_sha256'] == config[name]['source_sha256']
    assert done[name]['parent_checkpoint_sha256'] == config[name]['parent_checkpoint_sha256']
    assert done[name]['teacher_checkpoint_sha256'] == config[name]['teacher_checkpoint_sha256']
    assert all(sha(run / f'joint_step_{step:05d}.pt') ==
        done[name]['checkpoint_sha256'][str(step)] for step in steps[name])
    assert not any(done[name][key] for key in ('calibrated', 'public_benchmark_used',
        'expert_real_truth_used', 'external_pretrained_weights_used'))
    assert all(sha(source / file) == digest
        for file, digest in config[name]['source_sha256'].items())
assert sha(source.parent / 'docs/publication/V3_MIXED_VIRTUAL_OBLIQUE_113_PROTOCOL_20261009.md') == config['113']['protocol_sha256']
assert sha(source.parent / 'docs/publication/V3_HIGH_VIRTUAL_OBLIQUE_115_PROTOCOL_20261009.md') == config['115']['protocol_sha256']
for key in ('seed', 'side', 'virtual_train_animal_ids', 'virtual_holdout_animal_ids',
    'virtual_holdout_seeds', 'virtual_source_audit_sha256', 'synthetic_provenance',
    'coronal_bindings', 'sagittal_output_sha256', 'parent_checkpoint_sha256',
    'teacher_checkpoint_sha256'):
    assert config['115'][key] == config['113'][key], key
assert config['115']['checkpoints'] == list(steps['115'])
assert config['115']['comparison_113_completed_sha256'] == sha(runs['113'] / 'completed.json')
assert done['115']['comparison_113_completed_sha256'] == config['115']['comparison_113_completed_sha256']
assert config['115']['expected_113_draw_prefix_sha256'] == done['115']['matched_113_draw_prefix_sha256'] == done['115']['draws_sha256']
draws_115 = (runs['115'] / 'draws.jsonl').read_bytes()
assert draws_115 and (runs['113'] / 'draws.jsonl').read_bytes().startswith(draws_115)
assert json.loads(draws_115.splitlines()[-1])['batch'] == 1306

old_done = json.loads((old_eval / 'completed.json').read_text())
assert not any(old_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used'))
assert sha(old_eval / 'rows.jsonl') == old_done['rows_sha256']
assert sha(old_eval / 'summary.json') == old_done['summary_sha256']
assert sha(source / 'evaluate_v3_mixed_virtual_oblique_113.py') == old_done['evaluator_sha256']
assert sha(runs['113'] / 'completed.json') == old_done['run_completion_sha256']
old_rows = [json.loads(line) for line in (old_eval / 'rows.jsonl').open()]
expected = [row for row in old_rows if row['step'] == 0]
assert len(expected) == 318
assert all([(row['kind'], row['animal_id'], row['specimen_id'],
    row['experiment_id'], row['section_id']) for row in old_rows if row['step'] == step]
    == [(row['kind'], row['animal_id'], row['specimen_id'],
    row['experiment_id'], row['section_id']) for row in expected]
    for step in (1000, 2000, 3000, 4000))
assert len(config['113']['virtual_holdout_animal_ids']) == 8
assert len(config['113']['virtual_train_animal_ids']) == 55
assert not set(config['113']['virtual_holdout_animal_ids']) & set(config['113']['virtual_train_animal_ids'])
draws = [json.loads(line) for line in (runs['113'] / 'draws.jsonl').open()]
assert not set(config['113']['virtual_holdout_animal_ids']) & {
    str(row['animal_id']) for row in draws if row['kind'] in
    ('coronal_weak_train', 'virtual_oblique_weak_train')}

panel_done = json.loads((panel / 'completed.json').read_text())
assert panel_done['physical_sections'] == 256 and panel_done['eligible'] == 247
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in records if row['eligible']]
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
synthetic_rows = [row for plan in plans for row in sorted((row for row in eligible
    if row['synthetic_subject_plan_id'] == plan),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(synthetic_rows) == 64 and len(plans) == 8
assert all(sha(panel / row['file']) == row['sha256'] for row in synthetic_rows)
synthetic_draws = [row for row in draws if row['kind'] == 'synthetic_train' and row['used']]
assert len(synthetic_draws) == 8000
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in records} & {row['base_lineage'][key]
        for row in synthetic_draws}
synthetic_panel = []
for row in synthetic_rows:
    with np.load(panel / row['file'], allow_pickle=False) as arrays:
        synthetic_panel.append({'image': arrays['inputs'].copy(),
            'state': arrays['target_state'].copy(),
            'reflection': int(arrays['reflection']),
            'valid': arrays['valid_mask'].copy(),
            'animal_id': row['animal_id'], 'specimen_id': row['specimen_id'],
            'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
            'synthetic_subject_plan_id': row['synthetic_subject_plan_id'],
            'panel_file_sha256': row['sha256'], 'kind': 'synthetic_dev'})

train = load_reserved_real_train()
assert train['manifest_sha256'] == old_done['source_manifest_sha256']
assert train['bindings'] == config['113']['coronal_bindings']
by_id = {str(donor['animal_id']): i for i, donor in enumerate(train['donors'])}
virtual_panel, virtual_attempts = [], []
with torch.inference_mode():
    for animal_id in config['113']['virtual_holdout_animal_ids']:
        index = by_id[str(animal_id)]
        for base_seed in config['113']['virtual_holdout_seeds']:
            seed = int(base_seed)
            while True:
                sample = sample_reserved_real_oblique_v3(train, index, seed,
                    side=side, device='cuda')
                used = bool(sample['eligible'][0])
                provenance = sample['provenance'][0]
                virtual_attempts.append({'animal_id': animal_id, 'seed': seed,
                    'used': used, 'section_id': provenance['section_id']})
                if used:
                    break
                seed += 10000
            virtual_panel.append({'image': sample['inputs'][0].cpu().numpy(),
                'state': sample['state'][0].cpu().numpy(),
                'reflection': int(sample['reflection'][0]),
                'valid': sample['valid_mask'][0].cpu().numpy(),
                'animal_id': animal_id, 'specimen_id': provenance['specimen_id'],
                'experiment_id': provenance['experiment_id'],
                'section_id': provenance['section_id'],
                'source_images_sha256': provenance['real_oblique_v3']['source_images_sha256'],
                'source_records_sha256': provenance['real_oblique_v3']['source_records_sha256'],
                'kind': 'virtual_oblique_train_holdout'})
assert len(virtual_panel) == 32 and virtual_attempts == old_done['virtual_attempts']

coronal_done = json.loads((coronal / 'completed.json').read_text())
for file in ('images.npy', 'geometry.npz', 'records.jsonl'):
    assert sha(coronal / file) == coronal_done['output_sha256'][file]
coronal_records = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
    if row['training_split'] == 'development']
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {str(row['animal_id']) for row in coronal_records} & set(by_id)
sagittal_done = json.loads((sagittal / 'summary.json').read_text())
for file, digest in sagittal_done['output_sha256'].items():
    assert sha(sagittal / file) == digest
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert not {str(row['donor_id']) for row in sagittal_records} & set(by_id)
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert sha(sagittal_train / 'summary.json') == config['113']['sagittal_summary_sha256']
assert all(sha(sagittal_train / file) == digest
    for file, digest in config['113']['sagittal_output_sha256'].items())
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
assert not {row['section_id'] for row in sagittal_records} & {
    row['section_id'] for row in draws if row['kind'] == 'sagittal_weak_train'}
panel_identities = ([(row['kind'], row['animal_id'], row['specimen_id'],
    row['experiment_id'], row['section_id']) for row in synthetic_panel + virtual_panel]
    + [('coronal_weak_dev', row['animal_id'], row['specimen_id'],
        row['experiment_id'], row['section_id']) for row in coronal_records]
    + [('sagittal_weak_dev', row['donor_id'], row['specimen_id'],
        row['experiment_id'], row['section_id']) for row in sagittal_records])
assert panel_identities == [(row['kind'], row['animal_id'], row['specimen_id'],
    row['experiment_id'], row['section_id']) for row in expected]

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
pixels = five * side
flags = torch.tensor([0, 1], device='cuda')[None, None]
rows = []
with torch.inference_mode():
    for name, checkpoints in steps.items():
        for step in checkpoints:
            checkpoint = torch.load(runs[name] / f'joint_step_{step:05d}.pt',
                map_location='cpu', weights_only=True)
            assert checkpoint['step'] == step and json.loads(json.dumps(checkpoint['config'])) == config[name]
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            for item in synthetic_panel + virtual_panel:
                image = torch.from_numpy(item['image'][None].copy()).cuda()
                truth_state = torch.from_numpy(item['state'][None].copy()).cuda()
                truth_reflection = torch.tensor([item['reflection']], device='cuda')
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
                state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflection = flags.expand(1, prediction['state'].shape[1], 2)
                chart = chart16 if item['kind'] == 'synthetic_dev' else five
                discrepancy = (rigid_points_090(state, reflection, chart)
                    - rigid_points_090(truth_state, truth_reflection, chart)[:, None, None]
                    ).norm(dim=-1)
                if item['kind'] == 'synthetic_dev':
                    valid16 = (F.interpolate(torch.from_numpy(item['valid'][None, None]).cuda().float(),
                        (16, 16), mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
                    discrepancy = discrepancy[..., valid16]
                error = discrepancy.mean(-1).flatten() / 1000
                selected = int(prior[0].argmax())
                best14 = int(beam[error[beam].argmin()])
                best160 = int(error.argmin())
                assert error.numel() == 160 and beam.numel() == 14 and selected in beam.tolist()
                rows.append({'model': name, 'step': step, 'kind': item['kind'],
                    'animal_id': item['animal_id'], 'section_id': item['section_id'],
                    'specimen_id': item['specimen_id'], 'experiment_id': item['experiment_id'],
                    'synthetic_subject_plan_id': item.get('synthetic_subject_plan_id'),
                    'panel_file_sha256': item.get('panel_file_sha256'),
                    'source_images_sha256': item.get('source_images_sha256'),
                    'source_records_sha256': item.get('source_records_sha256'),
                    'selected_branch': selected, 'best14_branch': best14,
                    'best160_branch': best160, 'selected_error_mm': float(error[selected]),
                    'best14_error_mm': float(error[best14]),
                    'best160_oracle_error_mm': float(error[best160])})
            for domain, source_records, images in (
                ('coronal_weak_dev', coronal_records, coronal_images),
                ('sagittal_weak_dev', sagittal_records, sagittal_images)):
                for record in source_records:
                    i = record['array_row_index']
                    native = np.concatenate((np.asarray(images[i], dtype=np.float32),
                        np.zeros((4, *images.shape[-2:]), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    if domain == 'coronal_weak_dev':
                        image = F.interpolate(image, (side, side),
                            mode='bilinear', align_corners=False)
                        affine = torch.as_tensor(coronal_affines[i], device='cuda', dtype=torch.float32)
                        reference = (affine[:, 2] + (192 / side) * pixels[:, :1] * affine[:, 0]
                            + (192 / side) * pixels[:, 1:] * affine[:, 1])
                        animal_id = record['animal_id']
                        image_sha256 = record['source_image_sha256']
                    else:
                        affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                            device='cuda', dtype=torch.float32)
                        reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
                        animal_id = record['donor_id']
                        image_sha256 = record['image_sha256']
                    prediction = model.predict(image)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    branch = int(prior[0].argmax())
                    proposed = rigid_points_090(prediction['state'][:, branch // 2],
                        torch.tensor([branch % 2], device='cuda'), five)[0]
                    rows.append({'model': name, 'step': step, 'kind': domain,
                        'animal_id': animal_id, 'section_id': record['section_id'],
                        'specimen_id': record['specimen_id'],
                        'experiment_id': record['experiment_id'],
                        'source_image_sha256': image_sha256,
                        'selected_branch': branch,
                        'selected_error_mm': float((proposed - reference).norm(dim=-1).mean() / 1000)})
            print(json.dumps({'model': name, 'step': step, 'evaluated_sections': 318}), flush=True)

assert len(rows) == 318 * sum(len(checkpoints) for checkpoints in steps.values())
frozen = {(row['step'], row['kind'], row['section_id']): row for row in old_rows
    if row['step'] in steps['113']}
for row in rows:
    if row['model'] == '113':
        prior_row = frozen[row['step'], row['kind'], row['section_id']]
        assert row['selected_branch'] == prior_row['selected_branch']
        assert abs(row['selected_error_mm'] - prior_row['selected_error_mm']) < 1e-4
        if row['kind'] in ('synthetic_dev', 'virtual_oblique_train_holdout'):
            assert abs(row['best14_error_mm'] - prior_row['best14_error_mm']) < 1e-4
zero = {(row['kind'], row['section_id']): row for row in rows
    if row['model'] == '113' and row['step'] == 0}
for row in rows:
    if row['model'] == '115' and row['step'] == 0:
        parent = zero[row['kind'], row['section_id']]
        assert row['selected_branch'] == parent['selected_branch']
        assert abs(row['selected_error_mm'] - parent['selected_error_mm']) < 1e-4
        if row['kind'] in ('synthetic_dev', 'virtual_oblique_train_holdout'):
            assert row['best14_branch'] == parent['best14_branch']
            assert row['best160_branch'] == parent['best160_branch']
            assert abs(row['best14_error_mm'] - parent['best14_error_mm']) < 1e-4
            assert abs(row['best160_oracle_error_mm'] - parent['best160_oracle_error_mm']) < 1e-4

summary = {'role': 'paired loss-weight diagnostic on reused 113 TRAIN-subset virtual cuts and DEV guardrails',
    'caveat': 'Non-final: virtual cuts are weak-Allen-affine serial TRAIN reslices, not acquired oblique; parent saw cardinal donor images. Synthetic/weak DEV is reused, not untouched animal-independent validation, calibration, public benchmarking, or GUI selection.',
    'metrics': 'Synthetic and virtual: prior-selected, prior top8 base + top6 anchor best-of-14, full160 both-reflection oracle. Cardinal guardrails: prior-selected five-point weak-affine error.',
    'models': {}}
for name, checkpoints in steps.items():
    summary['models'][name] = {}
    for step in checkpoints:
        groups = {}
        for kind in ('synthetic_dev', 'virtual_oblique_train_holdout',
            'coronal_weak_dev', 'sagittal_weak_dev'):
            group = [row for row in rows if row['model'] == name and row['step'] == step
                and row['kind'] == kind]
            identity_key = ('synthetic_subject_plan_id' if kind == 'synthetic_dev'
                else 'animal_id')
            identities = sorted({row[identity_key] for row in group})
            metrics = (('selected_error_mm', 'best14_error_mm', 'best160_oracle_error_mm')
                if kind in ('synthetic_dev', 'virtual_oblique_train_holdout')
                else ('selected_error_mm',))
            groups[kind] = {'sections': len(group), 'equal_unit': identity_key,
                'units': len(identities),
                ('plan_equal' if kind == 'synthetic_dev' else 'donor_equal'): {metric: float(np.mean([
                    np.mean([row[metric] for row in group if row[identity_key] == identity])
                    for identity in identities])) for metric in metrics},
                'selected_over_3mm': sum(row['selected_error_mm'] > 3 for row in group)}
        summary['models'][name][str(step)] = groups

summary['paired_deltas_115_minus_113'] = {}
for label, step115, step113 in (('step0_parity', 0, 0),
    ('step1000_matched_prefix', 1000, 1000),
    ('step1306_vs_113_step1000_unequal_updates', 1306, 1000)):
    summary['paired_deltas_115_minus_113'][label] = {}
    for kind in ('synthetic_dev', 'virtual_oblique_train_holdout',
        'coronal_weak_dev', 'sagittal_weak_dev'):
        reference = {row['section_id']: row for row in rows if row['model'] == '113'
            and row['step'] == step113 and row['kind'] == kind}
        group = [row for row in rows if row['model'] == '115'
            and row['step'] == step115 and row['kind'] == kind]
        identity_key = ('synthetic_subject_plan_id' if kind == 'synthetic_dev'
            else 'animal_id')
        units = sorted({row[identity_key] for row in group})
        metrics = (('selected_error_mm', 'best14_error_mm', 'best160_oracle_error_mm')
            if kind in ('synthetic_dev', 'virtual_oblique_train_holdout')
            else ('selected_error_mm',))
        summary['paired_deltas_115_minus_113'][label][kind] = {
            metric: float(np.mean([np.mean([row[metric] - reference[row['section_id']][metric]
                for row in group if row[identity_key] == unit]) for unit in units]))
            for metric in metrics}

output_config = {'steps': steps,
    'training_config_sha256': {name: sha(run / 'config.json') for name, run in runs.items()},
    'training_completion_sha256': {name: sha(run / 'completed.json') for name, run in runs.items()},
    'training_draws_sha256': {name: done[name]['draws_sha256'] for name in runs},
    'checkpoint_sha256': {name: {str(step): done[name]['checkpoint_sha256'][str(step)]
        for step in steps[name]} for name in runs},
    'training_source_sha256': {name: config[name]['source_sha256'] for name in runs},
    'frozen_113_eval_completion_sha256': sha(old_eval / 'completed.json'),
    'frozen_113_eval_rows_sha256': old_done['rows_sha256'],
    'frozen_113_eval_summary_sha256': old_done['summary_sha256'],
    'frozen_panel_identity_sha256': hashlib.sha256(json.dumps(panel_identities).encode()).hexdigest(),
    'virtual_source_manifest_sha256': train['manifest_sha256'],
    'virtual_attempts': virtual_attempts,
    'synthetic_panel_completed_sha256': sha(panel / 'completed.json'),
    'synthetic_panel_records_sha256': panel_done['records_sha256'],
    'coronal_completed_sha256': sha(coronal / 'completed.json'),
    'coronal_output_sha256': {file: coronal_done['output_sha256'][file]
        for file in ('images.npy', 'geometry.npz', 'records.jsonl')},
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_output_sha256': sagittal_done['output_sha256'],
    'sagittal_train_summary_sha256': config['113']['sagittal_summary_sha256'],
    'sagittal_train_output_sha256': config['113']['sagittal_output_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'sections_per_checkpoint': 318,
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
