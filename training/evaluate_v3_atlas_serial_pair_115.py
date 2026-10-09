"""Evaluation-only paired donor/atlas-serial source readout on frozen 113 virtual cuts."""
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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import VERSION
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.diagnose_atlas_serial_control_115 import sample_atlas_serial_control_115
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
runs = {'113': root / 'runs/v3_mixed_virtual_oblique_113',
    '115': root / 'runs/v3_high_virtual_oblique_115'}
old_eval = root / 'runs/v3_mixed_virtual_oblique_113_eval'
out = root / 'runs/v3_high_virtual_oblique_115_atlas_serial_pair_eval'
steps = {'113': (1000,), '115': (1000, 1306)}
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
    assert all(sha(source / file) == digest
        for file, digest in config[name]['source_sha256'].items())
    assert all(sha(run / f'joint_step_{step:05d}.pt') ==
        done[name]['checkpoint_sha256'][str(step)] for step in steps[name])
    assert not any(done[name][key] for key in ('calibrated', 'public_benchmark_used',
        'expert_real_truth_used', 'external_pretrained_weights_used'))
assert config['115']['comparison_113_completed_sha256'] == sha(runs['113'] / 'completed.json')
assert done['115']['comparison_113_completed_sha256'] == config['115']['comparison_113_completed_sha256']
assert config['115']['expected_113_draw_prefix_sha256'] == done['115']['matched_113_draw_prefix_sha256'] == done['115']['draws_sha256']
draws_115 = (runs['115'] / 'draws.jsonl').read_bytes()
assert draws_115 and (runs['113'] / 'draws.jsonl').read_bytes().startswith(draws_115)
for key in ('seed', 'side', 'virtual_train_animal_ids', 'virtual_holdout_animal_ids',
    'virtual_holdout_seeds', 'virtual_source_audit_sha256', 'synthetic_provenance',
    'coronal_bindings', 'parent_checkpoint_sha256', 'teacher_checkpoint_sha256'):
    assert config['115'][key] == config['113'][key], key
assert sha(source.parent / 'docs/publication/V3_MIXED_VIRTUAL_OBLIQUE_113_PROTOCOL_20261009.md') == config['113']['protocol_sha256']
assert sha(source.parent / 'docs/publication/V3_HIGH_VIRTUAL_OBLIQUE_115_PROTOCOL_20261009.md') == config['115']['protocol_sha256']
assert sha(source / 'arbitrary_plane_allen_atlas_binding_v6.py') == config['113'][
    'synthetic_provenance']['source_sha256']['arbitrary_plane_allen_atlas_binding_v6.py']

old_done = json.loads((old_eval / 'completed.json').read_text())
assert not any(old_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used'))
assert sha(old_eval / 'rows.jsonl') == old_done['rows_sha256']
assert sha(old_eval / 'summary.json') == old_done['summary_sha256']
assert sha(source / 'evaluate_v3_mixed_virtual_oblique_113.py') == old_done['evaluator_sha256']
assert sha(runs['113'] / 'completed.json') == old_done['run_completion_sha256']
old_rows = [json.loads(line) for line in (old_eval / 'rows.jsonl').open()]
frozen = [row for row in old_rows if row['step'] == 1000
    and row['kind'] == 'virtual_oblique_train_holdout']
assert len(frozen) == 32
assert [(row['animal_id'], row['section_id']) for row in frozen] == [
    (row['animal_id'], row['section_id']) for row in old_rows if row['step'] == 0
    and row['kind'] == 'virtual_oblique_train_holdout']

binding = config['113']['synthetic_provenance']['atlas_binding']
assert binding == {'template_path': str(allen.TEMPLATE_PATH_V6),
    'template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'annotation_path': str(allen.ANNOTATION_PATH_V6),
    'annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'normalized_array_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'axes': ['AP', 'DV', 'ML'], 'physical_units': 'um'}
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
del annotation
atlas_context = {'atlas': torch.from_numpy(atlas_array).cuda(),
    'provenance': {'atlas_binding': binding}}
del atlas_array
helper_sha256 = sha(source / 'diagnose_atlas_serial_control_115.py')
train = load_reserved_real_train()
assert train['manifest_sha256'] == old_done['source_manifest_sha256']
assert train['bindings'] == config['113']['coronal_bindings']
by_id = {str(donor['animal_id']): i for i, donor in enumerate(train['donors'])}
items, virtual_attempts = [], []
with torch.inference_mode():
    for animal_id in config['113']['virtual_holdout_animal_ids']:
        index = by_id[str(animal_id)]
        donor = train['donors'][index]
        for base_seed in config['113']['virtual_holdout_seeds']:
            seed = int(base_seed)
            while True:
                real = sample_reserved_real_oblique_v3(train, index, seed,
                    side=side, device='cuda', return_source=True)
                used = bool(real['eligible'][0])
                record = real['provenance'][0]
                virtual_attempts.append({'animal_id': animal_id, 'seed': seed,
                    'used': used, 'section_id': record['section_id']})
                if used:
                    break
                seed += 10000
            geometry = record['real_oblique_v3']
            assert geometry['source_images_sha256'] == train['bindings'][str(donor['images'])]
            assert geometry['source_records_sha256'] == train['bindings'][str(donor['images'].parent / 'records.jsonl')]
            atlas = sample_atlas_serial_control_115(train, atlas_context, real,
                index, seed, device='cuda')
            paired_record = atlas['provenance'][0]
            control = paired_record['atlas_serial_control_115']
            assert {key: value for key, value in paired_record.items()
                if key not in (VERSION, 'atlas_serial_control_115')} == {
                key: value for key, value in record.items() if key != VERSION}
            assert paired_record[VERSION]['events'] == record[VERSION]['events']
            assert paired_record[VERSION]['parameters'] == record[VERSION]['parameters']
            assert torch.equal(atlas['state'], real['state'])
            assert torch.equal(atlas['reflection'], real['reflection'])
            assert torch.equal(atlas['brush_mask'], real['brush_mask'])
            assert torch.equal(atlas['inputs'][:, 1:3], real['inputs'][:, 1:3])
            assert control['paired_real_section_id'] == record['section_id']
            assert control['seed'] == seed and control['donor_index'] == index
            assert control['source_images_sha256'] == geometry['source_images_sha256']
            assert control['source_records_sha256'] == geometry['source_records_sha256']
            assert control['source_section_ids_sha256'] == geometry['source_section_ids_sha256']
            assert control['helper_sha256'] == helper_sha256
            assert control['atlas_binding'] == binding
            provenance_sha256 = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            control_sha256 = hashlib.sha256(json.dumps(control, sort_keys=True).encode()).hexdigest()
            for source_name, sample in (('donor_virtual', real), ('atlas_serial', atlas)):
                image = sample['inputs'][0].cpu().numpy().copy()
                items.append({'source': source_name, 'image': image,
                    'input_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'state': sample['state'][0].cpu().numpy().copy(),
                    'reflection': int(sample['reflection'][0]),
                    'valid_pixels': int(sample['valid_mask'][0].sum()),
                    'eligible': bool(sample['eligible'][0]),
                    'animal_id': animal_id, 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'section_id': record['section_id'], 'seed': seed,
                    'virtual_provenance_sha256': provenance_sha256,
                    'atlas_control_receipt_sha256': control_sha256,
                    'source_images_sha256': geometry['source_images_sha256'],
                    'source_records_sha256': geometry['source_records_sha256'],
                    'source_section_ids_sha256': geometry['source_section_ids_sha256'],
                    'source_state_sha256': control['source_state_sha256'],
                    'source_thickness_sha256': control['source_thickness_sha256'],
                    'atlas_grid_sha256': control['grid_sha256']})
assert virtual_attempts == old_done['virtual_attempts'] and len(items) == 64
assert [(row['animal_id'], row['section_id']) for row in items
    if row['source'] == 'donor_virtual'] == [
    (row['animal_id'], row['section_id']) for row in frozen]
del atlas_context
torch.cuda.empty_cache()

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
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
            for item in items:
                image = torch.from_numpy(item['image'][None]).cuda()
                state = torch.from_numpy(item['state'][None]).cuda()
                reflection = torch.tensor([item['reflection']], device='cuda')
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(1, prediction['state'].shape[1], 2)
                error = ((rigid_points_090(states, reflections, five)
                    - rigid_points_090(state, reflection, five)[:, None, None])
                    .norm(dim=-1).mean(-1).flatten() / 1000)
                selected = int(prior[0].argmax())
                best14 = int(beam[error[beam].argmin()])
                best160 = int(error.argmin())
                assert error.numel() == 160 and beam.numel() == 14 and selected in beam.tolist()
                rows.append({'model': name, 'step': step, 'source': item['source'],
                    'animal_id': item['animal_id'], 'specimen_id': item['specimen_id'],
                    'experiment_id': item['experiment_id'], 'section_id': item['section_id'],
                    'seed': item['seed'], 'input_sha256': item['input_sha256'],
                    'virtual_provenance_sha256': item['virtual_provenance_sha256'],
                    'atlas_control_receipt_sha256': item['atlas_control_receipt_sha256'],
                    'valid_pixels': item['valid_pixels'], 'eligible': item['eligible'],
                    'source_images_sha256': item['source_images_sha256'],
                    'source_records_sha256': item['source_records_sha256'],
                    'source_section_ids_sha256': item['source_section_ids_sha256'],
                    'source_state_sha256': item['source_state_sha256'],
                    'source_thickness_sha256': item['source_thickness_sha256'],
                    'atlas_grid_sha256': item['atlas_grid_sha256'],
                    'selected_branch': selected, 'best14_branch': best14,
                    'best160_branch': best160, 'selected_error_mm': float(error[selected]),
                    'best14_error_mm': float(error[best14]),
                    'best160_oracle_error_mm': float(error[best160])})
            print(json.dumps({'model': name, 'step': step, 'paired_sections': 32}), flush=True)

assert len(rows) == 192
old_1000 = {row['section_id']: row for row in frozen}
for row in rows:
    if row['model'] == '113' and row['source'] == 'donor_virtual':
        reference = old_1000[row['section_id']]
        assert row['selected_branch'] == reference['selected_branch']
        assert abs(row['selected_error_mm'] - reference['selected_error_mm']) < 1e-4
        assert abs(row['best14_error_mm'] - reference['best14_error_mm']) < 1e-4

metrics = ('selected_error_mm', 'best14_error_mm', 'best160_oracle_error_mm')
donors = sorted({row['animal_id'] for row in rows})
assert len(donors) == 8
summary = {'role': 'paired source-domain diagnostic on exactly the 113 32 virtual TRAIN-subset cuts',
    'caveat': 'Non-final: both sources use the same weak Allen affine pose, not acquired oblique truth. Atlas-derived serial sections isolate source appearance but are not an independent animal validation, calibration, public benchmark, or GUI selection set.',
    'source_gap_sign': 'atlas_serial minus donor_virtual, same donor and physical cut',
    'model_delta_note': 'Step 1000 is the matched-prefix checkpoint; 115 step 1306 versus 113 step 1000 has unequal update counts.',
    'models': {}}
for name, checkpoints in steps.items():
    summary['models'][name] = {}
    for step in checkpoints:
        group = [row for row in rows if row['model'] == name and row['step'] == step]
        paired = {(row['section_id'], row['source']): row for row in group}
        assert len(paired) == 64
        by_source = {}
        for source_name in ('donor_virtual', 'atlas_serial'):
            subset = [row for row in group if row['source'] == source_name]
            by_source[source_name] = {'sections': len(subset), 'donors': len(donors),
                'eligible_sections': sum(row['eligible'] for row in subset),
                'donor_equal': {metric: float(np.mean([np.mean([
                    row[metric] for row in subset if row['animal_id'] == donor])
                    for donor in donors])) for metric in metrics}}
        per_donor = {donor: {metric: float(np.mean([
            paired[(row['section_id'], 'atlas_serial')][metric] - row[metric]
            for row in group if row['source'] == 'donor_virtual'
            and row['animal_id'] == donor])) for metric in metrics} for donor in donors}
        summary['models'][name][str(step)] = {'sources': by_source,
            'paired_source_gap': {'per_donor': per_donor,
                'donor_equal': {metric: float(np.mean([
                    per_donor[donor][metric] for donor in donors])) for metric in metrics}}}

reference = {(row['section_id'], row['source']): row for row in rows
    if row['model'] == '113' and row['step'] == 1000}
summary['model_deltas_115_minus_113_step1000'] = {}
for step in steps['115']:
    summary['model_deltas_115_minus_113_step1000'][str(step)] = {}
    for source_name in ('donor_virtual', 'atlas_serial'):
        group = [row for row in rows if row['model'] == '115' and row['step'] == step
            and row['source'] == source_name]
        summary['model_deltas_115_minus_113_step1000'][str(step)][source_name] = {
            metric: float(np.mean([np.mean([row[metric] - reference[
                (row['section_id'], source_name)][metric] for row in group
                if row['animal_id'] == donor]) for donor in donors])) for metric in metrics}

output_config = {'steps': steps,
    'training_config_sha256': {name: sha(run / 'config.json') for name, run in runs.items()},
    'training_completion_sha256': {name: sha(run / 'completed.json') for name, run in runs.items()},
    'training_draws_sha256': {name: done[name]['draws_sha256'] for name in runs},
    'checkpoint_sha256': {name: {str(step): done[name]['checkpoint_sha256'][str(step)]
        for step in steps[name]} for name in runs},
    'frozen_113_eval_completion_sha256': sha(old_eval / 'completed.json'),
    'frozen_113_eval_rows_sha256': old_done['rows_sha256'],
    'virtual_source_manifest_sha256': train['manifest_sha256'],
    'frozen_virtual_identity_sha256': hashlib.sha256(json.dumps([
        (row['animal_id'], row['section_id']) for row in frozen]).encode()).hexdigest(),
    'virtual_attempts': virtual_attempts,
    'atlas_binding': binding, 'helper_sha256': helper_sha256,
    'paired_input_sha256': hashlib.sha256(json.dumps([
        (item['section_id'], item['source'], item['input_sha256'])
        for item in items]).encode()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'paired_sections_per_checkpoint': 32,
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
