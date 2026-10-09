"""Non-final direct-atlas versus frozen donor/atlas-serial virtual-cut diagnostic."""
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
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import VERSION, sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.diagnose_atlas_serial_control_115 import sample_atlas_serial_control_115
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
run = root / 'runs/v3_high_virtual_oblique_115'
paired = root / 'runs/v3_high_virtual_oblique_115_atlas_serial_pair_eval'
out = root / 'runs/v3_high_virtual_oblique_116_direct_atlas_eval'
step, side = 1306, 256
torch.set_num_threads(4)
assert not out.exists()

run_done = json.loads((run / 'completed.json').read_text())
run_config = json.loads((run / 'config.json').read_text())
pair_done = json.loads((paired / 'completed.json').read_text())
pair_config = json.loads((paired / 'config.json').read_text())
assert run_done['batches'] == run_config['batches'] == step
assert sha(run / 'completed.json') == pair_config['training_completion_sha256']['115']
assert sha(run / 'config.json') == run_done['config_sha256'] == pair_config['training_config_sha256']['115']
assert sha(run / 'draws.jsonl') == run_done['draws_sha256'] == pair_config['training_draws_sha256']['115']
assert sha(run / 'training.jsonl') == run_done['training_sha256']
assert sha(run / f'joint_step_{step:05d}.pt') == run_done['checkpoint_sha256'][str(step)] == pair_config['checkpoint_sha256']['115'][str(step)]
assert run_done['source_sha256'] == run_config['source_sha256']
assert all(sha(source / file) == digest for file, digest in run_config['source_sha256'].items())
assert not any(run_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert sha(paired / 'config.json') == pair_done['config_sha256']
assert sha(paired / 'rows.jsonl') == pair_done['rows_sha256']
assert sha(paired / 'summary.json') == pair_done['summary_sha256']
assert sha(source / 'evaluate_v3_atlas_serial_pair_115.py') == pair_done['evaluator_sha256']
assert pair_done['rows'] == 192 and pair_done['paired_sections_per_checkpoint'] == 32
assert not any(pair_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used'))
helper_sha256 = sha(source / 'diagnose_atlas_serial_control_115.py')
assert helper_sha256 == pair_config['helper_sha256']
pair_rows = [json.loads(line) for line in (paired / 'rows.jsonl').open()]
reference = {(row['section_id'], row['source']): row for row in pair_rows
    if row['model'] == '115' and row['step'] == step}
frozen = [row for row in pair_rows if row['model'] == '115' and row['step'] == step
    and row['source'] == 'donor_virtual']
assert len(reference) == 64 and len(frozen) == 32
assert len({row['section_id'] for row in frozen}) == 32
assert pair_config['frozen_virtual_identity_sha256'] == hashlib.sha256(json.dumps([
    (row['animal_id'], row['section_id']) for row in frozen]).encode()).hexdigest()

binding = pair_config['atlas_binding']
assert binding == run_config['synthetic_provenance']['atlas_binding']
assert binding == {'template_path': str(allen.TEMPLATE_PATH_V6),
    'template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'annotation_path': str(allen.ANNOTATION_PATH_V6),
    'annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'normalized_array_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'axes': ['AP', 'DV', 'ML'], 'physical_units': 'um'}
assert sha(source / 'arbitrary_plane_allen_atlas_binding_v6.py') == run_config[
    'synthetic_provenance']['source_sha256']['arbitrary_plane_allen_atlas_binding_v6.py']
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
del annotation
atlas_context = {'atlas': torch.from_numpy(atlas_array).cuda(),
    'provenance': {'atlas_binding': binding}}
del atlas_array
train = load_reserved_real_train()
assert train['manifest_sha256'] == pair_config['virtual_source_manifest_sha256']
assert train['bindings'] == run_config['coronal_bindings']
by_id = {str(donor['animal_id']): i for i, donor in enumerate(train['donors'])}
items, attempts = [], []
with torch.inference_mode():
    for animal_id in run_config['virtual_holdout_animal_ids']:
        index = by_id[str(animal_id)]
        for base_seed in run_config['virtual_holdout_seeds']:
            seed = int(base_seed)
            while True:
                real = sample_reserved_real_oblique_v3(train, index, seed,
                    side=side, device='cuda', return_source=True)
                record = real['provenance'][0]
                used = bool(real['eligible'][0])
                attempts.append({'animal_id': animal_id, 'seed': seed,
                    'used': used, 'section_id': record['section_id']})
                if used:
                    break
                seed += 10000
            assert real['offsets'].shape[1] == real['weights'].shape[1] == 9
            geometry = record['real_oblique_v3']
            serial = sample_atlas_serial_control_115(train, atlas_context, real,
                index, seed, device='cuda')
            serial_control = serial['provenance'][0]['atlas_serial_control_115']
            normal = torch.as_tensor(geometry['plane_rotation'], device='cuda',
                dtype=torch.float32)[:, 2]
            coordinates = (real['source_centre'][:, None]
                + real['offsets'][:, :, None, None, None] * normal[None, None, None, None])
            rendered = render_finite_thickness_coordinate_grid(atlas_context['atlas'],
                coordinates, (0., 0., 0.), (25., 25., 25.), real['weights'])
            support = rendered[:, 1]
            source_image = (rendered[:, 0] / support.clamp_min(1e-4)).clamp(0, 1)
            appearance = record['appearance']
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, side, device='cuda'),
                torch.linspace(-1, 1, side, device='cuda'), indexing='ij')
            background = (appearance['background_mean']
                + appearance['background_slope_yx'][0] * yy
                + appearance['background_slope_yx'][1] * xx).clamp(0, 1)
            image = (source_image.pow(appearance['tissue_gamma']) * appearance['tissue_gain']
                + (1 - support).clamp(0, 1) * background).clamp(0, 1)
            inputs = torch.zeros_like(real['inputs'])
            inputs[:, 0] = image
            original = {key: value for key, value in record.items() if key != VERSION}
            direct_receipt = {'role': 'evaluation-only direct nine-offset oblique atlas render; no donor serial-volume intermediate',
                'paired_real_section_id': record['section_id'], 'seed': seed,
                'coordinates_sha256': hashlib.sha256(coordinates.cpu().numpy().tobytes()).hexdigest(),
                'atlas_binding': binding, 'source_images_sha256': geometry['source_images_sha256'],
                'source_records_sha256': geometry['source_records_sha256'],
                'source_section_ids_sha256': geometry['source_section_ids_sha256']}
            source_sample = {'inputs': inputs, 'state': real['source_state'],
                'reflection': real['reflection'], 'offsets': real['offsets'],
                'weights': real['weights'], 'centre': real['source_centre'],
                'visible': support, 'support': support,
                'masks': torch.ones_like(support, dtype=torch.bool),
                'eligible': torch.tensor([int((support > .2).sum()) >= side * side * .05],
                    device='cuda'),
                'provenance': [{**original, 'direct_atlas_control_116': direct_receipt}]}
            direct = sample_one_shot_slide_artifacts_v3(train, [index], seed,
                side, source_sample=source_sample)
            direct['inputs'][:, 0] *= real['brush_mask']
            direct['inputs'][:, 1:3] = real['inputs'][:, 1:3]
            direct['brush_mask'] = real['brush_mask']
            assert torch.equal(direct['state'], real['state'])
            assert torch.equal(direct['reflection'], real['reflection'])
            assert torch.equal(direct['offsets'], real['offsets'])
            assert torch.equal(direct['weights'], real['weights'])
            assert torch.equal(direct['brush_mask'], real['brush_mask'])
            assert torch.equal(direct['inputs'][:, 1:3], real['inputs'][:, 1:3])
            assert direct['provenance'][0][VERSION]['events'] == record[VERSION]['events']
            assert direct['provenance'][0][VERSION]['parameters'] == record[VERSION]['parameters']
            assert {key: value for key, value in direct['provenance'][0].items()
                if key not in (VERSION, 'direct_atlas_control_116')} == original
            valid16 = (F.interpolate(real['valid_mask'][:, None].float(),
                (16, 16), mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
            assert bool(valid16.any())
            provenance_sha256 = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            serial_receipt_sha256 = hashlib.sha256(json.dumps(
                serial_control, sort_keys=True).encode()).hexdigest()
            for source_name, sample in (('donor_virtual', real),
                ('atlas_serial', serial), ('direct_atlas', direct)):
                assert torch.equal(sample['state'], real['state'])
                assert torch.equal(sample['reflection'], real['reflection'])
                item_image = sample['inputs'][0].cpu().numpy().copy()
                digest = hashlib.sha256(item_image.tobytes()).hexdigest()
                if source_name != 'direct_atlas':
                    frozen_row = reference[(record['section_id'], source_name)]
                    assert digest == frozen_row['input_sha256']
                    assert frozen_row['animal_id'] == animal_id
                    assert frozen_row['specimen_id'] == record['specimen_id']
                    assert frozen_row['experiment_id'] == record['experiment_id']
                    assert frozen_row['seed'] == seed
                    assert frozen_row['virtual_provenance_sha256'] == provenance_sha256
                    assert frozen_row['atlas_control_receipt_sha256'] == serial_receipt_sha256
                    assert frozen_row['source_images_sha256'] == geometry['source_images_sha256']
                    assert frozen_row['source_records_sha256'] == geometry['source_records_sha256']
                items.append({'source': source_name, 'image': item_image,
                    'input_sha256': digest, 'state': sample['state'][0].cpu().numpy().copy(),
                    'reflection': int(sample['reflection'][0]),
                    'valid16': valid16.cpu().numpy().copy(),
                    'valid_tissue_fraction16': float(valid16.float().mean()),
                    'source_valid_pixels': int(sample['valid_mask'][0].sum()),
                    'eligible': bool(sample['eligible'][0]),
                    'animal_id': animal_id, 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                    'seed': seed, 'virtual_provenance_sha256': provenance_sha256,
                    'atlas_serial_receipt_sha256': serial_receipt_sha256,
                    'direct_coordinates_sha256': direct_receipt['coordinates_sha256'],
                    'source_images_sha256': geometry['source_images_sha256'],
                    'source_records_sha256': geometry['source_records_sha256'],
                    'source_section_ids_sha256': geometry['source_section_ids_sha256']})
assert attempts == pair_config['virtual_attempts'] and len(items) == 96
assert [item['section_id'] for item in items if item['source'] == 'donor_virtual'] == [
    row['section_id'] for row in frozen]
del atlas_context
torch.cuda.empty_cache()

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
    map_location='cpu', weights_only=True)
assert checkpoint['step'] == step and json.loads(json.dumps(checkpoint['config'])) == run_config
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(-1, 2)
flags = torch.tensor([0, 1], device='cuda')[None, None]
rows = []
with torch.inference_mode():
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
        row = {'model': '115', 'step': step, 'source': item['source'],
            'animal_id': item['animal_id'], 'specimen_id': item['specimen_id'],
            'experiment_id': item['experiment_id'], 'section_id': item['section_id'],
            'seed': item['seed'], 'input_sha256': item['input_sha256'],
            'virtual_provenance_sha256': item['virtual_provenance_sha256'],
            'atlas_serial_receipt_sha256': item['atlas_serial_receipt_sha256'],
            'direct_coordinates_sha256': item['direct_coordinates_sha256'],
            'source_images_sha256': item['source_images_sha256'],
            'source_records_sha256': item['source_records_sha256'],
            'source_section_ids_sha256': item['source_section_ids_sha256'],
            'source_valid_pixels': item['source_valid_pixels'],
            'eligible': item['eligible'],
            'donor_valid_tissue_fraction16': item['valid_tissue_fraction16'],
            'selected_branch': selected, 'best14_branch': best14,
            'best160_branch': best160, 'tissue_best14_branch': tissue_best14,
            'tissue_best160_branch': tissue_best160,
            'selected_error_mm': float(five_error[selected]),
            'best14_error_mm': float(five_error[best14]),
            'best160_oracle_error_mm': float(five_error[best160]),
            'tissue_selected_error_mm': float(tissue_error[selected]),
            'tissue_best14_error_mm': float(tissue_error[tissue_best14]),
            'tissue_best160_oracle_error_mm': float(tissue_error[tissue_best160])}
        if item['source'] != 'direct_atlas':
            frozen_row = reference[(item['section_id'], item['source'])]
            assert row['selected_branch'] == frozen_row['selected_branch']
            assert row['best14_branch'] == frozen_row['best14_branch']
            assert row['best160_branch'] == frozen_row['best160_branch']
            for metric in ('selected_error_mm', 'best14_error_mm',
                'best160_oracle_error_mm'):
                assert abs(row[metric] - frozen_row[metric]) < 1e-4
        rows.append(row)

assert len(rows) == 96
metrics = ('selected_error_mm', 'best14_error_mm', 'best160_oracle_error_mm',
    'tissue_selected_error_mm', 'tissue_best14_error_mm',
    'tissue_best160_oracle_error_mm')
donors = sorted({row['animal_id'] for row in rows})
assert len(donors) == 8
by_section = {(row['section_id'], row['source']): row for row in rows}
assert len(by_section) == 96
summary = {'role': 'direct atlas source ablation on the frozen 113/115 32 virtual TRAIN-subset cuts',
    'caveat': 'Non-final: direct atlas and atlas-serial are rendered from the same Allen atlas that defines the weak pose label. No acquired-oblique truth, untouched animal validation, calibration, public benchmark, or GUI selection claim.',
    'five_point': 'same fixed five canvas points as frozen 115 paired evaluator',
    'tissue': 'same 16x16 donor-virtual tissue chart for all three sources; keep points where bilinear-downsampled donor valid mask equals one',
    'donor_valid_tissue_fraction16': float(np.mean([row['donor_valid_tissue_fraction16']
        for row in rows if row['source'] == 'donor_virtual'])),
    'sources': {}, 'paired_gaps': {}}
for source_name in ('donor_virtual', 'atlas_serial', 'direct_atlas'):
    group = [row for row in rows if row['source'] == source_name]
    summary['sources'][source_name] = {'sections': len(group),
        'eligible_sections': sum(row['eligible'] for row in group),
        'donor_equal': {metric: float(np.mean([np.mean([
            row[metric] for row in group if row['animal_id'] == donor])
            for donor in donors])) for metric in metrics},
        'tissue_minus_five_selected_mm': float(np.mean([np.mean([
            row['tissue_selected_error_mm'] - row['selected_error_mm']
            for row in group if row['animal_id'] == donor]) for donor in donors]))}
for baseline in ('donor_virtual', 'atlas_serial'):
    per_donor = {donor: {metric: float(np.mean([
        row[metric] - by_section[(row['section_id'], baseline)][metric]
        for row in rows if row['source'] == 'direct_atlas' and row['animal_id'] == donor]))
        for metric in metrics} for donor in donors}
    summary['paired_gaps'][f'direct_atlas_minus_{baseline}'] = {
        'per_donor': per_donor,
        'donor_equal': {metric: float(np.mean([per_donor[donor][metric]
            for donor in donors])) for metric in metrics}}

output_config = {'model': '115', 'step': step,
    'run_completion_sha256': sha(run / 'completed.json'),
    'run_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': run_done['checkpoint_sha256'][str(step)],
    'paired_completion_sha256': sha(paired / 'completed.json'),
    'paired_config_sha256': pair_done['config_sha256'],
    'paired_rows_sha256': pair_done['rows_sha256'],
    'frozen_virtual_identity_sha256': pair_config['frozen_virtual_identity_sha256'],
    'virtual_attempts': attempts, 'virtual_source_manifest_sha256': train['manifest_sha256'],
    'atlas_binding': binding, 'atlas_serial_helper_sha256': helper_sha256,
    'generated_inputs_sha256': hashlib.sha256(json.dumps([
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
    'rows': len(rows), 'sections': 32, 'sources': 3,
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
