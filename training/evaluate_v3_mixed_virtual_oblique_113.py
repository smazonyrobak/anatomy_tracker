"""Frozen 113 checkpoints on virtual TRAIN holdout and separate synthetic/real DEV."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
run = root / 'runs/v3_mixed_virtual_oblique_113'
out = root / 'runs/v3_mixed_virtual_oblique_113_eval'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
steps, side = (0, 1000, 2000, 3000, 4000), 256
torch.set_num_threads(4)
assert not out.exists() and (run / 'completed.json').is_file()
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert done['batches'] == config['batches'] == 4000
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert all(sha(run / f'joint_step_{step:05d}.pt') == done['checkpoint_sha256'][str(step)]
    for step in steps)
assert not any(done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
assert sha(source.parent / 'docs/publication/V3_MIXED_VIRTUAL_OBLIQUE_113_PROTOCOL_20261009.md') == config['protocol_sha256']

draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
holdout_ids = set(map(str, config['virtual_holdout_animal_ids']))
assert len(holdout_ids) == 8 and len(config['virtual_train_animal_ids']) == 55
assert not holdout_ids & set(map(str, config['virtual_train_animal_ids']))
assert not holdout_ids & {str(row['animal_id']) for row in draws
    if row['kind'] in ('coronal_weak_train', 'virtual_oblique_weak_train')}

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
    assert not {row[key] for row in records} & {
        row['base_lineage'][key] for row in synthetic_draws}
synthetic_panel = []
for row in synthetic_rows:
    with np.load(panel / row['file'], allow_pickle=False) as arrays:
        synthetic_panel.append({'image': arrays['inputs'].copy(),
            'state': arrays['target_state'].copy(),
            'reflection': int(arrays['reflection']),
            'valid': arrays['valid_mask'].copy(),
            'animal_id': row['animal_id'],
            'specimen_id': row['specimen_id'],
            'experiment_id': row['experiment_id'],
            'section_id': row['section_id'], 'kind': 'synthetic_dev'})

train = load_reserved_real_train()
by_id = {str(donor['animal_id']): i for i, donor in enumerate(train['donors'])}
virtual_panel, virtual_attempts = [], []
with torch.inference_mode():
    for animal_id in config['virtual_holdout_animal_ids']:
        index = by_id[str(animal_id)]
        for base_seed in config['virtual_holdout_seeds']:
            seed = int(base_seed)
            while True:
                sample = sample_reserved_real_oblique_v3(train, index, seed,
                    side=side, device='cuda')
                used = bool(sample['eligible'][0])
                virtual_attempts.append({'animal_id': animal_id, 'seed': seed,
                    'used': used, 'section_id': sample['provenance'][0]['section_id']})
                if used:
                    break
                seed += 10000
            virtual_panel.append({'image': sample['inputs'][0].cpu().numpy(),
                'state': sample['state'][0].cpu().numpy(),
                'reflection': int(sample['reflection'][0]),
                'valid': sample['valid_mask'][0].cpu().numpy(),
                'animal_id': animal_id,
                'specimen_id': sample['provenance'][0]['specimen_id'],
                'experiment_id': sample['provenance'][0]['experiment_id'],
                'section_id': sample['provenance'][0]['section_id'],
                'kind': 'virtual_oblique_train_holdout'})
assert len(virtual_panel) == 32

coronal_done = json.loads((coronal / 'completed.json').read_text())
for name in ('images.npy', 'geometry.npz', 'records.jsonl'):
    assert sha(coronal / name) == coronal_done['output_sha256'][name]
coronal_records = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
    if row['training_split'] == 'development']
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {str(row['animal_id']) for row in coronal_records} & {
    str(donor['animal_id']) for donor in train['donors']}
sagittal_done = json.loads((sagittal / 'summary.json').read_text())
for name, digest in sagittal_done['output_sha256'].items():
    assert sha(sagittal / name) == digest
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert not {str(row['donor_id']) for row in sagittal_records} & {
    str(donor['animal_id']) for donor in train['donors']}
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
assert not {row['section_id'] for row in sagittal_records} & {
    row['section_id'] for row in draws if row['kind'] == 'sagittal_weak_train'}

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
pixels = five * side
rows = []
with torch.inference_mode():
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
            map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and checkpoint['config'] == config
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
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            state = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            chart = chart16 if item['kind'] == 'synthetic_dev' else five
            discrepancy = (rigid_points_090(state, beam % 2, chart)
                - rigid_points_090(truth_state, truth_reflection, chart)[:, None]
                ).norm(dim=-1)
            if item['kind'] == 'synthetic_dev':
                valid16 = (F.interpolate(torch.from_numpy(item['valid'][None, None]).cuda().float(),
                    (16, 16), mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
                discrepancy = discrepancy[..., valid16]
            error = discrepancy.mean(-1)[0] / 1000
            branch = int(prior[0].argmax())
            rows.append({'step': step, 'kind': item['kind'],
                'animal_id': item['animal_id'], 'section_id': item['section_id'],
                'specimen_id': item['specimen_id'],
                'experiment_id': item['experiment_id'],
                'selected_branch': branch,
                'selected_error_mm': float(error[(beam[0] == branch).nonzero()[0, 0]]),
                'best14_error_mm': float(error.min())})
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
                else:
                    affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                        device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
                    animal_id = record['donor_id']
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                branch = int(prior[0].argmax())
                state = prediction['state'][:, branch // 2]
                proposed = rigid_points_090(state,
                    torch.tensor([branch % 2], device='cuda'), five)[0]
                rows.append({'step': step, 'kind': domain,
                    'animal_id': animal_id, 'section_id': record['section_id'],
                    'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'selected_branch': branch,
                    'selected_error_mm': float((proposed - reference).norm(dim=-1).mean() / 1000)})
        print(json.dumps({'step': step, 'evaluated_sections': len(rows)}), flush=True)

summary = {'role': 'virtual TRAIN-subset novel cuts plus reused synthetic/weak real DEV guardrails',
    'virtual_holdout_parent_exposure_caveat': config['virtual_holdout_role'],
    'steps': {}}
for step in steps:
    groups = {}
    for kind in ('synthetic_dev', 'virtual_oblique_train_holdout',
                 'coronal_weak_dev', 'sagittal_weak_dev'):
        group = [row for row in rows if row['step'] == step and row['kind'] == kind]
        donor_ids = sorted({str(row['animal_id']) for row in group})
        by_donor = {animal_id: [row for row in group
            if str(row['animal_id']) == animal_id] for animal_id in donor_ids}
        metrics = (('selected_error_mm', 'best14_error_mm')
            if kind in ('synthetic_dev', 'virtual_oblique_train_holdout')
            else ('selected_error_mm',))
        groups[kind] = {'sections': len(group), 'donors': len(donor_ids),
            'donor_equal': {metric: float(np.mean([
                np.mean([row[metric] for row in subset]) for subset in by_donor.values()]))
                for metric in metrics},
            'selected_over_3mm': sum(row['selected_error_mm'] > 3 for row in group)}
    summary['steps'][str(step)] = groups
out.mkdir(parents=True, exist_ok=False)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'run_completion_sha256': sha(run / 'completed.json'),
    'source_manifest_sha256': train['manifest_sha256'],
    'virtual_attempts': virtual_attempts,
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'evaluator_sha256': sha(__file__),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
