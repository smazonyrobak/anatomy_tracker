"""Frozen stage-1 DEV readout of the 176 blind-beam fitter and ranker."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.gradient_guided_plane_fit_176 import FitEvidenceRanker176, GradientGuidedPlaneFitter176


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GRADIENT_GUIDED_BLIND_BEAM_176_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
train = root / 'runs/gradient_guided_blind_beam_176_train_001'
out = root / 'runs/gradient_guided_blind_beam_176_stage1_dev_eval_001'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
steps = (0, 500)
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists() and torch.cuda.is_available()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert train_done['start_update'] == train_config['start_update'] == 0
assert train_done['end_update'] == train_config['end_update'] == 500
assert train_config['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['protocol_sha256'] == sha(protocol)
assert sha(train / 'config.json') == train_done['config_sha256']
assert all(sha(train / name) == digest for name, digest in train_done['output_sha256'].items())
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert not any(receipt.get(key, False) for receipt in (parent_done, train_done) for key in
    ('calibrated', 'expert_real_truth_used', 'final_animals_used',
     'public_benchmark_used', 'external_pretrained_weights_used'))
assert train_done['real_training_images'] == train_config['real_training_images'] == 0
checkpoints = {step: train / f'ranker_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == train_done['output_sha256'][path.name] for path in checkpoints.values())

panel_records, panel_done = {}, {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    panel_protocol = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in panel_protocol['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == done['eligible'] == (248 if cohort == 'v4' else 243)
    assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[cohort], panel_done[cohort] = records, done
assert panel_done['v4']['plan_completed_sha256'] == panel_done['v3']['plan_completed_sha256']
assert not {row['panel_physical_section_id'] for row in panel_records['v4']} & {
    row['panel_physical_section_id'] for row in panel_records['v3']}

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_records = [row for row in coronal_all if row['training_split'] == 'development']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {row['animal_id'] for row in coronal_records} & {
    row['animal_id'] for row in coronal_all if row['training_split'] == 'train'}
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
assert hashlib.sha256(memoryview(atlas_array).cast('B')).hexdigest() == allen.ATLAS_FLOAT32_RECEIPT_V6['sha256']
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, _
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side
weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda', dtype=torch.float32) / 16

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['config'] == parent_config and parent['calibrated'] is False
model.load_state_dict(parent['model'], strict=True)
del parent
fitter = GradientGuidedPlaneFitter176().cuda().eval()
rankers = {}
for step, checkpoint in checkpoints.items():
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['step'] == step and saved['config'] == train_config and saved['calibrated'] is False
    rankers[step] = FitEvidenceRanker176().cuda().eval().requires_grad_(False)
    rankers[step].load_state_dict(saved['ranker'], strict=True)
    del saved


def evaluate_case(image, offsets, mass, reference, sites, metadata):
    with torch.no_grad():
        prediction = model.predict(image)
        prior, beam = beam_for(prediction)
        ids = beam[0]
        states = prediction['state'][:, ids // 2]
        reflection = (ids % 2)[None]
        logits = prior[:, ids]

    def error(candidate):
        return (rigid_points_090(candidate, reflection[0], sites) - reference).norm(dim=-1).mean(-1) / 1000

    original = error(states[0]).detach()
    prior_slot = int(logits[0].argmax())
    common = {**metadata, 'beam_branch_ids': ids.tolist(), 'beam_prior_logits': logits[0].tolist(),
        'original16_mm': original.tolist(), 'prior_selected_branch_id': int(ids[prior_slot]),
        'prior_selected_original_mm': float(original[prior_slot]),
        'prior_selected_capture_1p5': bool(original[prior_slot] <= 1.5),
        'original_best16_mm': float(original.min()),
        'original_best16_capture_1p5': bool(original.min() <= 1.5)}
    arms = {}
    for name, intensity_null in (('full', False), ('permuted', True)):
        fitted = fitter(image, states, reflection, offsets, mass, atlas, intensity_null=intensity_null)
        corrected = error(fitted['state'][0]).detach()
        valid = fitted['support_sites'][0] >= 100
        arms[name] = (fitted, corrected, valid)
    assert torch.equal(arms['full'][0]['support_sites'], arms['permuted'][0]['support_sites'])

    rows = []
    for step, ranker in rankers.items():
        row = {**common, 'step': step}
        for name, (fitted, corrected, valid) in arms.items():
            with torch.no_grad():
                action_logits = ranker(logits, fitted)[0]
            choice = int(action_logits.flatten().argmax())
            slot, action = divmod(choice, 2)
            available = torch.stack((original, corrected.masked_fill(~valid, float('inf'))), -1)
            row.update({
                f'{name}_corrected16_mm': corrected.tolist(),
                f'{name}_original_fit16': fitted['original_fit'][0].tolist(),
                f'{name}_corrected_fit16': fitted['fit'][0].tolist(),
                f'{name}_support_sites16': fitted['support_sites'][0].tolist(),
                f'{name}_support_retention16': fitted['support_retention'][0].tolist(),
                f'{name}_gradient_norm16': fitted['gradient_norm'][0].tolist(),
                f'{name}_local_update16': fitted['local_update'][0].tolist(),
                f'{name}_action_logits32': action_logits.flatten().tolist(),
                f'{name}_valid_corrected_actions': int(valid.sum()),
                f'{name}_selected_action_index': choice,
                f'{name}_selected_branch_id': int(ids[slot]),
                f'{name}_selected_no_shift': action == 0,
                f'{name}_selected_mm': float(available[slot, action]),
                f'{name}_selected_capture_1p5': bool(available[slot, action] <= 1.5),
                f'{name}_best_available32_mm': float(available.min())})
        if step == 0:
            assert all(row[f'{name}_selected_no_shift'] and
                       abs(row[f'{name}_selected_mm'] - row['prior_selected_original_mm']) < 1e-5
                       for name in ('full', 'permuted'))
        rows.append(row)
    return rows


synthetic_rows, real_rows = [], []
for cohort, panel in panels.items():
    for record in (row for row in panel_records[cohort] if row['eligible']):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            mass = torch.from_numpy(arrays['weights'].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        sites = chart[valid]
        reference = rigid_points_090(truth, reflection, sites)[0]
        synthetic_rows.extend(evaluate_case(image, offsets, mass, reference, sites, {
            'cohort': cohort, 'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'subject_plan_id': record['synthetic_subject_plan_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'panel_file_sha256': record['sha256'],
            'appearance_mode': record['appearance_mode'],
            'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1))))}))
    print(json.dumps({'event': 'synthetic_completed', 'cohort': cohort}), flush=True)

for family, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
    for record in records:
        index = record['array_row_index']
        if family == 'coronal':
            native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(coronal_affines[index], device='cuda', dtype=torch.float32)
            old_pixels = (five_pixels + .5) * (192 / side) - .5
            reference = affine[:, 2] + old_pixels[:, :1] * affine[:, 0] + old_pixels[:, 1:] * affine[:, 1]
            thickness = float(record['section_thickness_um'])
            donor = record['animal_id']
        else:
            native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                np.zeros((4, side, side), dtype=np.float32)))[None]
            image = torch.from_numpy(native).cuda()
            affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                     device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + five_pixels[:, :1] * affine[:, 0] + five_pixels[:, 1:] * affine[:, 1]
            thickness = 50.
            donor = record['donor_id']
        offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * thickness
        real_rows.extend(evaluate_case(image, offsets, weights, reference, five_chart, {
            'family': family, 'donor_id': donor, 'section_id': record['section_id'],
            'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
            'array_row_index': index,
            'acquired_image_sha256': record['source_image_sha256'] if family == 'coronal'
                else record['image_sha256'],
            'render_thickness_um': thickness,
            'label_role': 'inherited weak Allen affine, not expert truth'}))
    print(json.dumps({'event': 'weak_real_completed', 'family': family}), flush=True)

assert len(synthetic_rows) == len(steps) * sum(done['eligible'] for done in panel_done.values())
assert len(real_rows) == len(steps) * (len(coronal_records) + len(sagittal_records))
metrics = ('prior_selected_original_mm', 'original_best16_mm', 'full_selected_mm',
           'permuted_selected_mm', 'full_best_available32_mm', 'permuted_best_available32_mm')


def summarise(rows):
    return {'sections': len(rows),
        'mean_mm': {key: float(np.mean([row[key] for row in rows])) if rows else None
            for key in metrics},
        'capture_le_1p5': {key: float(np.mean([row[key] <= 1.5 for row in rows]))
            if rows else None for key in metrics},
        'no_shift_fraction': {name: float(np.mean([row[f'{name}_selected_no_shift'] for row in rows]))
            if rows else None for name in ('full', 'permuted')}}


synthetic, real = {}, {}
for cohort in panels:
    synthetic[cohort] = {}
    for step in steps:
        rows = [row for row in synthetic_rows if row['cohort'] == cohort and row['step'] == step]
        by_plan = {str(plan): summarise([row for row in rows if row['subject_plan_id'] == plan])
                   for plan in sorted({row['subject_plan_id'] for row in rows})}
        synthetic[cohort][str(step)] = {'section_weighted': summarise(rows),
            'plan_equal_mean_mm': {key: float(np.mean([part['mean_mm'][key]
                for part in by_plan.values()])) for key in metrics},
            'plan_equal_capture_le_1p5': {key: float(np.mean([part['capture_le_1p5'][key]
                for part in by_plan.values()])) for key in metrics},
            'by_plan': by_plan,
            'by_appearance_mode': {mode: summarise([row for row in rows
                if row['appearance_mode'] == mode]) for mode in ('raw', 'exact_black', 'imperfect_brush')},
            'by_angle_deg': {f'[{low},{high})': summarise([row for row in rows
                if low <= row['nearest_cardinal_angle_deg'] < high])
                for low, high in ((0, 15), (15, 30), (30, 45), (45, 55))}}
for family in ('coronal', 'sagittal'):
    real[family] = {}
    for step in steps:
        rows = [row for row in real_rows if row['family'] == family and row['step'] == step]
        by_donor = {str(donor): summarise([row for row in rows if row['donor_id'] == donor])
                    for donor in sorted({row['donor_id'] for row in rows})}
        real[family][str(step)] = {'section_weighted': summarise(rows),
            'donor_equal_mean_mm': {key: float(np.mean([part['mean_mm'][key]
                for part in by_donor.values()])) for key in metrics},
            'donor_equal_capture_le_1p5': {key: float(np.mean([part['capture_le_1p5'][key]
                for part in by_donor.values()])) for key in metrics},
            'by_donor': by_donor}

summary = {'scope': 'frozen v4/v3 synthetic DEV; donor-disjoint acquired weak-affine DEV is secondary',
    'beam': 'frozen 132 blind16: top8 base, top6 anchor, two diverse anchor reflections',
    'metric': 'synthetic mean 3D rigid error on observed valid pixels, mm; acquired weak-affine five-point disagreement, mm',
    'control': 'same atlas support and beam, within-initial-support permutation of rendered atlas intensity',
    'original_best16_mm': 'truth-chosen oracle availability only, not an inference selector',
    'synthetic': synthetic, 'real_weak_affine': real,
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
summary['paired_effects'] = {cohort: {str(step): {
    'full_selected_gain_mm': synthetic[cohort][str(step)]['section_weighted']['mean_mm']['prior_selected_original_mm'] -
        synthetic[cohort][str(step)]['section_weighted']['mean_mm']['full_selected_mm'],
    'full_capture_gain': synthetic[cohort][str(step)]['section_weighted']['capture_le_1p5']['full_selected_mm'] -
        synthetic[cohort][str(step)]['section_weighted']['capture_le_1p5']['prior_selected_original_mm'],
    'intensity_advantage_mm': synthetic[cohort][str(step)]['section_weighted']['mean_mm']['permuted_selected_mm'] -
        synthetic[cohort][str(step)]['section_weighted']['mean_mm']['full_selected_mm']}
    for step in steps} for cohort in panels}
config = {'evaluator_source_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'training_completion_sha256': sha(train / 'completed.json'),
    'training_config_sha256': sha(train / 'config.json'),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoints.items()},
    'fit_source_sha256': sha(source / 'gradient_guided_plane_fit_176.py'),
    'panel_completion_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'panel_protocol_sha256': {cohort: sha(panel / 'protocol.json') for cohort, panel in panels.items()},
    'panel_records_sha256': {cohort: sha(panel / 'records.jsonl') for cohort, panel in panels.items()},
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_train_geometry_sha256': sha(sagittal_train / 'geometry.jsonl'),
    'atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'real_coronal_resize': 'align_corners=False: native192 pixel=(model256 pixel+0.5)*192/256-0.5',
    'real_sagittal_render_thickness_um': 50.,
    'real_sagittal_render_thickness_role': 'fixed sensitivity assumption; absent from source geometry',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, rows in (('synthetic_rows.jsonl', synthetic_rows), ('real_weak_rows.jsonl', real_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'eligible_v4_sections': panel_done['v4']['eligible'],
    'eligible_v3_sections': panel_done['v3']['eligible'],
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'synthetic_rows.jsonl', 'real_weak_rows.jsonl')},
    'evaluator_source_sha256': config['evaluator_source_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'output': str(out)}, allow_nan=False), flush=True)
