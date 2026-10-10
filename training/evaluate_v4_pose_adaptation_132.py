"""Matched frozen DEV direct-pose evaluation of 128 and 132 checkpoints."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V4_POSE_ADAPTATION_132_PROTOCOL_20261010.md'
parent_dir = root / 'runs/joint_in_path_correspondence_128_treatment'
parent_path = parent_dir / 'joint_step_06000.pt'
train = root / 'runs/v4_pose_adaptation_132'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/v4_pose_adaptation_132_dev_eval'
arms = ('parent', '0', '500', '1000', '2000')
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['6000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256']
assert train_config['parent_checkpoint_sha256'] == train_done['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['protocol_sha256'] == train_done['protocol_sha256'] == sha(protocol)
assert train_config['checkpoints'] == [0, 500, 1000, 2000]
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(train_done.get(key, False) or parent_done.get(key, False) for key in
    ('calibrated', 'expert_real_truth_used', 'final_animals_used',
     'public_benchmark_used', 'external_pretrained_weights_used'))
checkpoints = {'parent': parent_path, **{str(step): train / f'joint_step_{step:05d}.pt'
    for step in (0, 500, 1000, 2000)}}
assert all(sha(path) == (parent_done['checkpoint_sha256']['6000'] if arm == 'parent'
    else train_done['checkpoint_sha256'][arm]) for arm, path in checkpoints.items())

panel_records, panel_done = {}, {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == done['eligible']
    assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
    assert len({row['panel_physical_section_id'] for row in records}) == 256
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[cohort], panel_done[cohort] = records, done
assert panel_done['v4']['plan_completed_sha256'] == panel_done['v3']['plan_completed_sha256']
assert not {row['panel_physical_section_id'] for row in panel_records['v4']} & {
    row['panel_physical_section_id'] for row in panel_records['v3']}
assert not {row['section_id'] for row in panel_records['v4']} & {
    row['section_id'] for row in panel_records['v3']}

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
sagittal_train_records = [json.loads(line) for line in
    (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side


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


model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
synthetic_rows, real_rows = [], []
with torch.inference_mode():
    for arm, checkpoint_path in checkpoints.items():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert saved['step'] == (6000 if arm == 'parent' else int(arm))
        assert saved['config'] == (parent_config if arm == 'parent' else train_config)
        assert saved['calibrated'] is False
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for cohort, panel in panels.items():
            for record in (row for row in panel_records[cohort] if row['eligible']):
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).cuda().bool()
                    truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                assert int(valid.sum()) == record['valid_pixels']
                target = rigid_points_090(truth, reflection, chart)[0]
                prediction = model.predict(image)
                prior, beam = beam_for(prediction)
                ids = beam[0]
                proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                            (ids % 2)[None], chart)[0]
                costs = ((proposed - target).norm(dim=-1)[:, valid].mean(-1) / 1000).tolist()
                top_id = int(prior[0].argmax())
                assert top_id in ids.tolist()
                tissue = image[0, 0].flatten()[valid]
                exposure = (record['provenance']['one_shot_slide_artifacts_v4']['exposure']
                            if cohort == 'v4' else 1.)
                synthetic_rows.append({'arm': arm, 'cohort': cohort,
                    'section_id': record['section_id'],
                    'physical_section_id': record['panel_physical_section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'panel_file_sha256': record['sha256'],
                    'appearance_mode': record['appearance_mode'],
                    'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                        np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
                    'applied_exposure_multiplier': exposure,
                    'observed_valid_tissue_mean': float(tissue.mean()),
                    'valid_pixels': int(valid.sum()),
                    'beam_branch_ids': ids.tolist(), 'beam_rigid_mm': costs,
                    'beam_best_rigid_mm': min(costs), 'top1_branch_id': top_id,
                    'top1_rigid_mm': costs[ids.tolist().index(top_id)]})
            print(json.dumps({'event': 'synthetic_completed', 'arm': arm,
                              'cohort': cohort, 'eligible': panel_done[cohort]['eligible']}), flush=True)

        for family, records in (('coronal', coronal_records),
                                ('sagittal', sagittal_records)):
            for record in records:
                index = record['array_row_index']
                if family == 'coronal':
                    native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                        mode='bilinear', align_corners=False)
                    affine = torch.as_tensor(coronal_affines[index], device='cuda',
                                             dtype=torch.float32)
                    old_pixels = (five_pixels + .5) * (192 / side) - .5
                    reference = affine[:, 2] + (
                        old_pixels[:, :1] * affine[:, 0] + old_pixels[:, 1:] * affine[:, 1])
                    donor = record['animal_id']
                else:
                    native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                        np.zeros((4, side, side), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                             device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + (
                        five_pixels[:, :1] * affine[:, 0] + five_pixels[:, 1:] * affine[:, 1])
                    donor = record['donor_id']
                prediction = model.predict(image)
                prior, _ = beam_for(prediction)
                top_id = int(prior[0].argmax())
                proposed = rigid_points_090(prediction['state'][:, top_id // 2],
                    torch.tensor([top_id % 2], device='cuda'), five_chart)[0]
                real_rows.append({'arm': arm, 'family': family, 'donor_id': donor,
                    'section_id': record['section_id'], 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'top1_branch_id': top_id,
                    'direct_top1_weak_five_point_mm': float(
                        (proposed - reference).norm(dim=-1).mean() / 1000),
                    'label_role': 'inherited weak Allen affine, not expert truth'})
            print(json.dumps({'event': 'weak_real_completed', 'arm': arm,
                              'family': family, 'sections': len(records)}), flush=True)

assert len(synthetic_rows) == len(arms) * sum(done['eligible'] for done in panel_done.values())
assert len(real_rows) == len(arms) * (len(coronal_records) + len(sagittal_records))
for cohort in panels:
    parent_rows = {row['section_id']: row for row in synthetic_rows
                   if row['cohort'] == cohort and row['arm'] == 'parent'}
    zero_rows = {row['section_id']: row for row in synthetic_rows
                 if row['cohort'] == cohort and row['arm'] == '0'}
    assert parent_rows.keys() == zero_rows.keys()
    assert all(abs(parent_rows[key][metric] - zero_rows[key][metric]) < 1e-5
               for key in parent_rows for metric in ('beam_best_rigid_mm', 'top1_rigid_mm'))
parent_real = {(row['family'], row['section_id']): row for row in real_rows
               if row['arm'] == 'parent'}
zero_real = {(row['family'], row['section_id']): row for row in real_rows
             if row['arm'] == '0'}
assert parent_real.keys() == zero_real.keys()
assert all(abs(parent_real[key]['direct_top1_weak_five_point_mm']
               - zero_real[key]['direct_top1_weak_five_point_mm']) < 1e-5
           for key in parent_real)


def summarise(group):
    return {'sections': len(group),
        'mean_mm': {key: float(np.mean([row[key] for row in group])) if group else None
            for key in ('beam_best_rigid_mm', 'top1_rigid_mm')},
        'capture_le_1p5': {key: float(np.mean([row[key] <= 1.5 for row in group]))
            if group else None for key in ('beam_best_rigid_mm', 'top1_rigid_mm')}}


synthetic = {}
for cohort in panels:
    synthetic[cohort] = {}
    for arm in arms:
        rows = [row for row in synthetic_rows if row['cohort'] == cohort and row['arm'] == arm]
        plans = sorted({row['subject_plan_id'] for row in rows})
        by_plan = {plan: summarise([row for row in rows if row['subject_plan_id'] == plan])
                   for plan in plans}
        by_mode = {mode: summarise([row for row in rows if row['appearance_mode'] == mode])
                   for mode in ('raw', 'exact_black', 'imperfect_brush')}
        by_angle = {f'[{low},{high})': summarise([row for row in rows
            if low <= row['nearest_cardinal_angle_deg'] < high])
            for low, high in ((0, 15), (15, 30), (30, 45), (45, 55))}
        by_exposure = {f'[{low},{high})': summarise([row for row in rows
            if low <= row['applied_exposure_multiplier'] < high])
            for low, high in ((0, .15), (.15, .30), (.30, .60), (.60, 1.21))}
        synthetic[cohort][arm] = {'section_weighted': summarise(rows),
            'plan_equal': {'mean_mm': {key: float(np.mean([
                part['mean_mm'][key] for part in by_plan.values()]))
                for key in ('beam_best_rigid_mm', 'top1_rigid_mm')},
                'capture_le_1p5': {key: float(np.mean([
                    part['capture_le_1p5'][key] for part in by_plan.values()]))
                    for key in ('beam_best_rigid_mm', 'top1_rigid_mm')}},
            'by_plan': by_plan, 'by_mode': by_mode,
            'by_angle_deg': by_angle, 'by_exposure': by_exposure}

real = {}
for family in ('coronal', 'sagittal'):
    donors = sorted({row['donor_id'] for row in real_rows if row['family'] == family})
    real[family] = {'donors': len(donors), 'sections': sum(row['family'] == family
        for row in real_rows if row['arm'] == 'parent'), 'by_donor': {},
        'donor_equal_mean_mm': {}}
    for donor in donors:
        real[family]['by_donor'][str(donor)] = {'sections': sum(
            row['donor_id'] == donor and row['family'] == family and row['arm'] == 'parent'
            for row in real_rows), 'mean_mm': {arm: float(np.mean([
                row['direct_top1_weak_five_point_mm'] for row in real_rows
                if row['donor_id'] == donor and row['family'] == family and row['arm'] == arm]))
                for arm in arms}}
    real[family]['donor_equal_mean_mm'] = {arm: float(np.mean([
        donor['mean_mm'][arm] for donor in real[family]['by_donor'].values()])) for arm in arms}

baseline_v4 = synthetic['v4']['parent']['section_weighted']['capture_le_1p5']
baseline_v3 = synthetic['v3']['parent']['section_weighted']['capture_le_1p5']
gate = {}
for arm in arms[1:]:
    v4 = synthetic['v4'][arm]['section_weighted']['capture_le_1p5']
    v3 = synthetic['v3'][arm]['section_weighted']['capture_le_1p5']
    conditions = {
        'v4_blind16_gain_ge_0p15': v4['beam_best_rigid_mm'] - baseline_v4['beam_best_rigid_mm'] >= .15,
        'v4_direct_top1_gain_ge_0p10': v4['top1_rigid_mm'] - baseline_v4['top1_rigid_mm'] >= .10,
        'v3_blind16_drop_le_0p05': v3['beam_best_rigid_mm'] >= baseline_v3['beam_best_rigid_mm'] - .05,
        'each_weak_real_donor_direct_top1_regression_le_0p20_mm': all(
            donor['mean_mm'][arm] <= donor['mean_mm']['parent'] + .20
            for family in real.values() for donor in family['by_donor'].values())}
    gate[arm] = {'conditions': conditions, 'passed': all(conditions.values()),
        'development_only': True}

summary = {'scope': 'fresh independent v4 sections on reused eight synthetic DEV plans; existing v3 DEV; donor-disjoint acquired weak affine DEV only',
    'metric': 'mean 3D rigid-gauge error over observed valid tissue pixels, mm; rigid_points_090 and 256-grid x/W,y/H as frozen 128',
    'beam': 'frozen 128 direct-pose blind16: top8 base, top6 anchor, two antipodal-normal diversity additions',
    'primary_gate_weighting': 'section-weighted eligible frozen sections; weak-real each donor mean direct-top1 five-point disagreement',
    'synthetic': synthetic, 'real_weak_affine': real, 'advance_gate': gate,
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
config = {'evaluator_source_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'training_completion_sha256': sha(train / 'completed.json'),
    'training_config_sha256': sha(train / 'config.json'),
    'checkpoint_sha256': {arm: sha(path) for arm, path in checkpoints.items()},
    'panel_completion_sha256': {cohort: sha(panel / 'completed.json')
        for cohort, panel in panels.items()},
    'panel_protocol_sha256': {cohort: sha(panel / 'protocol.json')
        for cohort, panel in panels.items()},
    'panel_records_sha256': {cohort: sha(panel / 'records.jsonl')
        for cohort, panel in panels.items()},
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'input_resolution': side, 'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'real_coronal_resize': 'align_corners=False: native192 pixel=(model256 pixel+0.5)*192/256-0.5',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, rows in (('synthetic_rows.jsonl', synthetic_rows),
                   ('real_weak_rows.jsonl', real_rows)):
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
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'advance_gate': gate}, allow_nan=False), flush=True)
