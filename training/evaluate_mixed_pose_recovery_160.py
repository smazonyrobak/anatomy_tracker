"""Frozen blind-pose, weak-real, and supplied-plane map readout after run 160 exits."""

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
    return prior[0], beam[0]


train = root / 'runs/mixed_pose_recovery_160_train_001'
out = root / 'runs/mixed_pose_recovery_160_dev_eval_001'
protocol = Path(__file__).resolve().parent.parent / 'docs/publication/MIXED_POSE_RECOVERY_160_PROTOCOL_20261011.md'
panels = {'v4': root / 'data/coarse_atlas_pose_150_fresh_dev_panel',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
arms = ('0', '326', '653')
checkpoints = {arm: train / f'joint_step_{int(arm):05d}.pt' for arm in arms}
source = Path(__file__).resolve().parent
assert all(path.drive.upper() == 'I:' for path in (*panels.values(), train, out, source))
assert not out.exists()
done = json.loads((train / 'completed.json').read_text())
config160 = json.loads((train / 'config.json').read_text())
assert sha(train / 'config.json') == done['config_sha256']
assert sha(train / 'draws.jsonl') == done['draws_sha256']
assert sha(train / 'training.jsonl') == done['training_sha256']
assert sha(protocol) == config160['protocol_sha256']
assert sha(protocol) == done['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in config160['source_sha256'].items())
assert all(sha(path) == done['checkpoint_sha256'][arm] for arm, path in checkpoints.items())
assert not any(done.get(key, False) or config160.get(key, False) for key in
               ('calibrated', 'public_benchmark_used', 'final_animals_used',
                'expert_real_truth_used', 'external_pretrained_weights_used'))

panel_records, panel_receipts = {}, {}
for cohort, folder in panels.items():
    receipt = json.loads((folder / 'completed.json').read_text())
    frozen = json.loads((folder / 'protocol.json').read_text())
    records = [json.loads(line) for line in (folder / 'records.jsonl').open()]
    assert len(records) == receipt['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == receipt['eligible']
    assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert sha(folder / 'protocol.json') == receipt['protocol_sha256']
    assert sha(folder / 'records.jsonl') == receipt['records_sha256']
    assert all(sha(folder / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert all(sha(folder / row['file']) == row['sha256'] for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
    panel_receipts[cohort] = sha(folder / 'completed.json')
assert not {r['section_id'] for r in panel_records['v4']} & {
    r['section_id'] for r in panel_records['v3']}

map_ids = set()
for plan in sorted({r['synthetic_subject_plan_id'] for r in panel_records['v4']}):
    group = sorted((r for r in panel_records['v4']
                    if r['synthetic_subject_plan_id'] == plan), key=lambda r: r['section_id'])
    map_ids.update(group[int(i)]['section_id'] for i in np.linspace(0, len(group) - 1, 8))
assert len(map_ids) == 64

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_records = [r for r in coronal_all if r['training_split'] == 'development']
assert len(coronal_records) == 64 and len({r['animal_id'] for r in coronal_records}) == 6
assert not {r['animal_id'] for r in coronal_records} & {
    r['animal_id'] for r in coronal_all if r['training_split'] == 'train'}
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({r['donor_id'] for r in sagittal_records}) == 8
assert all(r['split'] == 'weak_dev2_source_available' for r in sagittal_records)
assert not {r['donor_id'] for r in sagittal_records} & {
    r['donor_id'] for r in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
axis = torch.arange(256, device='cuda') / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / 256

config = {'evaluator_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
          'training_completion_sha256': sha(train / 'completed.json'),
          'training_config_sha256': sha(train / 'config.json'),
          'training_draws_sha256': sha(train / 'draws.jsonl'),
          'training_curve_sha256': sha(train / 'training.jsonl'),
          'checkpoint_sha256': {arm: sha(path) for arm, path in checkpoints.items()},
          'panel_completion_sha256': panel_receipts,
          'panel_sections': {cohort: {'recorded': 256, 'eligible': len(rows),
                                      'ineligible': 256 - len(rows)}
                             for cohort, rows in panel_records.items()},
          'coronal_completion_sha256': sha(coronal / 'completed.json'),
          'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
          'map_subset_v4_section_ids': sorted(map_ids),
          'map_subset_rule': 'eight evenly indexed eligible records by section_id per frozen synthetic DEV plan',
          'blind16': '132 prior top8 free + top6 anchor + two antipodal-normal diversity anchors; truth-best is diagnostic only',
          'synthetic_metric': 'mean rigid CCF distance in mm over all surviving valid native-256 tissue pixels',
          'map_metric': 'mean frozen local-map CCF distance in mm at valid native-256 [2::4,2::4] sites',
          'weak_real_metric': 'five full-frame Allen-affine points; inherited weak labels, not expert anatomy',
          'weak_coronal_resize': '(model256 pixel + .5) * 192/256 - .5, align_corners=False',
          'calibrated': False, 'public_benchmark_used': False,
          'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
synthetic_rows, map_rows, real_rows, baseline_near = [], [], [], {}
with torch.inference_mode(), (out / 'synthetic_rows.jsonl').open('w') as synthetic_stream, \
        (out / 'map_rows.jsonl').open('w') as map_stream, \
        (out / 'real_weak_rows.jsonl').open('w') as real_stream:
    for arm, path in checkpoints.items():
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == int(arm) and saved['config'] == config160
        assert saved['calibrated'] is False
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for cohort, folder in panels.items():
            for case in panel_records[cohort]:
                with np.load(folder / case['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    if cohort == 'v4' and case['section_id'] in map_ids:
                        truth_surface = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                assert int(valid.sum()) == case['valid_pixels']
                prediction = model.predict(image)
                prior, ids = beam_for(prediction)
                observed = chart[valid]
                target = rigid_points_090(truth_state, truth_reflection, observed)[0]
                proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                            (ids % 2)[None], observed)[0]
                errors = ((proposed - target).norm(dim=-1).mean(-1) / 1000).tolist()
                first_id = int(prior.argmax())
                assert first_id in ids.tolist()
                angle = float(np.degrees(np.arccos(np.clip(
                    np.max(np.abs(case['plane_normal_ap_dv_ml'])), 0, 1))))
                row = {'arm': arm, 'cohort': cohort, 'section_id': case['section_id'],
                       'physical_section_id': case['panel_physical_section_id'],
                       'synthetic_subject_plan_id': case['synthetic_subject_plan_id'],
                       'animal_id': case['animal_id'], 'specimen_id': case['specimen_id'],
                       'experiment_id': case['experiment_id'], 'panel_file_sha256': case['sha256'],
                       'appearance_mode': case['appearance_mode'], 'valid_pixels': int(valid.sum()),
                       'valid_fraction': float(valid.float().mean()),
                       'nearest_cardinal_angle_deg': angle, 'beam_branch_ids': ids.tolist(),
                       'beam_rigid_mm': errors, 'top1_branch_id': first_id,
                       'top1_rigid_mm': errors[ids.tolist().index(first_id)],
                       'top16_truth_best_rigid_mm': min(errors)}
                synthetic_rows.append(row)
                synthetic_stream.write(json.dumps(row, allow_nan=False) + '\n')
                if cohort == 'v4' and case['section_id'] in map_ids:
                    key = case['section_id']
                    if arm == '0':
                        near_slot = int(np.argmin(errors))
                        near_id = int(ids[near_slot])
                        baseline_near[key] = (prediction['state'][0, near_id // 2].detach().clone(),
                                              near_id % 2, errors[near_slot])
                    near_state, near_reflection, near_mm = baseline_near[key]
                    states = torch.stack((truth_state[0], near_state), 0)[None]
                    reflections = torch.tensor([[int(truth_reflection[0]), near_reflection]], device='cuda')
                    mapped = model.map({**prediction, 'state': states}, offsets,
                        torch.tensor([[0, 1]], device='cuda'), reflections,
                        (256, 256), atlas, weights)
                    surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0, :, 2::4, 2::4]
                    valid64 = valid[2::4, 2::4]
                    target64 = truth_surface[2::4, 2::4]
                    distances = (surface - target64).norm(dim=-1) / 1000
                    for role, slot in (('exact', 0), ('baseline_near', 1)):
                        mapped_row = {'arm': arm, 'role': role, 'cohort': 'v4',
                                      'section_id': key, 'synthetic_subject_plan_id': case['synthetic_subject_plan_id'],
                                      'animal_id': case['animal_id'], 'specimen_id': case['specimen_id'],
                                      'experiment_id': case['experiment_id'],
                                      'panel_file_sha256': case['sha256'],
                                      'valid_sites': int(valid64.sum()),
                                      'baseline_near_rigid_mm': float(near_mm),
                                      'mapped_valid_mean_mm': float(distances[slot][valid64].mean())}
                        map_rows.append(mapped_row)
                        map_stream.write(json.dumps(mapped_row, allow_nan=False) + '\n')
            print(json.dumps({'event': 'synthetic_completed', 'arm': arm,
                              'cohort': cohort, 'eligible': len(panel_records[cohort])}), flush=True)
        for family, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
            for case in records:
                index = case['array_row_index']
                if family == 'coronal':
                    native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                                             np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (256, 256),
                                          mode='bilinear', align_corners=False)
                    affine = torch.as_tensor(coronal_affines[index], device='cuda', dtype=torch.float32)
                    old_pixels = (five_pixels + .5) * (192 / 256) - .5
                    reference = affine[:, 2] + old_pixels[:, :1] * affine[:, 0] + old_pixels[:, 1:] * affine[:, 1]
                    donor = case['animal_id']
                else:
                    native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                                             np.zeros((4, 256, 256), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(case['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                             device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + five_pixels[:, :1] * affine[:, 0] + five_pixels[:, 1:] * affine[:, 1]
                    donor = case['donor_id']
                prediction = model.predict(image)
                prior, ids = beam_for(prediction)
                proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                            (ids % 2)[None], five_chart)[0]
                errors = ((proposed - reference).norm(dim=-1).mean(-1) / 1000).tolist()
                first_id = int(prior.argmax())
                weak_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                normals = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
                normal_deg = torch.rad2deg((normals @ weak_normal).abs().clamp(0, 1).acos())
                row = {'arm': arm, 'family': family, 'donor_id': donor,
                       'section_id': case['section_id'], 'specimen_id': case['specimen_id'],
                       'experiment_id': case['experiment_id'], 'array_row_index': index,
                       'beam_branch_ids': ids.tolist(), 'beam_weak_five_point_mm': errors,
                       'top1_branch_id': first_id,
                       'top1_weak_five_point_mm': errors[ids.tolist().index(first_id)],
                       'top16_truth_best_weak_five_point_mm': min(errors),
                       'top1_weak_normal_deg': float(normal_deg[first_id // 2]),
                       'label_role': 'inherited weak Allen affine, not expert truth'}
                real_rows.append(row)
                real_stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'event': 'weak_real_completed', 'arm': arm,
                              'family': family, 'sections': len(records)}), flush=True)

assert len(synthetic_rows) == 3 * sum(len(v) for v in panel_records.values())
assert len(map_rows) == 3 * 64 * 2
assert len(real_rows) == 3 * (64 + 158)
for row in synthetic_rows:
    if row['arm'] == '0':
        continue
    base = next(r for r in synthetic_rows if r['arm'] == '0' and
                r['cohort'] == row['cohort'] and r['section_id'] == row['section_id'])
    assert row['panel_file_sha256'] == base['panel_file_sha256']

summary = {'synthetic': {}, 'map': {}, 'weak_real': {},
           'scope': 'development only; truth-best beam and supplied planes are diagnostics, not deployable inference',
           'calibrated': False, 'expert_real_truth_used': False,
           'public_benchmark_used': False, 'final_animals_used': False}
for arm in arms:
    for cohort in panels:
        group = [r for r in synthetic_rows if r['arm'] == arm and r['cohort'] == cohort]
        plans = sorted({r['synthetic_subject_plan_id'] for r in group})
        item = {'sections': len(group), 'plans': len(plans),
                'top1_plan_equal_mm': float(np.mean([np.mean([r['top1_rigid_mm'] for r in group
                    if r['synthetic_subject_plan_id'] == plan]) for plan in plans])),
                'top16_near_plan_equal': float(np.mean([np.mean([
                    r['top16_truth_best_rigid_mm'] <= 1.5 for r in group
                    if r['synthetic_subject_plan_id'] == plan]) for plan in plans])),
                'by_plan': {}, 'by_mode': {}, 'by_angle': {}, 'by_tissue_fraction': {}}
        for plan in plans:
            part = [r for r in group if r['synthetic_subject_plan_id'] == plan]
            item['by_plan'][plan] = {'sections': len(part),
                'top1_mean_mm': float(np.mean([r['top1_rigid_mm'] for r in part])),
                'top16_near': float(np.mean([r['top16_truth_best_rigid_mm'] <= 1.5
                                           for r in part]))}
        for name, field, bins in (
            ('by_mode', 'appearance_mode', ('raw', 'exact_black', 'imperfect_brush')),
            ('by_angle', 'nearest_cardinal_angle_deg', ((0, 15), (15, 30), (30, 45), (45, 55))),
            ('by_tissue_fraction', 'valid_fraction', ((0, .15), (.15, .30), (.30, .60), (.60, 1.01)))):
            for bin_value in bins:
                part = [r for r in group if (r[field] == bin_value if isinstance(bin_value, str)
                    else bin_value[0] <= r[field] < bin_value[1])]
                item[name][str(bin_value)] = {'sections': len(part),
                    'top1_mean_mm': float(np.mean([r['top1_rigid_mm'] for r in part])) if part else None,
                    'top16_near': float(np.mean([r['top16_truth_best_rigid_mm'] <= 1.5
                                                for r in part])) if part else None}
        summary['synthetic'][f'{arm}/{cohort}'] = item
    for role in ('exact', 'baseline_near'):
        part = [r for r in map_rows if r['arm'] == arm and r['role'] == role and
                (role == 'exact' or r['baseline_near_rigid_mm'] <= 1.5)]
        plans = sorted({r['synthetic_subject_plan_id'] for r in part})
        summary['map'][f'{arm}/{role}'] = {'sections': len(part), 'plans': len(plans),
            'plan_equal_valid_mean_mm': float(np.mean([np.mean([r['mapped_valid_mean_mm']
                for r in part if r['synthetic_subject_plan_id'] == plan]) for plan in plans])) if plans else None}
    for family in ('coronal', 'sagittal'):
        part = [r for r in real_rows if r['arm'] == arm and r['family'] == family]
        donors = sorted({r['donor_id'] for r in part})
        by_donor = {str(donor): float(np.mean([r['top1_weak_five_point_mm'] for r in part
                     if r['donor_id'] == donor])) for donor in donors}
        summary['weak_real'][f'{arm}/{family}'] = {'sections': len(part), 'donors': len(donors),
            'donor_equal_top1_mm': float(np.mean(list(by_donor.values()))),
            'donor_equal_top16_truth_best_mm': float(np.mean([np.mean([
                r['top16_truth_best_weak_five_point_mm'] for r in part if r['donor_id'] == donor])
                for donor in donors])),
            'donor_equal_top1_normal_deg': float(np.mean([np.mean([
                r['top1_weak_normal_deg'] for r in part if r['donor_id'] == donor])
                for donor in donors])), 'by_donor_top1_mm': by_donor}

baseline = summary['synthetic']['0/v4']
final = summary['synthetic']['653/v4']
conditions = {'v4_plan_equal_top1_gain_ge_0p25_mm':
              baseline['top1_plan_equal_mm'] - final['top1_plan_equal_mm'] >= .25,
              'v4_plan_equal_top16_near_drop_le_0p05':
              final['top16_near_plan_equal'] >= baseline['top16_near_plan_equal'] - .05,
              'both_weak_real_families_regression_le_0p20_mm': all(
                  summary['weak_real'][f'653/{family}']['donor_equal_top1_mm'] <=
                  summary['weak_real'][f'0/{family}']['donor_equal_top1_mm'] + .20
                  for family in ('coronal', 'sagittal')),
              'each_weak_real_donor_regression_le_0p50_mm': all(
                  summary['weak_real'][f'653/{family}']['by_donor_top1_mm'][donor] <= value + .50
                  for family in ('coronal', 'sagittal')
                  for donor, value in summary['weak_real'][f'0/{family}']['by_donor_top1_mm'].items()),
              'supplied_exact_and_near_map_regression_le_0p10_mm': all(
                  summary['map'][f'653/{role}']['plan_equal_valid_mean_mm'] <=
                  summary['map'][f'0/{role}']['plan_equal_valid_mean_mm'] + .10
                  for role in ('exact', 'baseline_near'))}
summary['development_gate'] = {'conditions': conditions, 'passed': all(conditions.values()),
    'note': 'A pass only supports continued model development; not expert truth, calibration, GUI, or benchmark'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'synthetic_rows': len(synthetic_rows), 'map_rows': len(map_rows), 'real_weak_rows': len(real_rows),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'synthetic_rows.jsonl', 'map_rows.jsonl', 'real_weak_rows.jsonl', 'summary.json')},
    'evaluator_sha256': sha(__file__), 'training_completion_sha256': sha(train / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'development_gate': summary['development_gate']}), flush=True)
