"""Frozen DEV proposal and weak-affine readout for the 138 anchor-residual experiment."""

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
from training.shared_anchor_pose_138 import AnchorResidual138, corrected_states


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SHARED_ANCHOR_POSE_138_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
all160_parent_eval = root / 'runs/structural_input_137_dev_eval'
train = root / 'runs/shared_anchor_pose_138'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/shared_anchor_pose_138_dev_eval'
controls = ('untied_visual', 'shared_source_zero')
arms = ('parent_132', 'shared_visual', *controls)
metrics = ('all160_best', 'beam_best', 'top1')
thresholds = (.5, 1., 1.5)
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
frozen_done = json.loads((parent_eval / 'completed.json').read_text())
frozen_synthetic = [json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open()]
frozen_real = [json.loads(line) for line in (parent_eval / 'real_weak_rows.jsonl').open()]
frozen_syn = {(row['cohort'], row['section_id']): row for row in frozen_synthetic
              if row['arm'] == '2000'}
frozen_selected = {(row['family'], row['section_id']): row for row in frozen_real
                   if row['arm'] == '2000'}
all160_done = json.loads((all160_parent_eval / 'completed.json').read_text())
all160_config = json.loads((all160_parent_eval / 'config.json').read_text())
assert all(sha(all160_parent_eval / name) == digest
           for name, digest in all160_done['output_sha256'].items())
all160_parent = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (all160_parent_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == 'control_132_2000'}
assert len(all160_parent) == 248 + 243
assert sum(row['all160_best_rigid_mm'] <= .5 for (cohort, _), row in
           all160_parent.items() if cohort == 'v4') == 35
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(parent_eval / name) == digest
           for name, digest in frozen_done['output_sha256'].items())
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256']
assert train_config['parent_checkpoint_sha256'] == train_done['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['protocol_sha256'] == train_done['protocol_sha256'] == sha(protocol)
assert train_config['source_sha256'] == train_done['source_sha256']
assert all160_config['checkpoint_sha256']['control_132_2000'] == sha(parent_path)
assert all160_config['control_evaluation_completion_sha256'] == sha(parent_eval / 'completed.json')
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(train_done.get(key, False) or parent_done.get(key, False) for key in
    ('calibrated', 'expert_real_truth_used', 'final_animals_used',
     'public_benchmark_used', 'external_pretrained_weights_used'))
checkpoints = {arm: {str(step): train / arm / f'head_step_{step:05d}.pt'
                     for step in (0, 4000)} for arm in arms[1:]}
assert all(sha(path) == train_done['checkpoint_sha256'][arm][step]
           for arm, paths in checkpoints.items() for step, path in paths.items())

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
assert panel_done['v4']['eligible'] == 248 and panel_done['v3']['eligible'] == 243
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


def beam_for(prediction, states):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(states)[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam[0].tolist()


def all160_costs(states, observed_chart, target):
    branch_cost = []
    for start in range(0, 160, 16):
        ids = torch.arange(start, start + 16, device='cuda')
        proposed = rigid_points_090(states[:, ids // 2], (ids % 2)[None], observed_chart)[0]
        branch_cost.append((proposed - target).norm(dim=-1).mean(-1) / 1000)
    return torch.cat(branch_cost)


def synthetic_metrics(states, beam, top_id, observed_chart, target, target_five):
    branch_cost = all160_costs(states, observed_chart, target)
    all_id = int(branch_cost.argmin())
    beam_id = beam[int(branch_cost[beam].argmin())]
    result = {'beam_branch_ids': beam, 'all160_best_branch_id': all_id,
              'beam_best_branch_id': beam_id, 'top1_branch_id': top_id}
    for name, branch in (('all160_best', all_id), ('beam_best', beam_id), ('top1', top_id)):
        chosen = torch.tensor([branch], device='cuda')
        observed = rigid_points_090(states[:, chosen // 2], (chosen % 2)[None], observed_chart)[0, 0]
        five = rigid_points_090(states[:, chosen // 2], (chosen % 2)[None], five_chart)[0, 0]
        result[f'{name}_rigid_mm'] = float(branch_cost[branch])
        result[f'{name}_p90_mm'] = float(torch.quantile((observed - target).norm(dim=-1) / 1000, .9))
        result[f'{name}_five_point_mm'] = float((five - target_five).norm(dim=-1).mean() / 1000)
    return result


model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
heads = {}
zero_heads = {}
for arm in arms[1:]:
    initial = torch.load(checkpoints[arm]['0'], map_location='cpu', weights_only=True)
    assert initial['step'] == 0 and initial['arm'] == arm and initial['config'] == train_config
    zero_head = AnchorResidual138(arm).cuda().eval().requires_grad_(False)
    zero_head.load_state_dict(initial['head'], strict=True)
    zero_heads[arm] = zero_head
    del initial
    saved = torch.load(checkpoints[arm]['4000'], map_location='cpu', weights_only=True)
    assert saved['step'] == 4000 and saved['arm'] == arm and saved['config'] == train_config
    head = AnchorResidual138(arm).cuda().eval().requires_grad_(False)
    head.load_state_dict(saved['head'], strict=True)
    heads[arm] = head
    del saved

v4_swap = {}
eligible_v4 = [row for row in panel_records['v4'] if row['eligible']]
for plan in {row['synthetic_subject_plan_id'] for row in eligible_v4}:
    plan_rows = sorted((row for row in eligible_v4
        if row['synthetic_subject_plan_id'] == plan), key=lambda row: str(row['section_id']))
    for mode in {row['appearance_mode'] for row in plan_rows}:
        group = [row for row in plan_rows if row['appearance_mode'] == mode]
        if len(group) > 1:
            for index, row in enumerate(group):
                v4_swap[row['section_id']] = group[(index + 1) % len(group)]
        else:
            row = group[0]
            v4_swap[row['section_id']] = plan_rows[(plan_rows.index(row) + 1) % len(plan_rows)]
assert len(v4_swap) == 248 and all(key != donor['section_id']
    for key, donor in v4_swap.items())

synthetic_rows, real_rows, swap_rows = [], [], []
zero_checked = False
with torch.inference_mode():
    for cohort, panel in panels.items():
        for record in (row for row in panel_records[cohort] if row['eligible']):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            assert int(valid.sum()) == record['valid_pixels']
            observed_chart = chart[valid]
            target = rigid_points_090(truth, reflection, observed_chart)[0]
            target_five = rigid_points_090(truth, reflection, five_chart)[0]
            prediction = model.predict(image)
            if not zero_checked:
                assert all(torch.allclose(corrected_states(prediction, head, model),
                    prediction['state'], rtol=0, atol=1e-5) for head in zero_heads.values())
                zero_heads.clear()
                zero_checked = True
            states_by_arm = {'parent_132': prediction['state'], **{
                arm: corrected_states(prediction, head, model) for arm, head in heads.items()}}
            assert all(torch.equal(states[:, :16], prediction['state'][:, :16])
                       for states in states_by_arm.values())
            tissue = image[0, 0].flatten()[valid]
            exposure = (record['provenance']['one_shot_slide_artifacts_v4']['exposure']
                        if cohort == 'v4' else 1.)
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'], 'panel_file_sha256': record['sha256'],
                'appearance_mode': record['appearance_mode'],
                'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                    np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
                'applied_exposure_multiplier': exposure,
                'observed_valid_tissue_mean': float(tissue.mean()),
                'valid_pixels': int(valid.sum()), 'valid_fraction': int(valid.sum()) / side ** 2}
            for arm, states in states_by_arm.items():
                prior, beam = beam_for(prediction, states)
                top_id = int(prior[0].argmax())
                assert top_id in beam
                row = {'arm': arm, **common, **synthetic_metrics(
                    states, beam, top_id, observed_chart, target, target_five)}
                if arm == 'parent_132':
                    frozen = frozen_syn[(cohort, record['section_id'])]
                    assert beam == frozen['beam_branch_ids'] and top_id == frozen['top1_branch_id']
                    assert abs(row['beam_best_rigid_mm'] - frozen['beam_best_rigid_mm']) < 1e-4
                    assert abs(row['top1_rigid_mm'] - frozen['top1_rigid_mm']) < 1e-4
                    assert abs(row['all160_best_rigid_mm'] - all160_parent[
                        (cohort, record['section_id'])]['all160_best_rigid_mm']) < 1e-4
                if arm == 'shared_visual':
                    treatment_row = row
                synthetic_rows.append(row)
            if cohort == 'v4':
                donor_record = v4_swap[record['section_id']]
                with np.load(panel / donor_record['file'], allow_pickle=False) as arrays:
                    donor_image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                donor_prediction = model.predict(donor_image)
                swapped = {**prediction, 'feature': donor_prediction['feature']}
                swapped_states = corrected_states(swapped, heads['shared_visual'], model)
                swap_rows.append({'section_id': record['section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'appearance_mode': record['appearance_mode'],
                    'donor_section_id': donor_record['section_id'],
                    'donor_appearance_mode': donor_record['appearance_mode'],
                    'donor_file_sha256': donor_record['sha256'],
                    'intact_all160_best_rigid_mm': treatment_row['all160_best_rigid_mm'],
                    'swapped_all160_best_rigid_mm': float(all160_costs(
                        swapped_states, observed_chart, target).min())})
        print(json.dumps({'event': 'synthetic_completed', 'cohort': cohort,
                          'eligible': panel_done[cohort]['eligible']}), flush=True)

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
            prior, _ = beam_for(prediction, prediction['state'])
            top_id = int(prior[0].argmax())
            states_by_arm = {'parent_132': prediction['state'], **{
                arm: corrected_states(prediction, head, model) for arm, head in heads.items()}}
            for arm, states in states_by_arm.items():
                costs = []
                for start in range(0, 160, 16):
                    ids = torch.arange(start, start + 16, device='cuda')
                    proposed = rigid_points_090(states[:, ids // 2], (ids % 2)[None], five_chart)[0]
                    costs.append((proposed - reference).norm(dim=-1).mean(-1) / 1000)
                costs = torch.cat(costs)
                row = {'arm': arm, 'family': family, 'donor_id': donor,
                    'section_id': record['section_id'], 'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'], 'best160_branch_id': int(costs.argmin()),
                    'best160_weak_five_point_mm': float(costs.min()),
                    'top1_branch_id': top_id, 'direct_top1_weak_five_point_mm': float(costs[top_id]),
                    'label_role': 'inherited weak Allen affine, not expert truth'}
                if arm == 'parent_132':
                    frozen = frozen_selected[(family, record['section_id'])]
                    assert top_id == frozen['top1_branch_id']
                    assert abs(row['direct_top1_weak_five_point_mm']
                               - frozen['direct_top1_weak_five_point_mm']) < 1e-4
                real_rows.append(row)
        print(json.dumps({'event': 'weak_real_completed', 'family': family,
                          'sections': len(records)}), flush=True)

assert len(synthetic_rows) == len(arms) * sum(done['eligible'] for done in panel_done.values())
assert len(real_rows) == len(arms) * (len(coronal_records) + len(sagittal_records))
assert len(swap_rows) == 248
assert sum(row['all160_best_rigid_mm'] <= .5 for row in synthetic_rows
           if row['cohort'] == 'v4' and row['arm'] == 'parent_132') == 35


def summarise(rows):
    fields = [f'{metric}_{suffix}_mm' for metric in metrics
              for suffix in ('rigid', 'p90', 'five_point')]
    return {'sections': len(rows),
        'mean_mm': {field: float(np.mean([row[field] for row in rows])) if rows else None
                    for field in fields},
        'capture_le_mm': {str(threshold): {metric: float(np.mean([
            row[f'{metric}_rigid_mm'] <= threshold for row in rows])) if rows else None
            for metric in metrics} for threshold in thresholds}}


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
        by_support = {f'[{low},{high})': summarise([row for row in rows
            if low <= row['valid_fraction'] < high])
            for low, high in ((0, .25), (.25, .50), (.50, .75), (.75, 1.01))}
        overall = summarise(rows)
        synthetic[cohort][arm] = {'section_weighted': overall,
            'plan_equal_capture_le_mm': {str(threshold): {metric: float(np.mean([
                part['capture_le_mm'][str(threshold)][metric] for part in by_plan.values()]))
                for metric in metrics} for threshold in thresholds},
            'by_plan': by_plan, 'by_mode': by_mode,
            'by_angle_deg': by_angle, 'by_exposure': by_exposure,
            'by_valid_support_fraction': by_support}

conversions = {}
for cohort in panels:
    parent_rows = {row['section_id']: row for row in synthetic_rows
                   if row['cohort'] == cohort and row['arm'] == 'parent_132'}
    conversions[cohort] = {}
    for arm in arms[1:]:
        rows = {row['section_id']: row for row in synthetic_rows
                if row['cohort'] == cohort and row['arm'] == arm}
        assert rows.keys() == parent_rows.keys()
        conversions[cohort][arm] = {}
        for metric in ('all160_best', 'beam_best'):
            field = f'{metric}_rigid_mm'
            at_risk = [key for key, row in parent_rows.items() if .5 < row[field] <= 1.5]
            converted = sum(rows[key][field] <= .5 for key in at_risk)
            conversions[cohort][arm][metric] = {'parent_0p5_to_1p5_count': len(at_risk),
                'converted_to_le_0p5_count': converted,
                'conditional_fraction': converted / len(at_risk) if at_risk else None}

real = {}
for family in ('coronal', 'sagittal'):
    donors = sorted({row['donor_id'] for row in real_rows if row['family'] == family})
    real[family] = {'donors': len(donors), 'by_donor': {}, 'donor_equal_mean_mm': {}}
    for donor in donors:
        entries = [row for row in real_rows if row['family'] == family and row['donor_id'] == donor]
        old = [row['direct_top1_weak_five_point_mm'] for row in frozen_real
               if row['arm'] == 'parent' and row['family'] == family and row['donor_id'] == donor]
        real[family]['by_donor'][str(donor)] = {'sections': len(old),
            'frozen_128_selected_mean_mm': float(np.mean(old)),
            'best160_mean_mm': {arm: float(np.mean([row['best160_weak_five_point_mm']
                for row in entries if row['arm'] == arm])) for arm in arms},
            'selected_mean_mm': {arm: float(np.mean([row['direct_top1_weak_five_point_mm']
                for row in entries if row['arm'] == arm])) for arm in arms}}
    real[family]['donor_equal_mean_mm'] = {metric: {arm: float(np.mean([
        donor[f'{metric}_mean_mm'][arm] for donor in real[family]['by_donor'].values()]))
        for arm in arms} for metric in ('best160', 'selected')}

v4 = synthetic['v4']
v3 = synthetic['v3']
primary = lambda cohort, arm: synthetic[cohort][arm]['section_weighted']['capture_le_mm']['0.5']['all160_best']
by_plan_parent = v4['parent_132']['by_plan']
by_plan_treatment = v4['shared_visual']['by_plan']
plan_improvements = sum(
    by_plan_treatment[plan]['capture_le_mm']['0.5']['all160_best']
    > by_plan_parent[plan]['capture_le_mm']['0.5']['all160_best']
    for plan in by_plan_parent)
conditions = {
    'v4_all160_0p5_gain_vs_132_ge_0p15': primary('v4', 'shared_visual') - primary('v4', 'parent_132') >= .15,
    'v4_all160_0p5_gain_vs_each_control_ge_0p10': all(
        primary('v4', 'shared_visual') - primary('v4', arm) >= .10 for arm in controls),
    'v4_all160_0p5_improves_ge_6_of_8_plans': plan_improvements >= 6,
    'v3_all160_0p5_drop_vs_132_le_0p05': primary('v3', 'shared_visual') >= primary('v3', 'parent_132') - .05,
    'v3_all160_1p5_drop_vs_132_le_0p05': (
        v3['shared_visual']['section_weighted']['capture_le_mm']['1.5']['all160_best']
        >= v3['parent_132']['section_weighted']['capture_le_mm']['1.5']['all160_best'] - .05),
    'every_weak_real_donor_best160_vs_132_regression_le_0p20_mm': all(
        donor['best160_mean_mm']['shared_visual'] <= donor['best160_mean_mm']['parent_132'] + .20
        for family in real.values() for donor in family['by_donor'].values())}
selected_guard = {}
for arm in ('parent_132', 'shared_visual'):
    failures = [{'family': family_name, 'donor_id': donor_id,
        'regression_mm': donor['selected_mean_mm'][arm] - donor['frozen_128_selected_mean_mm']}
        for family_name, family in real.items() for donor_id, donor in family['by_donor'].items()
        if donor['selected_mean_mm'][arm] > donor['frozen_128_selected_mean_mm'] + .20]
    selected_guard[arm] = {'passed': not failures, 'failing_donors': failures}
assert selected_guard['parent_132']['passed'] is False
swap_capture = {name: float(np.mean([row[f'{name}_all160_best_rigid_mm'] <= .5
    for row in swap_rows])) for name in ('intact', 'swapped')}
swap_summary = {'role': 'exploratory conditional/OOD feature intervention; not a gate or source-causal estimate',
    'pairing': 'non-self cyclic section-ID permutation within synthetic DEV plan and appearance mode when possible, otherwise within plan; recipient parent state and score prior fixed',
    'sections': len(swap_rows), 'all160_capture_le_0p5': swap_capture,
    'capture_difference_swapped_minus_intact': swap_capture['swapped'] - swap_capture['intact'],
    'by_plan': {plan: {name: float(np.mean([row[f'{name}_all160_best_rigid_mm'] <= .5
        for row in swap_rows if row['subject_plan_id'] == plan]))
        for name in ('intact', 'swapped')}
        for plan in sorted({row['subject_plan_id'] for row in swap_rows})}}
summary = {'scope': 'frozen 132 proposer, matched terminal 138 residual heads, independent v4/v3 synthetic DEV sections and donor-disjoint weak-affine DEV only',
    'metric': 'mean and P90 rigid-gauge displacement over observed valid 256-grid tissue pixels; five-point error on the same mean-winning branch, all in mm',
    'oracle_role': 'truth-selected all160 and blind16 are proposal availability diagnostics, never inference or checkpoint selection',
    'blind16': 'frozen score prior: top8 base, top6 anchor, two normal-diversity additions using each corrected state',
    'conditional_conversion_definition': 'parent 0.5 < error <= 1.5 mm, now <= 0.5 mm',
    'synthetic': synthetic, 'conditional_1p5_to_0p5': conversions,
    'source_feature_swap': swap_summary, 'real_weak_affine': real,
    'endpoint': {'arm': 'shared_visual', 'matched_controls': controls,
                 'v4_plan_improvement_count': plan_improvements,
                 'conditions': conditions, 'mechanistic_proposal_gate_passed': all(conditions.values()),
                 'selected_pose_guard_vs_frozen_128': selected_guard,
                 'promotion_permitted': False, 'development_only': True},
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
config = {'evaluator_source_sha256': sha(__file__),
    'residual_source_sha256': sha(source / 'shared_anchor_pose_138.py'),
    'protocol_sha256': sha(protocol), 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_evaluation_completion_sha256': sha(parent_eval / 'completed.json'),
    'all160_parent_evaluation_completion_sha256': sha(all160_parent_eval / 'completed.json'),
    'training_completion_sha256': sha(train / 'completed.json'),
    'training_config_sha256': sha(train / 'config.json'),
    'checkpoint_sha256': {arm: {step: sha(path) for step, path in paths.items()}
                          for arm, paths in checkpoints.items()},
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
                   ('real_weak_rows.jsonl', real_rows),
                   ('source_feature_swap_rows.jsonl', swap_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'eligible_v4_sections': panel_done['v4']['eligible'],
    'eligible_v3_sections': panel_done['v3']['eligible'],
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'synthetic_rows.jsonl', 'real_weak_rows.jsonl',
         'source_feature_swap_rows.jsonl')},
    'evaluator_source_sha256': config['evaluator_source_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'endpoint': summary['endpoint']},
                 allow_nan=False), flush=True)
