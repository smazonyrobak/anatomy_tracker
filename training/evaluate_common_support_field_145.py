"""Independent frozen-DEV common-support ranking and physical-mapping gate."""

import hashlib
import json
import os
import sys
from collections import defaultdict
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
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import dense_local_targets_143, render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_state(a, b) for a, b in zip(left, right))
    return left == right


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COMMON_SUPPORT_FIELD_145_PROTOCOL_20261010.md'
run = root / 'runs/common_support_field_145'
out = root / 'runs/common_support_field_145_dev_eval'
parent132 = root / 'runs/v4_pose_adaptation_132'
parent143 = root / 'runs/coherent_anatomy_field_143'
eval132 = root / 'runs/v4_pose_adaptation_132_dev_eval'
eval143 = root / 'runs/coherent_anatomy_field_143_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms, steps, device, radius = ('treatment', 'control'), (0, 500, 2000), 'cuda', 6
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == protocol.drive.upper() == run.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()

done132 = json.loads((parent132 / 'completed.json').read_text())
config132 = json.loads((parent132 / 'config.json').read_text())
done143 = json.loads((parent143 / 'completed.json').read_text())
config143 = json.loads((parent143 / 'config.json').read_text())
done132eval = json.loads((eval132 / 'completed.json').read_text())
done143eval = json.loads((eval143 / 'completed.json').read_text())
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
path132 = parent132 / 'joint_step_02000.pt'
path143 = parent143 / 'full_step_10000.pt'
assert sha(parent132 / 'config.json') == done132['config_sha256']
assert sha(parent143 / 'config.json') == done143['config_sha256']
assert sha(path132) == done132['checkpoint_sha256']['2000'] == config143['parent_checkpoint_sha256']
assert sha(path143) == done143['checkpoint_sha256']['full']['10000'] == config['parent_checkpoint_sha256']
assert sha(parent132 / 'completed.json') == config143['parent_completion_sha256']
assert sha(parent143 / 'completed.json') == config['parent_completion_sha256']
assert sha(path132) == config['pose_parent_checkpoint_sha256']
assert sha(parent143 / 'draws.jsonl') == done143['draws_sha256']
assert sha(parent143 / 'training.jsonl') == done143['training_sha256']
assert sha(source.parent / 'docs/publication/COHERENT_ANATOMY_FIELD_143_PROTOCOL_20261010.md') == config143['protocol_sha256']
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(protocol) == config['protocol_sha256']
assert config['source_sha256'] == done['source_sha256']
assert all(sha(source / name) == digest for receipt in (config132, config143, config)
           for name, digest in receipt['source_sha256'].items())
assert done['steps'] == config['steps'] == 2000
assert done['accepted_synthetic_physical_sections'] >= 6000
assert done['v4_presentations'] == 4000 and done['v3_presentations'] == 2000
assert config['synthetic_per_step'] == {'v4': 2, 'v3': 1} and config['presentations'] == 6000
assert config['radius'] == radius and tuple(config['arms']) == arms
assert config['checkpoints'] == list(steps)
assert all(sha(run / f'{arm}_step_{step:05d}.pt') ==
           done['checkpoint_sha256'][arm][str(step)] for arm in arms for step in steps)
assert all(sha(eval132 / name) == digest for name, digest in done132eval['output_sha256'].items())
assert sha(source / 'evaluate_v4_pose_adaptation_132.py') == done132eval['evaluator_source_sha256']
assert all(sha(eval143 / name) == digest for name, digest in
           {'config.json': done143eval['config_sha256'],
            'rows.jsonl': done143eval['rows_sha256'],
            'summary.json': done143eval['summary_sha256']}.items())
assert sha(source / 'evaluate_coherent_anatomy_field_143.py') == done143eval['evaluator_sha256']
assert sha(parent143 / 'completed.json') == done143eval['train_completion_sha256']
assert not any(receipt.get(key, False) for receipt in
    (done132, done143, done132eval, done143eval, done) for key in
    ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
     'final_animals_used', 'external_pretrained_weights_used'))

rows132 = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (eval132 / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
rows143 = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (eval143 / 'rows.jsonl').open())
    if row['step'] == 10000 and row['arm'] == 'full' and row['role'] == 'blind_near'}
assert len(rows132) == len(rows143) == 491
records = {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == receipt['protocol_sha256']
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(all_records) == receipt['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
    assert all(row['provenance']['split'] == 'development' for row in all_records)
    records[cohort] = [row for row in all_records if row['eligible']]
    assert len(records[cohort]) == receipt['eligible'] == (248 if cohort == 'v4' else 243)
    assert all((cohort, row['section_id']) in rows132 and
               (cohort, row['section_id']) in rows143 for row in records[cohort])
train_plans = {row['plan_receipt'] for row in config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for cohort in records for row in records[cohort]}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans
assert all(row['lineage']['split'] == 'train'
           for row in config['synthetic_provenance']['base_subjects'])
assert {row['panel_physical_section_id'] for row in records['v4']}.isdisjoint(
       {row['panel_physical_section_id'] for row in records['v3']})

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved = torch.load(path132, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == config132
pose.load_state_dict(saved['model'], strict=True)
del saved
baseline = torch.load(path143, map_location='cpu', weights_only=True)
assert baseline['step'] == 10000 and baseline['arm'] == 'full' and baseline['config'] == config143
fields = {}
for step in steps:
    for arm in arms:
        saved = torch.load(run / f'{arm}_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == step and saved['arm'] == arm and saved['config'] == config
        assert {'field', 'optimizer', 'scaler', 'numpy_rng', 'torch_rng', 'cuda_rng'} <= saved.keys()
        if step == 0:
            assert all(same_state(saved[key], baseline[key]) for key in ('field', 'optimizer', 'scaler'))
        field = CoherentAnatomyField143(radius=radius).to(device).eval().requires_grad_(False)
        field.load_state_dict(saved['field'], strict=True)
        fields[step, arm] = field
        del saved
del baseline

axis = (torch.arange(64, device=device) + .5) / 64
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
axis256 = torch.arange(256, device=device) / 256
yy256, xx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((xx256, yy256), -1)
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device=device),
                                  torch.arange(64, device=device), indexing='ij')
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device=device)


def wrong_bank(state, reflection, valid, exact, offsets, weights, section_id):
    seed = int(hashlib.sha256(section_id.encode()).hexdigest()[:15], 16)
    generator = torch.Generator().manual_seed(seed)
    updates = ((2 * torch.rand((128, 9), generator=generator) - 1).to(device) * limits)
    updates[:, :3] *= 3
    updates[:, 3:6] *= 3
    updates[:, 6:] *= 2
    candidates = compose_full_frame_state(state.expand(128, -1), updates)
    observed = chart[valid]
    true_rigid = rigid_points_090(state, reflection, observed)[0]
    errors = (rigid_points_090(candidates, reflection.expand(128), observed) -
              true_rigid).norm(dim=-1).mean(-1) / 1000
    possible = ((errors >= 1.) & (errors <= 3.)).nonzero().flatten()
    exact_support = exact[0, 1, 3] > .5
    best_iou, best_index, best_common, best_wrong = -1., None, None, None
    for indices in possible.split(16):
        slabs, _, _, _ = render_coherent_atlas_143(atlas, candidates[indices],
            reflection.expand(len(indices)), offsets.expand(len(indices), -1),
            weights.expand(len(indices), -1))
        wrong_support = slabs[:, 1, 3] > .5
        intersection = (wrong_support & exact_support).sum((1, 2))
        union = (wrong_support | exact_support).sum((1, 2)).clamp_min(1)
        ious = intersection.float() / union
        maximum = int(ious.argmax())
        if float(ious[maximum]) > best_iou:
            best_iou = float(ious[maximum])
            best_index = int(indices[maximum])
            best_common = float(intersection[maximum]) / 4096
            best_wrong = slabs[maximum:maximum + 1].clone()
    eligible = best_index is not None and best_iou >= .45 and best_common >= .08
    bank = {'seed': seed, 'candidate_count': 128, 'rigid_1_to_3_mm_candidates': len(possible),
        'chosen_index': best_index, 'wrong_rigid_mm': float(errors[best_index])
            if best_index is not None else None,
        'central_support_iou': best_iou if best_index is not None else None,
        'central_common_support_fraction': best_common, 'rank_eligible': eligible,
        'wrong_state': candidates[best_index].tolist() if best_index is not None else None}
    return bank, best_wrong


def mapping(output, slot, truth, valid, slabs, base, basis, normal):
    valid64 = valid[0, 2::4, 2::4]
    truth64 = truth[0, 2::4, 2::4].float()
    local, _, reachable = dense_local_targets_143(truth, valid,
        base[slot:slot + 1], basis[slot:slot + 1], normal[slot:slot + 1], radius=radius)
    local = local[0]
    x = pixel_x + 64 * local[..., 0]
    y = pixel_y + 64 * local[..., 1]
    z = 3 + local[..., 2] / 500
    inside = (x >= 0) & (x <= 63) & (y >= 0) & (y <= 63) & (z >= 0) & (z <= 6)
    grid = torch.stack((2 * x / 63 - 1, 2 * y / 63 - 1, z / 3 - 1), -1)[None, None]
    support = F.grid_sample(slabs[slot:slot + 1, 1:2].float(), grid,
        align_corners=True)[0, 0, 0] > .5
    global_support = valid64 & inside & support
    estimate = output['offset_chart'][slot].permute(1, 2, 0).float()
    predicted = (base[slot] + torch.einsum('ij,hwj->hwi', basis[slot], estimate[..., :2]) +
                 normal[slot, None, None] * estimate[..., 2:3])
    distance = (predicted - truth64).norm(dim=-1)
    rigid_fail = valid64 & ((base[slot] - truth64).norm(dim=-1) > 500)
    matched = output['match_logits'][slot].argmax(0) != output['match_logits'].shape[1] - 1
    hit = matched & (distance <= 500)
    displacement = predicted - base[slot]
    base_h = (base[slot, :, 1:] - base[slot, :, :-1]).norm(dim=-1).clamp_min(1)
    base_v = (base[slot, 1:] - base[slot, :-1]).norm(dim=-1).clamp_min(1)
    strain_h = ((displacement[:, 1:] - displacement[:, :-1]).norm(dim=-1) /
                base_h)[valid64[:, 1:] & valid64[:, :-1]]
    strain_v = ((displacement[1:] - displacement[:-1]).norm(dim=-1) /
                base_v)[valid64[1:] & valid64[:-1]]
    strain = torch.cat((strain_h, strain_v))
    return {'valid_sites': int(valid64.sum()),
        'global_supported_sites': int(global_support.sum()),
        'locally_reachable_sites': int((global_support & reachable[0]).sum()),
        'rigid_null_failing_valid_sites': int(rigid_fail.sum()),
        'rigid_null_failing_global_sites': int((rigid_fail & global_support).sum()),
        'scored_le_500_um_all_valid': int((hit & valid64).sum()),
        'scored_le_500_um_global_supported': int((hit & global_support).sum()),
        'scored_le_500_um_rigid_null_failing_valid': int((hit & rigid_fail).sum()),
        'scored_le_500_um_rigid_null_failing_global': int((hit & rigid_fail & global_support).sum()),
        'argmax_dustbin_valid': int((valid64 & ~matched).sum()),
        'visible_gt_0p5_valid': int((valid64 & (output['visibility'][slot, 0] > .5)).sum()),
        'point_error_mm_valid_mean': float(distance[valid64].mean() / 1000),
        'point_error_mm_valid_median': float(distance[valid64].median() / 1000),
        'warp_rms_mm_valid': float(displacement[valid64].square().sum(-1).mean().sqrt() / 1000),
        'warp_relative_neighbor_strain_p95': float(torch.quantile(strain, .95)) if len(strain) else None}


def summarize(group):
    counts = {key: sum(row[key] for row in group) for key in (
        'valid_sites', 'global_supported_sites', 'locally_reachable_sites',
        'rigid_null_failing_valid_sites', 'rigid_null_failing_global_sites',
        'scored_le_500_um_all_valid', 'scored_le_500_um_global_supported',
        'scored_le_500_um_rigid_null_failing_valid',
        'scored_le_500_um_rigid_null_failing_global', 'argmax_dustbin_valid',
        'visible_gt_0p5_valid')}
    def percent(numerator, denominator):
        return 100 * counts[numerator] / counts[denominator] if counts[denominator] else None
    plans = defaultdict(list)
    for row in group:
        plans[row['subject_plan_id']].append(row)
    strains = [row['warp_relative_neighbor_strain_p95'] for row in group
               if row['warp_relative_neighbor_strain_p95'] is not None]
    return {'sections': len(group), 'plans': len(plans), **counts,
        'global_support_coverage_percent': percent('global_supported_sites', 'valid_sites'),
        'scored_over_all_valid_percent': percent('scored_le_500_um_all_valid', 'valid_sites'),
        'scored_over_global_percent': percent('scored_le_500_um_global_supported',
                                             'global_supported_sites'),
        'scored_over_rigid_null_failing_valid_percent': percent(
            'scored_le_500_um_rigid_null_failing_valid', 'rigid_null_failing_valid_sites'),
        'scored_over_rigid_null_failing_global_percent': percent(
            'scored_le_500_um_rigid_null_failing_global', 'rigid_null_failing_global_sites'),
        'non_dustbin_coverage_percent': 100 - percent('argmax_dustbin_valid', 'valid_sites')
            if counts['valid_sites'] else None,
        'point_error_mm_plan_equal': float(np.mean([np.mean([
            row['point_error_mm_valid_mean'] for row in items]) for items in plans.values()]))
            if plans else None,
        'warp_rms_mm_plan_equal': float(np.mean([np.mean([
            row['warp_rms_mm_valid'] for row in items]) for items in plans.values()]))
            if plans else None,
        'warp_strain_p95_section_median': float(np.median(strains)) if strains else None}


def rank_summary(group):
    ranked = [row for row in group if row['rank_eligible']]
    scores = [.5 if row['energy_margin'] == 0 else float(row['energy_margin'] > 0)
              for row in ranked]
    return {'eligible_sections': len(ranked), 'all_sections': len(group),
        'correct_tie_half_percent': 100 * sum(scores) / len(scores) if scores else None,
        'ties': sum(row['energy_margin'] == 0 for row in ranked),
        'median_energy_margin': float(np.median([row['energy_margin'] for row in ranked]))
            if ranked else None}


out.mkdir(parents=True, exist_ok=False)
readout = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'parent132_checkpoint_sha256': sha(path132), 'parent143_checkpoint_sha256': sha(path143),
    'parent132_eval_completion_sha256': sha(eval132 / 'completed.json'),
    'parent143_eval_completion_sha256': sha(eval143 / 'completed.json'),
    'panel_receipt_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'cohorts': {cohort: len(records[cohort]) for cohort in records}, 'arms': arms,
    'checkpoints': steps, 'wrong_bank': '128 section-seeded geometry-only poses; 1-3mm rigid error; maximum central binary-support IoU',
    'rank_retained': 'central common pixelwise support fraction>=0.08 and IoU>=0.45; identical per-depth support and zero exterior intensity',
    'rank': 'lower exact energy wins, tie=0.5; treatment zero-intensity and independently per-depth/pair spatial-permutation ablations',
    'mapping': 'normal full-intensity atlas slabs; non-dustbin continuous CCF point within 500um; all-valid, global-supported, and rigid-null-failing denominators',
    'blind_near': '143 frozen truth-best branch from 132 blind beam, <=1.5mm diagnostic only, never a deployable selector',
    'v3_material_regression': 'treatment minus control < -5 percentage points on exact or <=1.5mm blind-near globally-supported hits',
    'split': '64 train plans versus eight frozen overlapping-use development plans; no fresh biological validation',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(readout, indent=2))

pair_rows, mapping_rows, ranking_rows = [], [], []
with torch.inference_mode(), (out / 'pair_bank.jsonl').open('w') as pair_stream, \
        (out / 'mapping_rows.jsonl').open('w') as mapping_stream, \
        (out / 'ranking_rows.jsonl').open('w') as ranking_stream:
    for cohort, panel in panels.items():
        for index, record in enumerate(records[cohort]):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).to(device).bool()
                truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
            assert int(valid.sum()) == record['valid_pixels']
            prediction = pose.predict(image)
            frozen = rows143[cohort, record['section_id']]
            parent = rows132[cohort, record['section_id']]
            branch = frozen['blind_near_branch_id']
            assert branch in parent['beam_branch_ids']
            observed = chart256[valid[0]]
            target_rigid = rigid_points_090(state, reflection, observed)[0]
            beam_ids = torch.tensor(parent['beam_branch_ids'], device=device)
            beam_world = rigid_points_090(prediction['state'][:, beam_ids // 2],
                (beam_ids % 2)[None], observed)[0]
            beam_errors = (beam_world - target_rigid).norm(dim=-1).mean(-1) / 1000
            assert int(beam_ids[beam_errors.argmin()]) == branch
            near_state = prediction['state'][:, branch // 2]
            near_reflection = reflection.new_full((1,), branch % 2)
            near_mm = float((rigid_points_090(near_state, near_reflection, observed)[0] -
                             target_rigid).norm(dim=-1).mean() / 1000)
            assert abs(near_mm - frozen['blind_near_rigid_mm']) < 1e-4
            assert (near_mm <= 1.5) == frozen['blind_near_available_1p5mm']
            candidate_states = torch.cat((state, near_state))
            candidate_reflections = torch.cat((reflection, near_reflection))
            slabs, base, basis, normal = render_coherent_atlas_143(atlas, candidate_states,
                candidate_reflections, offsets.expand(2, -1), weights.expand(2, -1))
            bank, wrong_slab = wrong_bank(state, reflection, valid[0, 2::4, 2::4],
                slabs[:1], offsets, weights, record['section_id'])
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            angle_bin = '<15' if angle < 15 else '15-30' if angle < 30 else '30-45' if angle < 45 else '>=45'
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'], 'appearance_mode': record['appearance_mode'],
                'nearest_cardinal_angle_deg': angle, 'nearest_cardinal_angle_bin_deg': angle_bin,
                'blind_near_branch_id': branch, 'blind_near_rigid_mm': near_mm,
                'blind_near_available_1p5mm': near_mm <= 1.5,
                'rank_eligible': bank['rank_eligible']}
            pair_row = {**common, **bank}
            pair_stream.write(json.dumps(pair_row, allow_nan=False) + '\n')
            pair_rows.append(pair_row)
            if bank['rank_eligible']:
                shared = ((slabs[:1, 1:2] > .5) & (wrong_slab[:, 1:2] > .5)).float()
                shared = shared.expand(2, -1, -1, -1, -1)
                original = torch.cat((slabs[:1], wrong_slab), 0)
                full_pair = torch.cat((original[:, :1] * shared, shared), 1)
                zero_pair = torch.cat((torch.zeros_like(full_pair[:, :1]), shared), 1)
                scrambled_pair = full_pair.clone()
                scramble_rng = torch.Generator(device=device).manual_seed(bank['seed'] ^ 0x1455C)
                for slot in range(2):
                    for depth in range(7):
                        mask = shared[slot, 0, depth].bool()
                        values = scrambled_pair[slot, 0, depth][mask]
                        permutation = torch.randperm(len(values), device=device, generator=scramble_rng)
                        scrambled_pair[slot, 0, depth][mask] = values[permutation]
                assert torch.equal(full_pair[:, 1], zero_pair[:, 1]) and \
                    torch.equal(full_pair[:, 1], scrambled_pair[:, 1])
            feature = prediction['feature']
            for step in steps:
                for arm in arms:
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = fields[step, arm](image.expand(2, -1, -1, -1), slabs,
                                                  500., source_feature=feature.expand(2, -1, -1, -1))
                    for role, slot in (('exact', 0), ('blind_near', 1)):
                        row = {**common, 'step': step, 'arm': arm, 'role': role,
                            'fit_energy': float(output['energy'][slot]),
                            **mapping(output, slot, truth, valid, slabs, base, basis, normal)}
                        mapping_stream.write(json.dumps(row, allow_nan=False) + '\n')
                        mapping_rows.append(row)
                    if bank['rank_eligible']:
                        pairs = (torch.cat((full_pair, zero_pair, scrambled_pair))
                                 if arm == 'treatment' else full_pair)
                        with torch.autocast('cuda', dtype=torch.float16):
                            rank_output = fields[step, arm](image.expand(len(pairs), -1, -1, -1),
                                pairs, 500., source_feature=feature.expand(len(pairs), -1, -1, -1))
                        energies = rank_output['energy'].float().tolist()
                    else:
                        energies = None
                    conditions = ('full', 'zero_intensity', 'spatial_scramble') if arm == 'treatment' else ('full',)
                    for condition_index, condition in enumerate(conditions):
                        exact_energy = energies[2 * condition_index] if energies is not None else None
                        wrong_energy = energies[2 * condition_index + 1] if energies is not None else None
                        rank_row = {**common, 'step': step, 'arm': arm, 'condition': condition,
                            'exact_energy': exact_energy, 'wrong_energy': wrong_energy,
                            'energy_margin': wrong_energy - exact_energy
                                if energies is not None else None}
                        ranking_stream.write(json.dumps(rank_row, allow_nan=False) + '\n')
                        ranking_rows.append(rank_row)
            if (index + 1) % 64 == 0:
                pair_stream.flush()
                mapping_stream.flush()
                ranking_stream.flush()
                print(json.dumps({'event': 'development_milestone', 'cohort': cohort,
                                  'sections': index + 1}), flush=True)

summary = {'pair_bank': {}, 'mapping': {}, 'ranking': {},
    'warning': 'The truth-best blind-near branch and exact plane are frozen DEV oracle diagnostics, not deployable pose selection. The eight DEV deformation plans have informed earlier design choices.'}
for cohort in panels:
    pairs = [row for row in pair_rows if row['cohort'] == cohort]
    selected = [row for row in pairs if row['chosen_index'] is not None]
    overlaps = [row['central_support_iou'] for row in selected]
    common_fractions = [row['central_common_support_fraction'] for row in selected]
    summary['pair_bank'][cohort] = {'sections': len(pairs),
        'eligible_sections': sum(row['rank_eligible'] for row in pairs),
        'excluded_sections': sum(not row['rank_eligible'] for row in pairs),
        'no_rigid_candidate_sections': sum(row['chosen_index'] is None for row in pairs),
        'below_iou_0p45_sections': sum(row['chosen_index'] is not None and
                                      row['central_support_iou'] < .45 for row in pairs),
        'below_common_fraction_0p08_sections': sum(row['chosen_index'] is not None and
                                                  row['central_common_support_fraction'] < .08 for row in pairs),
        'selected_central_support_iou_quantiles':
            np.quantile(overlaps, [0, .1, .25, .5, .75, .9, 1]).tolist() if overlaps else None,
        'selected_central_common_fraction_quantiles':
            np.quantile(common_fractions, [0, .1, .25, .5, .75, .9, 1]).tolist()
            if common_fractions else None}
    summary['mapping'][cohort], summary['ranking'][cohort] = {}, {}
    appearances = sorted({row['appearance_mode'] for row in pairs})
    for step in steps:
        summary['mapping'][cohort][str(step)] = {}
        summary['ranking'][cohort][str(step)] = {}
        for arm in arms:
            summary['mapping'][cohort][str(step)][arm] = {}
            for role in ('exact', 'blind_near'):
                base_rows = [row for row in mapping_rows if row['cohort'] == cohort and
                             row['step'] == step and row['arm'] == arm and row['role'] == role]
                scopes = {'all': base_rows,
                    'rank_retained': [row for row in base_rows if row['rank_eligible']],
                    'rank_excluded': [row for row in base_rows if not row['rank_eligible']]}
                if role == 'blind_near':
                    scopes.update({'near_le_1p5': [row for row in base_rows if row['blind_near_available_1p5mm']],
                        'rank_retained_near_le_1p5': [row for row in base_rows if row['rank_eligible'] and row['blind_near_available_1p5mm']],
                        'rank_excluded_near_le_1p5': [row for row in base_rows if not row['rank_eligible'] and row['blind_near_available_1p5mm']]})
                summary['mapping'][cohort][str(step)][arm][role] = {
                    scope: {'all': summarize(group),
                            'angle': {angle: summarize([row for row in group
                                if row['nearest_cardinal_angle_bin_deg'] == angle])
                                for angle in ('<15', '15-30', '30-45', '>=45')},
                            'appearance': {appearance: summarize([row for row in group
                                if row['appearance_mode'] == appearance]) for appearance in appearances}}
                    for scope, group in scopes.items()}
            summary['ranking'][cohort][str(step)][arm] = {
                condition: rank_summary([row for row in ranking_rows if row['cohort'] == cohort and
                    row['step'] == step and row['arm'] == arm and row['condition'] == condition])
                for condition in (('full', 'zero_intensity', 'spatial_scramble')
                                  if arm == 'treatment' else ('full',))}

v4rank = summary['ranking']['v4']['2000']
v4map = summary['mapping']['v4']['2000']
v3map = summary['mapping']['v3']['2000']
treat_rank = v4rank['treatment']['full']['correct_tie_half_percent']
control_rank = v4rank['control']['full']['correct_tie_half_percent']
zero_rank = v4rank['treatment']['zero_intensity']['correct_tie_half_percent']
scramble_rank = v4rank['treatment']['spatial_scramble']['correct_tie_half_percent']
rigid_treatment = v4map['treatment']['exact']['rank_retained']['all']
rigid_control = v4map['control']['exact']['rank_retained']['all']
near_treatment = v4map['treatment']['blind_near']['rank_retained_near_le_1p5']['all']
near_control = v4map['control']['blind_near']['rank_retained_near_le_1p5']['all']
v3_exact_treatment = v3map['treatment']['exact']['all']['all']
v3_exact_control = v3map['control']['exact']['all']['all']
v3_near_treatment = v3map['treatment']['blind_near']['near_le_1p5']['all']
v3_near_control = v3map['control']['blind_near']['near_le_1p5']['all']
assert v4rank['treatment']['full']['eligible_sections'] == v4rank['control']['full']['eligible_sections']
assert rigid_treatment['rigid_null_failing_valid_sites'] == rigid_control['rigid_null_failing_valid_sites']
assert near_treatment['global_supported_sites'] == near_control['global_supported_sites']
assert v3_exact_treatment['global_supported_sites'] == v3_exact_control['global_supported_sites']
assert v3_near_treatment['global_supported_sites'] == v3_near_control['global_supported_sites']
eligible = v4rank['treatment']['full']['eligible_sections']
rigid_sites = rigid_treatment['rigid_null_failing_valid_sites']
near_sites = near_treatment['global_supported_sites']
v3_exact_sites = v3_exact_treatment['global_supported_sites']
v3_near_sites = v3_near_treatment['global_supported_sites']
conditions = {
    'v4_eligible_at_least_120': eligible >= 120,
    'rigid_null_fail_sites_at_least_100': rigid_sites >= 100,
    'near_global_sites_nonzero': near_sites > 0,
    'v3_global_sites_nonzero': v3_exact_sites > 0 and v3_near_sites > 0,
    'treatment_rank_at_least_75_percent': treat_rank is not None and treat_rank >= 75,
    'rank_treatment_minus_control_at_least_10pp': treat_rank is not None and
        control_rank is not None and treat_rank - control_rank >= 10,
    'zero_rank_at_most_60_and_full_minus_zero_at_least_10pp': zero_rank is not None and
        treat_rank is not None and zero_rank <= 60 and treat_rank - zero_rank >= 10,
    'scramble_rank_at_most_60_and_full_minus_scramble_at_least_10pp': scramble_rank is not None and
        treat_rank is not None and scramble_rank <= 60 and treat_rank - scramble_rank >= 10,
    'rigid_null_fail_mapping_gain_at_least_10pp': rigid_sites > 0 and
        rigid_treatment['scored_over_rigid_null_failing_valid_percent'] -
        rigid_control['scored_over_rigid_null_failing_valid_percent'] >= 10,
    'blind_near_global_mapping_gain_at_least_5pp': near_sites > 0 and
        near_treatment['scored_over_global_percent'] - near_control['scored_over_global_percent'] >= 5,
    'no_v3_exact_global_regression_over_5pp': v3_exact_sites > 0 and
        v3_exact_treatment['scored_over_global_percent'] -
        v3_exact_control['scored_over_global_percent'] >= -5,
    'no_v3_near_global_regression_over_5pp': v3_near_sites > 0 and
        v3_near_treatment['scored_over_global_percent'] -
        v3_near_control['scored_over_global_percent'] >= -5}
inconclusive = not all(conditions[key] for key in
    ('v4_eligible_at_least_120', 'rigid_null_fail_sites_at_least_100',
     'near_global_sites_nonzero', 'v3_global_sites_nonzero'))
summary['decision'] = {'status': 'inconclusive' if inconclusive else
    ('pass' if all(conditions.values()) else 'fail'), 'pass': all(conditions.values()),
    'conditions': conditions, 'v4_rank_eligible_sections': eligible,
    'v4_rank_all_sections': len(records['v4']),
    'v4_treatment_full_rank_percent': treat_rank,
    'v4_control_full_rank_percent': control_rank,
    'v4_treatment_zero_rank_percent': zero_rank,
    'v4_treatment_scramble_rank_percent': scramble_rank,
    'v4_rigid_null_failing_identical_valid_sites': rigid_sites,
    'v4_rigid_null_fail_mapping_gain_pp':
        rigid_treatment['scored_over_rigid_null_failing_valid_percent'] -
        rigid_control['scored_over_rigid_null_failing_valid_percent'] if rigid_sites else None,
    'v4_near_identical_global_sites': near_sites,
    'v4_near_global_mapping_gain_pp': near_treatment['scored_over_global_percent'] -
        near_control['scored_over_global_percent'] if near_sites else None,
    'v3_exact_global_mapping_gain_pp': v3_exact_treatment['scored_over_global_percent'] -
        v3_exact_control['scored_over_global_percent'] if v3_exact_sites else None,
    'v3_near_global_mapping_gain_pp': v3_near_treatment['scored_over_global_percent'] -
        v3_near_control['scored_over_global_percent'] if v3_near_sites else None}

assert len(pair_rows) == 491
assert len(mapping_rows) == 491 * len(steps) * len(arms) * 2
assert len(ranking_rows) == 491 * len(steps) * 4
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'pair_rows': len(pair_rows),
    'mapping_rows': len(mapping_rows), 'ranking_rows': len(ranking_rows),
    'config_sha256': sha(out / 'config.json'), 'pair_bank_sha256': sha(out / 'pair_bank.jsonl'),
    'mapping_rows_sha256': sha(out / 'mapping_rows.jsonl'),
    'ranking_rows_sha256': sha(out / 'ranking_rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'pair_rows': len(pair_rows),
                  'decision': summary['decision']['status']}), flush=True)
