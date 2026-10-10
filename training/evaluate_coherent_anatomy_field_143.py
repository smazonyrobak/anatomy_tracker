"""Independent exact/near correspondence and support-matched plane-rank DEV readout."""

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
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import (
    dense_local_targets_143, render_coherent_atlas_143)
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COHERENT_ANATOMY_FIELD_143_PROTOCOL_20261010.md'
run = root / 'runs/coherent_anatomy_field_143'
out = root / 'runs/coherent_anatomy_field_143_dev_eval'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('full', 'support_only')
checkpoints = (0, 2000, 5000, 10000)
roles = ('exact', 'blind_near')
angle_bins = ('all', '<15', '15-30', '30-45', '>=45')
device, radius = 'cuda', 6
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == protocol.drive.upper() == run.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_eval_done = json.loads((parent_eval / 'completed.json').read_text())
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000'] == train_config['parent_checkpoint_sha256']
assert sha(parent_dir / 'completed.json') == train_config['parent_completion_sha256']
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(run / 'draws.jsonl') == train_done['draws_sha256']
assert sha(run / 'training.jsonl') == train_done['training_sha256']
assert train_config['protocol_sha256'] == sha(protocol)
assert train_config['source_sha256'] == train_done['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert train_done['steps'] == train_config['steps'] == 10000
assert train_done['accepted_synthetic_physical_sections'] == 30000
assert train_done['v4_presentations'] == 20000 and train_done['v3_presentations'] == 10000
assert train_config['radius'] == radius and tuple(train_config['arms']) == arms
assert train_config['checkpoints'] == list(checkpoints)
assert all(sha(run / f'{arm}_step_{step:05d}.pt') ==
           train_done['checkpoint_sha256'][arm][str(step)]
           for arm in arms for step in checkpoints)
assert all(sha(parent_eval / name) == digest
           for name, digest in parent_eval_done['output_sha256'].items())
assert sha(source / 'evaluate_v4_pose_adaptation_132.py') == parent_eval_done['evaluator_source_sha256']
assert not any(receipt.get(key, False) for receipt in (parent_done, parent_eval_done, train_done)
    for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                'final_animals_used', 'external_pretrained_weights_used'))

parent_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
assert len(parent_rows) == 491
panel_records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert sum(row['eligible'] for row in records) == done['eligible']
    panel_records[cohort] = [row for row in records if row['eligible']]
    assert all((cohort, row['section_id']) in parent_rows for row in panel_records[cohort])
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243
train_plans = {row['plan_receipt'] for row in train_config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for records in panel_records.values() for row in records}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans
assert all(row['lineage']['split'] == 'train'
           for row in train_config['synthetic_provenance']['base_subjects'])
assert {row['panel_physical_section_id'] for row in panel_records['v4']}.isdisjoint(
       {row['panel_physical_section_id'] for row in panel_records['v3']})

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config
model.load_state_dict(saved['model'], strict=True)
del saved
heads = {}
for step in checkpoints:
    for arm in arms:
        saved = torch.load(run / f'{arm}_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
        head = CoherentAnatomyField143(radius=radius).to(device).eval().requires_grad_(False)
        head.load_state_dict(saved['field'], strict=True)
        heads[step, arm] = head
        del saved
assert all(torch.equal(heads[0, 'full'].state_dict()[key], heads[0, 'support_only'].state_dict()[key])
           for key in heads[0, 'full'].state_dict())

axis = (torch.arange(64, device=device) + .5) / 64
gy, gx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((gx, gy), -1)
axis256 = torch.arange(256, device=device) / 256
gy256, gx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((gx256, gy256), -1)
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device=device),
                                  torch.arange(64, device=device), indexing='ij')
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device=device)


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device=device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


def wrong_candidates(state, reflection, valid, section_id):
    observed = chart[valid]
    truth = rigid_points_090(state, reflection, observed)[0]
    generator = torch.Generator(device=device).manual_seed(
        int(hashlib.sha256(section_id.encode()).hexdigest()[:15], 16))
    updates = (2 * torch.rand((64, 9), device=device, generator=generator) - 1) * limits
    updates[:, :3] *= 3
    updates[:, 3:6] *= 3
    updates[:, 6:] *= 2
    candidates = compose_full_frame_state(state.expand(64, -1), updates)
    errors = (rigid_points_090(candidates, reflection.expand(64), observed) -
              truth).norm(dim=-1).mean(-1) / 1000
    eligible = (errors >= 1.5) & (errors <= 3.)
    chosen = (errors - 2.2).abs().masked_fill(~eligible, 100).topk(8, largest=False).indices
    return candidates[chosen], errors[chosen], eligible[chosen]


def global_and_local_support(truth, valid, slabs, base, basis, normal):
    local, labels, reachable = dense_local_targets_143(
        truth, valid, base, basis, normal, radius=radius)
    x = pixel_x[None] + 64 * local[..., 0]
    y = pixel_y[None] + 64 * local[..., 1]
    z = 3 + local[..., 2] / 500
    inside = (x >= 0) & (x <= 63) & (y >= 0) & (y <= 63) & (z >= 0) & (z <= 6)
    grid = torch.stack((2 * x / 63 - 1, 2 * y / 63 - 1, z / 3 - 1), -1)[:, None]
    sampled = F.grid_sample(slabs[:, 1:2].float(), grid, align_corners=True)[:, 0, 0]
    global_support = valid[:, 2::4, 2::4].bool() & inside & (sampled > .5)
    lattice = torch.stack((64 * local[..., 0], 64 * local[..., 1],
                           local[..., 2] / 500), -1).round().long()
    lx = (pixel_x[None] + lattice[..., 0]).clamp(0, 63)
    ly = (pixel_y[None] + lattice[..., 1]).clamp(0, 63)
    lz = (lattice[..., 2] + 3).clamp(0, 6)
    indices = torch.arange(len(base), device=device)[:, None, None]
    local_support = reachable & (slabs[:, 1][indices, lz, ly, lx] > .5)
    return global_support, local_support, labels


def mapping_counts(output, truth, valid, slabs, base, basis, normal, slot):
    global_support, local_support, labels = global_and_local_support(
        truth, valid, slabs[slot:slot + 1], base[slot:slot + 1],
        basis[slot:slot + 1], normal[slot:slot + 1])
    valid64 = valid[:, 2::4, 2::4].bool()[0]
    global_support, local_support = global_support[0], local_support[0]
    estimate = output['offset_chart'][slot].permute(1, 2, 0).float()
    predicted = (base[slot] +
        torch.einsum('ij,hwj->hwi', basis[slot], estimate[..., :2]) +
        normal[slot, None, None] * estimate[..., 2:3])
    truth64 = truth[0, 2::4, 2::4].float()
    distance = (predicted - truth64).norm(dim=-1)
    class_index = output['match_logits'][slot].argmax(0)
    matched = class_index != output['match_logits'].shape[1] - 1
    scored = matched & (distance <= 500)
    scores = distance[valid64]
    du = predicted - base[slot]
    base_h = (base[slot, :, 1:] - base[slot, :, :-1]).norm(dim=-1).clamp_min(1)
    base_v = (base[slot, 1:] - base[slot, :-1]).norm(dim=-1).clamp_min(1)
    strain_h = ((du[:, 1:] - du[:, :-1]).norm(dim=-1) / base_h)[valid64[:, 1:] & valid64[:, :-1]]
    strain_v = ((du[1:] - du[:-1]).norm(dim=-1) / base_v)[valid64[1:] & valid64[:-1]]
    strain = torch.cat((strain_h, strain_v))
    return {'valid_sites': int(valid64.sum()),
            'global_supported_sites': int(global_support.sum()),
            'locally_reachable_sites': int((global_support & local_support).sum()),
            'scored_le_500_um_all_valid': int((valid64 & scored).sum()),
            'scored_le_500_um_global_supported': int((global_support & scored).sum()),
            'scored_le_500_um_locally_reachable': int((global_support & local_support & scored).sum()),
            'argmax_dustbin_valid': int((valid64 & ~matched).sum()),
            'visible_gt_0p5_valid': int((valid64 &
                (output['visibility'][slot, 0] > .5)).sum()),
            'point_error_mm_valid_mean': float(scores.mean() / 1000),
            'point_error_mm_valid_median': float(scores.median() / 1000),
            'warp_rms_mm_valid': float((du[valid64].square().sum(-1).mean().sqrt()) / 1000),
            'warp_relative_neighbor_strain_p95': float(torch.quantile(strain, .95)) if len(strain) else None,
            'local_truth_class_sites': int((labels[0] >= 0).sum()),
            'fit_energy': float(output['energy'][slot])}


def summarize(group):
    plans = defaultdict(list)
    for row in group:
        plans[row['subject_plan_id']].append(row)
    sections = len(group)
    valid = sum(row['valid_sites'] for row in group)
    global_count = sum(row['global_supported_sites'] for row in group)
    local = sum(row['locally_reachable_sites'] for row in group)
    hits_global = sum(row['scored_le_500_um_global_supported'] for row in group)
    hits_all = sum(row['scored_le_500_um_all_valid'] for row in group)
    return {'sections': sections, 'plans': len(plans), 'valid_sites': valid,
        'global_supported_sites': global_count, 'locally_reachable_sites': local,
        'scored_le_500_um_global_supported': hits_global,
        'scored_le_500_um_all_valid': hits_all,
        'global_support_coverage_percent': 100 * global_count / valid if valid else None,
        'local_reach_over_global_percent': 100 * local / global_count if global_count else None,
        'scored_over_global_percent': 100 * hits_global / global_count if global_count else None,
        'scored_over_all_valid_percent': 100 * hits_all / valid if valid else None,
        'dustbin_over_valid_percent': 100 * sum(r['argmax_dustbin_valid'] for r in group) / valid if valid else None,
        'point_error_mm_plan_equal': float(np.mean([np.mean([r['point_error_mm_valid_mean']
            for r in rows]) for rows in plans.values()])) if plans else None,
        'warp_rms_mm_plan_equal': float(np.mean([np.mean([r['warp_rms_mm_valid']
            for r in rows]) for rows in plans.values()])) if plans else None,
        'warp_strain_p95_section_median': float(np.median([r['warp_relative_neighbor_strain_p95']
            for r in group if r['warp_relative_neighbor_strain_p95'] is not None])) if sections else None}


out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_evaluation_completion_sha256': sha(parent_eval / 'completed.json'),
    'panel_receipt_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'checkpoints': checkpoints, 'arms': arms, 'cohorts': {k: len(v) for k, v in panel_records.items()},
    'global_denominator': 'valid observed tissue with truth CCF point inside the rendered 7x64x64 slab grid and trilinear atlas tissue support >0.5; no local-radius or model-score filtering',
    'scored_hit': 'field argmax is not dustbin and continuous predicted CCF mapping is within 500um of dense truth; both all-valid and global-supported denominators reported',
    'blind_near': 'oracle truth-best original branch within frozen 132 blind16; near<=1.5mm subset separately reported, not a deployable pose selector',
    'wrong_plane': 'seeded state perturbations from exact pose; 1.5-3.0mm rigid error and central rendered tissue support fraction within 0.1 of exact pose; no model energy used for selection',
    'rank': 'uncalibrated exact-pose energy lower than support-matched wrong-pose energy',
    'split': 'train 64 deformation plans versus eight disjoint synthetic development plans; no new physical animals',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        for index, record in enumerate(panel_records[cohort]):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).to(device).bool()
                truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
            assert int(valid.sum()) == record['valid_pixels']
            prediction = model.predict(image)
            prior, ids = beam_for(prediction)
            observed = chart256[valid[0]]
            target_rigid = rigid_points_090(state, reflection, observed)[0]
            proposed = rigid_points_090(prediction['state'][:, ids // 2],
                                        (ids % 2)[None], observed)[0]
            errors = (proposed - target_rigid).norm(dim=-1).mean(-1) / 1000
            near_slot = int(errors.argmin())
            branch = int(ids[near_slot])
            frozen = parent_rows[cohort, record['section_id']]
            assert ids.tolist() == frozen['beam_branch_ids']
            assert abs(float(errors[near_slot]) - frozen['beam_best_rigid_mm']) < 1e-4
            wrong_states, wrong_errors, eligible = wrong_candidates(
                state, reflection, valid[0, 2::4, 2::4], record['section_id'])
            wrong_slabs, _, _, _ = render_coherent_atlas_143(
                atlas, wrong_states, reflection.expand(len(wrong_states)),
                offsets.expand(len(wrong_states), -1), weights.expand(len(wrong_states), -1))
            exact_slab, _, _, _ = render_coherent_atlas_143(atlas, state, reflection, offsets, weights)
            exact_support = exact_slab[:, 1, 3].mean()
            support_difference = (wrong_slabs[:, 1, 3].mean((1, 2)) - exact_support).abs()
            matched = eligible & (support_difference <= .1)
            wrong_slot = int((wrong_errors - 2.2).abs().masked_fill(~matched, 100).argmin())
            rank_allowed = bool(matched.any())
            candidate_states = torch.cat((state, prediction['state'][:, branch // 2],
                                          wrong_states[wrong_slot:wrong_slot + 1]))
            candidate_reflection = torch.cat((reflection, ids[near_slot:near_slot + 1] % 2,
                                              reflection))
            slabs, base, basis, normal = render_coherent_atlas_143(atlas, candidate_states,
                candidate_reflection, offsets.expand(3, -1), weights.expand(3, -1))
            candidate_image = image.expand(3, -1, -1, -1)
            candidate_feature = prediction['feature'].expand(3, -1, -1, -1)
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            angle_bin = '<15' if angle < 15 else '15-30' if angle < 30 else '30-45' if angle < 45 else '>=45'
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'appearance_mode': record['appearance_mode'],
                'nearest_cardinal_angle_deg': angle, 'nearest_cardinal_angle_bin_deg': angle_bin,
                'blind_near_branch_id': branch, 'blind_near_rigid_mm': float(errors[near_slot]),
                'blind_near_available_1p5mm': bool(errors[near_slot] <= 1.5),
                'wrong_rigid_mm': float(wrong_errors[wrong_slot]),
                'wrong_support_fraction_difference': float(support_difference[wrong_slot]),
                'rank_support_matched': rank_allowed}
            for step in checkpoints:
                for arm in arms:
                    atlas_input = slabs if arm == 'full' else torch.cat((
                        torch.zeros_like(slabs[:, :1]), slabs[:, 1:2]), 1)
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = heads[step, arm](candidate_image, atlas_input, 500.,
                                                  source_feature=candidate_feature)
                    energies = output['energy'].float()
                    for role, slot in (('exact', 0), ('blind_near', 1)):
                        counts = mapping_counts(output, truth, valid, slabs,
                            base, basis, normal, slot)
                        row = {**common, 'step': step, 'arm': arm, 'role': role,
                            'fit_energy': float(energies[slot]),
                            'rank_exact_below_wrong': bool(energies[0] < energies[2]) if rank_allowed else None,
                            'rank_energy_margin': float(energies[2] - energies[0]) if rank_allowed else None,
                            **counts}
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        rows.append(row)
            if (index + 1) % 64 == 0:
                stream.flush()
                print(json.dumps({'event': 'development_milestone', 'cohort': cohort,
                                  'sections': index + 1}), flush=True)

summary = {'metrics': {}, 'ranking': {},
    'warning': 'Exact pose and truth-best blind-near branch are DEV-truth oracle diagnostics. '
               'The energy ranking uses seeded support-matched synthetic wrong poses, not expert physical truth.'}
for cohort in panels:
    summary['metrics'][cohort] = {}
    summary['ranking'][cohort] = {}
    for step in checkpoints:
        summary['metrics'][cohort][str(step)] = {}
        summary['ranking'][cohort][str(step)] = {}
        for arm in arms:
            summary['metrics'][cohort][str(step)][arm] = {}
            arm_rows = [r for r in rows if r['cohort'] == cohort and
                        r['step'] == step and r['arm'] == arm]
            for role in roles:
                summary['metrics'][cohort][str(step)][arm][role] = {}
                for angle_bin in angle_bins:
                    group = [r for r in arm_rows if r['role'] == role and
                             (angle_bin == 'all' or r['nearest_cardinal_angle_bin_deg'] == angle_bin)]
                    summary['metrics'][cohort][str(step)][arm][role][angle_bin] = {
                        'all_sections': summarize(group),
                        'blind_near_le_1p5mm': summarize([r for r in group
                            if r['blind_near_available_1p5mm']])}
            ranked = [r for r in arm_rows if r['role'] == 'exact' and r['rank_support_matched']]
            summary['ranking'][cohort][str(step)][arm] = {
                'eligible_sections': len(ranked), 'all_sections': len(arm_rows) // 2,
                'correct_exact_below_wrong': sum(r['rank_exact_below_wrong'] for r in ranked),
                'correct_percent': 100 * sum(r['rank_exact_below_wrong'] for r in ranked) /
                    len(ranked) if ranked else None,
                'mean_energy_margin': float(np.mean([r['rank_energy_margin'] for r in ranked]))
                    if ranked else None}


def metric(cohort, step, arm, role, scope='all_sections'):
    return summary['metrics'][cohort][str(step)][arm][role]['all'][scope]


exact_full = metric('v4', 10000, 'full', 'exact')
exact_support = metric('v4', 10000, 'support_only', 'exact')
near_full = metric('v4', 10000, 'full', 'blind_near', 'blind_near_le_1p5mm')
rank_full = summary['ranking']['v4']['10000']['full']
rank_support = summary['ranking']['v4']['10000']['support_only']
assert exact_full['sections'] == exact_support['sections'] == 248
assert exact_full['global_supported_sites'] == exact_support['global_supported_sites']
assert near_full['sections'] == 179
assert rank_full['eligible_sections'] == rank_support['eligible_sections']
gate = {'exact_v4_full_scored_over_global_percent': exact_full['scored_over_global_percent'],
    'exact_v4_full_minus_support_percentage_points':
        exact_full['scored_over_global_percent'] - exact_support['scored_over_global_percent'],
    'blind_near_v4_scored_over_global_percent': near_full['scored_over_global_percent'],
    'blind_near_v4_global_supported_sites': near_full['global_supported_sites'],
    'rank_v4_full_correct_percent': rank_full['correct_percent'],
    'rank_v4_full_minus_support_percentage_points':
        (rank_full['correct_percent'] - rank_support['correct_percent'])
        if rank_full['eligible_sections'] else None,
    'rank_v4_support_matched_sections': rank_full['eligible_sections'],
    'thresholds': {'exact_scored_percent': 50., 'exact_vs_support_percentage_points': 10.,
        'blind_near_scored_percent': 35., 'rank_correct_percent': 75.,
        'rank_vs_support_percentage_points': 10.}}
gate['pass'] = bool(exact_full['global_supported_sites'] and near_full['global_supported_sites'] and
    rank_full['eligible_sections'] and gate['exact_v4_full_scored_over_global_percent'] >= 50 and
    gate['exact_v4_full_minus_support_percentage_points'] >= 10 and
    gate['blind_near_v4_scored_over_global_percent'] >= 35 and
    gate['rank_v4_full_correct_percent'] >= 75 and
    gate['rank_v4_full_minus_support_percentage_points'] >= 10)
summary['decision'] = gate
assert len(rows) == 491 * len(checkpoints) * len(arms) * len(roles)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows), 'gate_pass': gate['pass']}), flush=True)
