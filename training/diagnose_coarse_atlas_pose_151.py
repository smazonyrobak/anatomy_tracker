"""Posthoc oracle decomposition of frozen 150 matcher on its fresh synthetic DEV panel."""

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
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coarse_atlas_pose_150 import CoarseAtlasPose150
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_151_POSTHOC_ORACLE_PROTOCOL_20261011.md'
panel = root / 'data/coarse_atlas_pose_150_fresh_dev_panel'
train150 = root / 'runs/coarse_atlas_pose_150'
eval150 = root / 'runs/coarse_atlas_pose_150_fresh_dev_eval'
pose_run = root / 'runs/v4_pose_adaptation_132'
out = root / 'runs/coarse_atlas_pose_151_posthoc_oracle'
expected_completion_sha256 = {
    'training': '07f9852ddc7176dcb2e77db8510be671fe8ee04e6919c87397d57fa5785031fe',
    'panel': 'ffdd678cad121ff86b0e88ea582dafb8cb6497da86e5803a78c89fbef0656031',
    'evaluation': '02e83c4c0910d933021a1045659a865d29e1f1f39a11e85d869341a6e1050224',
}
fits = ('learned', 'true_key_learned_weight', 'true_key_valid_weight',
        'true_continuous_valid_weight', 'true_rigid_valid_weight')
roles = ('exact', 'blind_near')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert all(path.drive.upper() == 'I:' for path in
           (source, protocol, panel, train150, eval150, pose_run, out))
assert not out.exists()
assert sha(train150 / 'completed.json') == expected_completion_sha256['training']
assert sha(panel / 'completed.json') == expected_completion_sha256['panel']
assert sha(eval150 / 'completed.json') == expected_completion_sha256['evaluation']

train_done = json.loads((train150 / 'completed.json').read_text())
train_config = json.loads((train150 / 'config.json').read_text())
panel_done = json.loads((panel / 'completed.json').read_text())
panel_config = json.loads((panel / 'protocol.json').read_text())
eval_done = json.loads((eval150 / 'completed.json').read_text())
eval_config = json.loads((eval150 / 'config.json').read_text())
pose_done = json.loads((pose_run / 'completed.json').read_text())
pose_config = json.loads((pose_run / 'config.json').read_text())
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
references = {(row['section_id'], row['role']): row for row in
              (json.loads(line) for line in (eval150 / 'rows.jsonl').open())
              if row['arm'] == 'full' and row['role'] in roles}
assert sha(train150 / 'config.json') == train_done['config_sha256']
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(eval150 / 'config.json') == eval_done['config_sha256']
assert sha(eval150 / 'rows.jsonl') == eval_done['rows_sha256']
assert sha(eval150 / 'summary.json') == eval_done['summary_sha256']
assert sha(source / 'evaluate_coarse_atlas_pose_150.py') == eval_done['evaluator_sha256']
assert sha(source.parent / 'docs/publication/COARSE_ATLAS_POSE_150_PROTOCOL_20261010.md') == train_config['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert all(sha(panel / 'source' / name) == digest for name, digest in panel_config['source_sha256'].items())
assert len(records) == panel_done['physical_sections'] == 256
assert sum(row['eligible'] for row in records) == panel_done['eligible'] == 243
assert len(references) == panel_done['eligible'] * len(roles)
assert all(row['provenance']['split'] == 'development' for row in records)
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
assert len({row['plan_receipt_sha256'] for row in records}) == 8
train_plans = {row['plan_receipt'] for row in train_config['synthetic_provenance']['base_subjects']}
assert len(train_plans) == 64 and not train_plans & {row['plan_receipt_sha256'] for row in records}
assert train_done['selected_step'] == eval_config['selected_step'] == 6000
assert eval_config['panel_completion_sha256'] == expected_completion_sha256['panel']
assert eval_config['train150_completion_sha256'] == expected_completion_sha256['training']
assert not any(receipt.get(key, False) for receipt in
               (train_done, train_config, eval_done, eval_config, pose_done)
               for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                           'final_animals_used', 'external_pretrained_weights_used'))

pose_checkpoint = pose_run / 'joint_step_02000.pt'
matcher_checkpoint = train150 / 'full_step_06000.pt'
assert sha(pose_checkpoint) == pose_done['checkpoint_sha256']['2000'] == train_config['parent_checkpoint_sha256']
assert sha(matcher_checkpoint) == train_done['checkpoint_sha256']['full']['6000']
atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved = torch.load(pose_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == pose_config
pose.load_state_dict(saved['model'], strict=True)
del saved
matcher = CoarseAtlasPose150().cuda().eval().requires_grad_(False)
saved = torch.load(matcher_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 6000 and saved['arm'] == 'full' and saved['config'] == train_config
matcher.load_state_dict(saved['matcher'], strict=True)
del saved

axis256 = torch.arange(256, device='cuda') / 256
gy256, gx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((gx256, gy256), -1)
ridge = torch.diag(torch.tensor((24., 12., 12.), device='cuda'))
config = {'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
    'input_completion_sha256': expected_completion_sha256,
    'pose_checkpoint_sha256': sha(pose_checkpoint),
    'matcher_checkpoint_sha256': sha(matcher_checkpoint),
    'fits': fits, 'roles': roles, 'eligible_sections': panel_done['eligible'],
    'query': '150 24x24 source grid; exact observed CCF and validity bilinear-sampled from frozen native256 maps; valid >0.99',
    'key': '150 supported 24x24x9 lattice; candidate-local query-to-key <=3mm; nearest to true observed CCF; unreachable key queries zero weighted',
    'score': 'mean corrected-to-true-rigid Euclidean CCF error over valid native256 sites',
    'scope': 'posthoc synthetic DEV mechanism diagnostic, not a new gate or blind inference',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for index, record in enumerate(records):
        if not record['eligible']:
            continue
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            dense = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        prediction = pose.predict(image)
        prior_log = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior_log[:, :32].topk(8, -1).indices,
                          prior_log[:, 32:].topk(6, -1).indices + 32), -1)
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        for _ in range(2):
            selected_normals = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
            diversity = -(normals[:, 16:, None] * selected_normals[:, None]).sum(-1).abs().amax(-1)
            diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
            anchor = diversity.argmax(-1)
            reflected = prior_log[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
            beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
        observed = chart256[valid]
        target = rigid_points_090(state, reflection, observed)[0]
        proposed = rigid_points_090(prediction['state'][:, beam[0] // 2],
                                    beam % 2, observed)[0]
        beam_errors = (proposed - target).norm(dim=-1).mean(-1) / 1000
        near_slot = int(beam_errors.argmin())
        near_id = int(beam[0, near_slot])
        states = torch.cat((state[:, None],
            prediction['state'][:, near_id // 2:near_id // 2 + 1]), 1)
        reflections = torch.stack((reflection[0], reflection.new_tensor(near_id % 2)))[None]

        with torch.autocast('cuda', dtype=torch.float16):
            output = matcher(image, atlas, states, reflections, offsets, weights,
                             atlas_intensity=True)
        source_grid = output['source_grid'][0]
        source_chart = output['source_chart'][0].float()
        query_grid = (2 * source_grid - 1).reshape(1, 24, 24, 2)
        valid_query = (F.grid_sample(valid[None, None].float(), query_grid,
            mode='bilinear', align_corners=False).flatten() > .99)
        dense_query = F.grid_sample(dense.permute(0, 3, 1, 2), query_grid,
            mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)[0]
        rigid_query = rigid_points_090(state, reflection, source_grid - .5 / 256)[0]
        with torch.autocast('cuda', dtype=torch.float16):
            centre, frame, basis = full_frame_state_to_components(states[0])
            edges = frame[:, :, :2] @ basis
            query_ccf = centre[:, None] + torch.einsum('nij,nqj->nqi', edges, source_chart - .5)
        keys = output['key_ccf'][0].float()
        key_support = output['key_support'][0]
        local = ((torch.cdist(query_ccf.float() / 1000, keys / 1000) <= 3.) &
                 (key_support[:, None] > .5))
        distance = torch.cdist(dense_query[None].expand(2, -1, -1) / 1000, keys / 1000)
        distance.masked_fill_(~local, torch.inf)
        nearest_mm, key_index = distance.min(-1)
        reachable = torch.isfinite(nearest_mm)
        true_key = keys.gather(1, key_index[..., None].expand(-1, -1, 3))

        learned_weight = output['effective_fit_weight'][0].float()
        valid_weight = valid_query.float()[None].expand(2, -1)
        correspondence = torch.stack((output['expected_ccf'][0].float(), true_key,
            true_key, dense_query[None].expand(2, -1, -1),
            rigid_query[None].expand(2, -1, -1)))
        fit_weight = torch.stack((learned_weight, learned_weight * reachable,
            valid_weight * reachable, valid_weight, valid_weight))
        design = torch.cat((torch.ones_like(source_chart[..., :1]), source_chart - .5), -1)
        prior = torch.stack((centre.float(), edges[:, :, 0].float(),
                             edges[:, :, 1].float()), -2)
        design = design[None].expand(len(fits), -1, -1, -1)
        lhs = torch.einsum('mcqi,mcq,mcqj->mcij', design, fit_weight, design) + ridge
        rhs = torch.einsum('mcqi,mcq,mcqj->mcij', design, fit_weight, correspondence) + ridge @ prior
        fitted = torch.linalg.solve(lhs, rhs)
        fitted_ouv = torch.stack((fitted[..., 0, :] - .5 * (fitted[..., 1, :] + fitted[..., 2, :]),
                                  fitted[..., 1, :], fitted[..., 2, :]), -2)
        corrected = full_frame_state_from_components(*physical_ouv_to_frame(fitted_ouv))
        assert torch.allclose(corrected[0], output['corrected_state'][0].float(), atol=1e-3, rtol=1e-5)
        after = (rigid_points_090(corrected, reflections[0], observed) - target).norm(dim=-1).mean(-1) / 1000
        assert bool(torch.isfinite(after).all())

        for slot, role in enumerate(roles):
            reference = references[record['section_id'], role]
            before = 0. if slot == 0 else float(beam_errors[near_slot])
            assert reference['panel_file_sha256'] == record['sha256']
            assert reference['synthetic_subject_plan_id'] == record['synthetic_subject_plan_id']
            assert reference['valid_sites'] == int(valid.sum())
            assert reference['initial_branch_id'] == (None if slot == 0 else near_id)
            assert abs(reference['before_rigid_mm'] - before) < 1e-4
            assert abs(float(after[0, slot]) - reference['after_rigid_mm']) < 1e-3
            fit_mm = {name: float(after[i, slot]) for i, name in enumerate(fits)}
            reachable_valid = reachable[slot] & valid_query
            row = {'section_id': record['section_id'], 'role': role,
                'physical_section_id': record['panel_physical_section_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'plan_receipt_sha256': record['plan_receipt_sha256'],
                'animal_id': record['animal_id'], 'subject_id': record['subject_id'],
                'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'panel_file_sha256': record['sha256'],
                'appearance_mode': record['appearance_mode'],
                'nearest_cardinal_angle_deg': reference['nearest_cardinal_angle_deg'],
                'nearest_cardinal_angle_bin': reference['nearest_cardinal_angle_bin'],
                'valid_native256_sites': int(valid.sum()),
                'initial_branch_id': None if slot == 0 else near_id,
                'before_rigid_mm': before,
                'blind_near_initial_le_1p5mm': bool(beam_errors[near_slot] <= 1.5),
                'valid_query_sites': int(valid_query.sum()),
                'key_reachable_valid_query_sites': int(reachable_valid.sum()),
                'key_unmatched_valid_query_sites': int((~reachable[slot] & valid_query).sum()),
                'key_nearest_gt_0p9mm_valid_query_sites': int((reachable_valid &
                    (nearest_mm[slot] > .9)).sum()),
                'nearest_key_mm_mean_reachable_valid': float(nearest_mm[slot][reachable_valid].mean())
                    if bool(reachable_valid.any()) else None,
                'fit_mm': fit_mm}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if (index + 1) % 64 == 0:
            stream.flush()
            print(json.dumps({'event': 'section_milestone', 'processed_panel_sections': index + 1}), flush=True)

summary = {'scope': config['scope'], 'metrics': {}}
for role in roles:
    summary['metrics'][role] = {}
    for scope in ('all', 'blind_near_initial_le_1p5mm'):
        summary['metrics'][role][scope] = {}
        for mode in ('all', 'raw', 'exact_black', 'imperfect_brush'):
            summary['metrics'][role][scope][mode] = {}
            for angle in ('all', '<15', '15-30', '30-45', '>=45'):
                group = [row for row in rows if row['role'] == role and
                         (scope == 'all' or row['blind_near_initial_le_1p5mm']) and
                         (mode == 'all' or row['appearance_mode'] == mode) and
                         (angle == 'all' or row['nearest_cardinal_angle_bin'] == angle)]
                plans = defaultdict(list)
                for row in group:
                    plans[row['synthetic_subject_plan_id']].append(row)
                queries = sum(row['valid_query_sites'] for row in group)
                reached = sum(row['key_reachable_valid_query_sites'] for row in group)
                plan_reach = [np.mean([
                    100 * row['key_reachable_valid_query_sites'] / row['valid_query_sites']
                    for row in part if row['valid_query_sites']]) for part in plans.values()
                    if any(row['valid_query_sites'] for row in part)]
                summary['metrics'][role][scope][mode][angle] = {
                    'sections': len(group), 'synthetic_plans': len(plans),
                    'valid_query_sites': queries, 'key_reachable_valid_query_sites': reached,
                    'sections_with_zero_valid_queries': sum(not row['valid_query_sites'] for row in group),
                    'key_reachable_percent': 100 * reached / queries if queries else None,
                    'plan_equal_key_reachable_percent': float(np.mean(plan_reach))
                        if plan_reach else None,
                    'section_equal_fit_mm': {name: float(np.mean([
                        row['fit_mm'][name] for row in group])) if group else None for name in fits},
                    'plan_equal_fit_mm': {name: float(np.mean([
                        np.mean([row['fit_mm'][name] for row in part])
                        for part in plans.values()])) if plans else None for name in fits}}
assert len(rows) == panel_done['eligible'] * len(roles)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'script_sha256': sha(__file__),
    'protocol_sha256': sha(protocol), 'input_completion_sha256': expected_completion_sha256,
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows)}), flush=True)
