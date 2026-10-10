"""Frozen synthetic DEV correspondence/weighting oracle for 148 plane correction."""

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
from training.coarse_atlas_pose_148 import CoarseAtlasPose148
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_149_ORACLE_DIAGNOSTIC_PROTOCOL_20261010.md'
pose_run = root / 'runs/v4_pose_adaptation_132'
pose_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
train148 = root / 'runs/coarse_atlas_pose_148'
eval148 = root / 'runs/coarse_atlas_pose_148_dev_eval'
out = root / 'runs/coarse_atlas_pose_149_oracle_diagnostic'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
fits = ('learned', 'key_learned_weight', 'key_valid_weight',
        'dense_valid_weight', 'rigid_valid_weight')
roles = ('exact', 'blind_near')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert all(path.drive.upper() == 'I:' for path in
           (root, source, protocol, pose_run, pose_eval, train148, eval148, out, *panels.values()))
assert not out.exists()

pose_done = json.loads((pose_run / 'completed.json').read_text())
pose_config = json.loads((pose_run / 'config.json').read_text())
pose_eval_done = json.loads((pose_eval / 'completed.json').read_text())
train_done = json.loads((train148 / 'completed.json').read_text())
train_config = json.loads((train148 / 'config.json').read_text())
eval_done = json.loads((eval148 / 'completed.json').read_text())
eval_config = json.loads((eval148 / 'config.json').read_text())
checkpoint = pose_run / 'joint_step_02000.pt'
matcher_checkpoint = train148 / 'full_step_06000.pt'
assert hashlib.sha256((pose_run / 'config.json').read_bytes()).hexdigest() == pose_done['config_sha256']
assert hashlib.sha256((train148 / 'config.json').read_bytes()).hexdigest() == train_done['config_sha256']
assert hashlib.sha256((eval148 / 'config.json').read_bytes()).hexdigest() == eval_done['config_sha256']
assert hashlib.sha256((eval148 / 'rows.jsonl').read_bytes()).hexdigest() == eval_done['rows_sha256']
assert hashlib.sha256((eval148 / 'summary.json').read_bytes()).hexdigest() == eval_done['summary_sha256']
assert hashlib.sha256((pose_eval / 'synthetic_rows.jsonl').read_bytes()).hexdigest() == pose_eval_done['output_sha256']['synthetic_rows.jsonl']
assert hashlib.sha256((source / 'evaluate_coarse_atlas_pose_148.py').read_bytes()).hexdigest() == eval_done['evaluator_sha256']
assert hashlib.sha256((source.parent / 'docs/publication/COARSE_ATLAS_POSE_148_PROTOCOL_20261010.md').read_bytes()).hexdigest() == train_config['protocol_sha256'] == eval_config['protocol_sha256']
assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
           for name, digest in train_config['source_sha256'].items())
with checkpoint.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == pose_done['checkpoint_sha256']['2000'] == train_config['parent_checkpoint_sha256']
with matcher_checkpoint.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == train_done['checkpoint_sha256']['full']['6000']
assert eval_config['train_completion_sha256'] == eval_done['train_completion_sha256'] == hashlib.sha256((train148 / 'completed.json').read_bytes()).hexdigest()
assert eval_config['parent_evaluation_completion_sha256'] == hashlib.sha256((pose_eval / 'completed.json').read_bytes()).hexdigest()
assert train_done['steps'] == 6000 and not train_done['pose_unfrozen'] and not train_done['joint_feedback_trained']
assert not any(receipt.get(key, False) for receipt in
               (pose_done, pose_eval_done, train_done, eval_done, train_config, eval_config)
               for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                           'final_animals_used', 'external_pretrained_weights_used'))

frozen132 = {(row['cohort'], row['section_id']): row for row in
             (json.loads(line) for line in (pose_eval / 'synthetic_rows.jsonl').open())
             if row['arm'] == '2000'}
frozen148 = {(row['cohort'], row['section_id'], row['role'], row['step'], row['arm']): row
             for row in (json.loads(line) for line in (eval148 / 'rows.jsonl').open())
             if row['role'] in roles and row['step'] in (3000, 6000)}
assert len(frozen132) == 491 and len(frozen148) == 491 * 2 * 2 * 2
records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert hashlib.sha256((panel / 'protocol.json').read_bytes()).hexdigest() == done['protocol_sha256']
    assert hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest() == done['records_sha256']
    assert all(hashlib.sha256((panel / 'source' / name).read_bytes()).hexdigest() == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(all_records) == done['physical_sections'] == 256
    assert all(row['provenance']['split'] == 'development' for row in all_records)
    records[cohort] = [row for row in all_records if row['eligible']]
    assert len(records[cohort]) == done['eligible'] == (248 if cohort == 'v4' else 243)
    assert hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest() == eval_config['panel_receipt_sha256'][cohort]
    assert all((cohort, row['section_id']) in frozen132 for row in records[cohort])
train_plans = {row['plan_receipt'] for row in train_config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for group in records.values() for row in group}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == pose_config
pose.load_state_dict(saved['model'], strict=True)
del saved
matcher = CoarseAtlasPose148().cuda().eval().requires_grad_(False)
saved = torch.load(matcher_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 6000 and saved['arm'] == 'full' and saved['config'] == train_config
matcher.load_state_dict(saved['matcher'], strict=True)
del saved

axis256 = torch.arange(256, device='cuda') / 256
gy256, gx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((gx256, gy256), -1)
ridge = torch.diag(torch.tensor((24., 12., 12.), device='cuda'))
out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'pose_checkpoint_sha256': pose_done['checkpoint_sha256']['2000'],
    'matcher_checkpoint_sha256': train_done['checkpoint_sha256']['full']['6000'],
    'evaluation_148_completion_sha256': hashlib.sha256((eval148 / 'completed.json').read_bytes()).hexdigest(),
    'panel_completion_sha256': {cohort: hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest()
                                for cohort, panel in panels.items()},
    'fits': fits, 'roles': roles, 'cohorts': {cohort: len(group) for cohort, group in records.items()},
    'query': '148 24x24 source grid; true observed CCF bilinear-sampled like 148 training; ideal weight is sampled valid mask >0.99',
    'key': '148 full-step6000 supported 24x24x9 lattice; candidate query-to-key distance <=3mm; no-key queries have zero fit weight',
    'score': 'mean corrected-to-original-rigid Euclidean CCF error over all valid native256 observed sites',
    'warning': 'Frozen synthetic DEV diagnostic only; truth-best beam and oracle correspondences are not implementable inference or physical validation',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        for record in records[cohort]:
            assert hashlib.sha256((panel / record['file']).read_bytes()).hexdigest() == record['sha256']
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
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior_log[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            observed = chart256[valid]
            target = rigid_points_090(state, reflection, observed)[0]
            proposed = rigid_points_090(prediction['state'][:, beam[0] // 2],
                                        (beam % 2), observed)[0]
            beam_errors = (proposed - target).norm(dim=-1).mean(-1) / 1000
            near_slot = int(beam_errors.argmin())
            near_id = int(beam[0, near_slot])
            frozen = frozen132[cohort, record['section_id']]
            assert beam[0].tolist() == frozen['beam_branch_ids']
            assert near_id == frozen['beam_branch_ids'][int(np.argmin(frozen['beam_rigid_mm']))]
            assert abs(float(beam_errors[near_slot]) - frozen['beam_best_rigid_mm']) < 1e-4
            states = torch.cat((state[:, None], prediction['state'][:, near_id // 2:near_id // 2 + 1]), 1)
            reflections = torch.stack((reflection[0], reflection.new_tensor(near_id % 2)))[None]

            with torch.autocast('cuda', dtype=torch.float16):
                output = matcher(image, atlas, states, reflections, offsets, weights)
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
                query_ccf = centre[:, None] + torch.einsum(
                    'nij,nqj->nqi', edges, source_chart - .5)
            keys = output['key_ccf'][0].float()
            local = (torch.cdist(query_ccf.float() / 1000, keys / 1000) <= 3.) & \
                    (output['key_support'][0, :, None] > .5)
            distance = torch.cdist(dense_query[None].expand(2, -1, -1) / 1000, keys / 1000)
            distance.masked_fill_(~local, torch.inf)
            nearest_mm, key_index = distance.min(-1)
            reachable = torch.isfinite(nearest_mm)
            nearest_key = keys.gather(1, key_index[..., None].expand(-1, -1, 3))

            learned_weight = output['confidence'][0].float()
            ideal_weight = valid_query.float()[None].expand(2, -1)
            correspondence = torch.stack((output['expected_ccf'][0].float(), nearest_key,
                nearest_key, dense_query[None].expand(2, -1, -1),
                rigid_query[None].expand(2, -1, -1)))
            fit_weight = torch.stack((learned_weight, learned_weight * reachable,
                ideal_weight * reachable, ideal_weight, ideal_weight))
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
            predicted = rigid_points_090(corrected, reflections[0], observed)
            after = (predicted - target).norm(dim=-1).mean(-1) / 1000
            assert bool(torch.isfinite(after).all())
            for slot, role in enumerate(roles):
                references = {f'{step}_{arm}': frozen148[cohort, record['section_id'], role, step, arm]
                              for step in (3000, 6000) for arm in ('full', 'support_only')}
                assert all(ref['panel_file_sha256'] == record['sha256'] and
                           ref['valid_sites'] == int(valid.sum()) and
                           ref['initial_branch_id'] == (None if slot == 0 else near_id) and
                           abs(ref['before_rigid_mm'] - (0. if slot == 0 else
                               float(beam_errors[near_slot]))) < 1e-4
                           for ref in references.values())
                assert abs(float(after[0, slot]) - references['6000_full']['after_rigid_mm']) < 1e-3
                fit_mm = {name: float(after[i, slot]) for i, name in enumerate(fits)}
                trained_mm = {name: ref['after_rigid_mm'] for name, ref in references.items()}
                metrics = {'before_rigid_mm': 0. if slot == 0 else float(beam_errors[near_slot])}
                metrics.update({f'fit_{name}_mm': value for name, value in fit_mm.items()})
                metrics.update({f'trained_{name}_mm': value for name, value in trained_mm.items()})
                metrics.update({f'gap_trained_full_{step}_minus_{name}_mm':
                    trained_mm[f'{step}_full'] - value
                    for step in (3000, 6000) for name, value in fit_mm.items()})
                row = {'cohort': cohort, 'section_id': record['section_id'], 'role': role,
                    'physical_section_id': record['panel_physical_section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'plan_receipt_sha256': record['plan_receipt_sha256'],
                    'panel_file_sha256': record['sha256'], 'valid_sites': int(valid.sum()),
                    'initial_branch_id': None if slot == 0 else near_id,
                    'initial_reflection': int(reflections[0, slot]),
                    'blind_near_initial_le_1p5mm': bool(beam_errors[near_slot] <= 1.5),
                    'query_sites': int(valid_query.sum()),
                    'key_reachable_query_sites': int((reachable[slot] & valid_query).sum()),
                    'key_unmatched_query_sites': int((~reachable[slot] & valid_query).sum()),
                    'key_nearest_gt_0p9mm_query_sites': int((reachable[slot] & valid_query &
                        (nearest_mm[slot] > .9)).sum()),
                    'key_unmatched_all_576_sites': int((~reachable[slot]).sum()),
                    'nearest_key_mm_mean_reachable_valid': float(nearest_mm[slot][reachable[slot] & valid_query].mean())
                        if bool((reachable[slot] & valid_query).any()) else None,
                    'metrics_mm': metrics}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'event': 'cohort_complete', 'cohort': cohort,
                          'sections': len(records[cohort])}), flush=True)

summary = {'cohorts': {}, 'warning': config['warning']}
for cohort in panels:
    summary['cohorts'][cohort] = {}
    for role in roles:
        summary['cohorts'][cohort][role] = {}
        for scope in ('all', 'blind_near_initial_le_1p5mm'):
            group = [row for row in rows if row['cohort'] == cohort and row['role'] == role and
                     (scope == 'all' or row['blind_near_initial_le_1p5mm'])]
            plans = defaultdict(list)
            for row in group:
                plans[row['subject_plan_id']].append(row)
            names = group[0]['metrics_mm'] if group else ()
            section_means = {name: float(np.mean([row['metrics_mm'][name] for row in group]))
                             for name in names}
            plan_means = {name: float(np.mean([np.mean([
                row['metrics_mm'][name] for row in part]) for part in plans.values()]))
                for name in names}
            matched = sum(row['key_reachable_query_sites'] for row in group)
            queries = sum(row['query_sites'] for row in group)
            section_reach = [row['key_reachable_query_sites'] / row['query_sites']
                             if row['query_sites'] else 0. for row in group]
            plan_reach = [np.mean([row['key_reachable_query_sites'] / row['query_sites']
                                  if row['query_sites'] else 0. for row in part])
                          for part in plans.values()]
            summary['cohorts'][cohort][role][scope] = {
                'sections': len(group), 'plans': len(plans),
                'valid_native256_sites': sum(row['valid_sites'] for row in group),
                'query_sites': queries, 'key_reachable_query_sites': matched,
                'key_unmatched_query_sites': queries - matched,
                'key_reachable_percent': 100 * matched / queries if queries else None,
                'section_equal_key_reachable_percent': 100 * float(np.mean(section_reach))
                    if section_reach else None,
                'plan_equal_key_reachable_percent': 100 * float(np.mean(plan_reach))
                    if plan_reach else None,
                'key_nearest_gt_0p9mm_query_sites': sum(
                    row['key_nearest_gt_0p9mm_query_sites'] for row in group),
                'section_equal_mean': section_means, 'plan_equal_mean': plan_means}
assert len(rows) == 982
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows)}), flush=True)
