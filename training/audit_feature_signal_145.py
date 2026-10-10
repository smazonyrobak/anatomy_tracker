"""Frozen 145 DEV diagnostic: do learned features distinguish matched-support anatomy?"""

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
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import dense_local_targets_143, render_coherent_atlas_143
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
train = root / 'runs/common_support_field_145'
dev = root / 'runs/common_support_field_145_dev_eval'
out = root / 'runs/common_support_field_145_feature_audit'
pose_path = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
steps, arms = (0, 500, 2000), ('treatment', 'control')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == train.drive.upper() == dev.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
dev_done = json.loads((dev / 'completed.json').read_text())
assert sha(train / 'completed.json') == dev_done['train_completion_sha256']
assert sha(dev / 'pair_bank.jsonl') == dev_done['pair_bank_sha256']
assert sha(dev / 'config.json') == dev_done['config_sha256']
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(pose_path) == train_config['pose_parent_checkpoint_sha256']
assert sha(source.parent / 'evaluate_common_support_field_145.py') == dev_done['evaluator_sha256']
checkpoint_sha = {f'{arm}_{step}': sha(train / f'{arm}_step_{step:05d}.pt')
                  for step in steps for arm in arms}
assert all(checkpoint_sha[f'{arm}_{step}'] == train_done['checkpoint_sha256'][arm][str(step)]
           for step in steps for arm in arms)
records, panel_sha = {}, {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    records.update({(cohort, record['section_id']): record for record in
                    (json.loads(line) for line in (panel / 'records.jsonl').open())
                    if record['eligible']})
    panel_sha[cohort] = sha(panel / 'completed.json')
pairs = [json.loads(line) for line in (dev / 'pair_bank.jsonl').open()]
assert len(pairs) == len(records) == 491
assert {(row['cohort'], row['section_id']) for row in pairs} == set(records)

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
pose.load_state_dict(torch.load(pose_path, map_location='cpu', weights_only=True)['model'])
fields = {}
for step in steps:
    for arm in arms:
        field = CoherentAnatomyField143(radius=6).cuda().eval().requires_grad_(False)
        field.load_state_dict(torch.load(train / f'{arm}_step_{step:05d}.pt',
                                         map_location='cpu', weights_only=True)['field'])
        fields[step, arm] = field
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device='cuda'),
                                  torch.arange(64, device='cuda'), indexing='ij')
provenance = {'diagnostic_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'dev_completion_sha256': sha(dev / 'completed.json'),
    'frozen_pair_bank_sha256': sha(dev / 'pair_bank.jsonl'),
    'train_source_sha256': train_config['source_sha256'],
    'eval_source_sha256': dev_done['evaluator_sha256'],
    'pose_checkpoint_sha256': sha(pose_path), 'field_checkpoint_sha256': checkpoint_sha,
    'panel_receipt_sha256': panel_sha, 'steps': steps, 'arms': arms,
    'site_rule': 'rank-eligible frozen pairs; identical 3x3 eroded valid source and 3x3x3 eroded all-depth common atlas support; exact CCF point reachable; same atlas grid for both keys; exclude wrong physical key <1mm from true CCF point',
    'metric': 'unit-normalized source dot exact atlas key minus source dot geometry-matched wrong-plane key; positive_gt_wrong is strict',
    'status': 'development feature diagnostic only; no pose/deformation/animal qualification or uncertainty calibration'}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(provenance, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for pair in pairs:
        cohort, section_id = pair['cohort'], pair['section_id']
        record = records[cohort, section_id]
        section = {'cohort': cohort, 'section_id': section_id,
            'physical_section_id': pair['physical_section_id'],
            'subject_plan_id': pair['subject_plan_id'], 'animal_id': pair['animal_id'],
            'specimen_id': pair['specimen_id'], 'experiment_id': pair['experiment_id'],
            'rank_eligible': pair['rank_eligible'],
            'wrong_rigid_mm': pair['wrong_rigid_mm'],
            'central_support_iou': pair['central_support_iou'],
            'central_common_support_fraction': pair['central_common_support_fraction'],
            'panel_file_sha256': record['sha256']}
        if not pair['rank_eligible']:
            section.update({'eroded_common_reachable_sites': 0,
                            'near_wrong_key_excluded_sites': 0, 'site_count': 0})
            for step in steps:
                for arm in arms:
                    row = {**section, 'step': step, 'arm': arm,
                           'positive_dot_mean': None, 'wrong_dot_mean': None,
                           'margin_mean': None, 'positive_gt_wrong_fraction': None}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
            continue
        panel = panels[cohort]
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            wrong = torch.tensor(pair['wrong_state'], device='cuda', dtype=state.dtype)[None]
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        feature = pose.predict(image)['feature']
        slabs, base, basis, normal = render_coherent_atlas_143(atlas,
            torch.cat((state, wrong)), reflection.expand(2), offsets.expand(2, -1),
            weights.expand(2, -1))
        common = ((slabs[:1, 1:2] > .5) & (slabs[1:2, 1:2] > .5)).float()
        atlas_pair = torch.cat((slabs[:, :1] * common.expand(2, -1, -1, -1, -1),
                                common.expand(2, -1, -1, -1, -1)), 1)
        local, _, reachable = dense_local_targets_143(truth, valid, base[:1],
                                                       basis[:1], normal[:1], radius=6)
        local = local[0]
        grid = torch.stack((2 * (pixel_x + local[..., 0] * 64 + .5) / 64 - 1,
                            2 * (pixel_y + local[..., 1] * 64 + .5) / 64 - 1,
                            2 * (3 + local[..., 2] / 500 + .5) / 7 - 1), -1)
        source_interior = F.avg_pool2d(valid[:, None, 2::4, 2::4].float(), 3, 1, 1)[0, 0] == 1
        common_interior = F.avg_pool3d(common, 3, 1, 1)
        supported_truth = F.grid_sample(common_interior, grid[None, None],
            mode='nearest', align_corners=False)[0, 0, 0] == 1
        locations = (reachable[0] & source_interior & supported_truth).nonzero()
        y, x = locations[:, 0], locations[:, 1]
        delta = local[y, x]
        truth_ccf = truth[0, 2::4, 2::4][y, x]
        wrong_ccf = (base[1, y, x] + delta[:, 0, None] * basis[1, :, 0] +
                     delta[:, 1, None] * basis[1, :, 1] + delta[:, 2, None] * normal[1])
        far = (wrong_ccf - truth_ccf).norm(dim=-1) >= 1000
        y, x = y[far], x[far]
        section['eroded_common_reachable_sites'] = len(locations)
        section['near_wrong_key_excluded_sites'] = int((~far).sum())
        section['site_count'] = len(y)
        if len(y):
            sample_grid = grid[y, x][None, None, None].expand(2, -1, -1, -1, -1)
            for step in steps:
                for arm in arms:
                    field = fields[step, arm]
                    with torch.autocast('cuda', dtype=torch.float16):
                        source_features = field.source_fine(F.interpolate(image, (128, 128),
                            mode='bilinear', align_corners=False))
                        source_features = source_features + field.shared_source(
                            F.interpolate(feature, (64, 64), mode='bilinear', align_corners=False))
                        source_features = F.normalize(source_features + F.interpolate(
                            field.source_context(source_features), (64, 64),
                            mode='bilinear', align_corners=False), dim=1)
                        atlas_features = F.normalize(field.atlas(atlas_pair), dim=1)
                        keys = F.grid_sample(atlas_features, sample_grid,
                                             mode='bilinear', align_corners=False)
                    query = F.normalize(source_features[0, :, y, x].T.float(), dim=1)
                    keys = F.normalize(keys[:, :, 0, 0].permute(0, 2, 1).float(), dim=-1)
                    positive = (query * keys[0]).sum(-1)
                    negative = (query * keys[1]).sum(-1)
                    margin = positive - negative
                    row = {**section, 'step': step, 'arm': arm,
                           'positive_dot_mean': float(positive.mean()),
                           'wrong_dot_mean': float(negative.mean()),
                           'margin_mean': float(margin.mean()),
                           'positive_gt_wrong_fraction': float((margin > 0).float().mean())}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
        else:
            for step in steps:
                for arm in arms:
                    row = {**section, 'step': step, 'arm': arm,
                           'positive_dot_mean': None, 'wrong_dot_mean': None,
                           'margin_mean': None, 'positive_gt_wrong_fraction': None}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)

summary = {}
for cohort in panels:
    summary[cohort] = {}
    for step in steps:
        summary[cohort][str(step)] = {}
        for arm in arms:
            group = [row for row in rows if row['cohort'] == cohort and row['step'] == step and
                     row['arm'] == arm and row['site_count'] > 0]
            plans = defaultdict(list)
            for row in group:
                plans[row['subject_plan_id']].append(row)
            summary[cohort][str(step)][arm] = {
                'rank_eligible_sections': sum(row['rank_eligible'] for row in rows
                    if row['cohort'] == cohort and row['step'] == step and row['arm'] == arm),
                'sections_with_sites': len(group), 'subject_plans_with_sites': len(plans),
                'sites': sum(row['site_count'] for row in group),
                'section_equal_mean_margin': float(np.mean([row['margin_mean'] for row in group]))
                    if group else None,
                'section_equal_positive_gt_wrong_percent': float(100 * np.mean([
                    row['positive_gt_wrong_fraction'] for row in group])) if group else None,
                'plan_equal_mean_margin': float(np.mean([np.mean([
                    row['margin_mean'] for row in plan]) for plan in plans.values()]))
                    if plans else None,
                'plan_equal_positive_gt_wrong_percent': float(100 * np.mean([np.mean([
                    row['positive_gt_wrong_fraction'] for row in plan]) for plan in plans.values()]))
                    if plans else None,
                'site_weighted_mean_margin': float(np.average([row['margin_mean'] for row in group],
                    weights=[row['site_count'] for row in group])) if group else None}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'diagnostic_sha256': sha(source),
    'rows': len(rows), 'rank_eligible_sections': sum(row['rank_eligible'] for row in pairs),
    'calibrated': False, 'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'feature_diagnostic_complete', 'summary': summary}), flush=True)
