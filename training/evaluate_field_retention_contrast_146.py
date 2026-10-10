"""Independent frozen-DEV retention and CCF-feature readout for field 146."""

import copy
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
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import dense_local_targets_143, render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right))
    return left == right


source = Path(__file__).resolve()
protocol = source.parent.parent / 'docs/publication/RETENTION_FIELD_146_PROTOCOL_20261010.md'
train = root / 'runs/field_retention_contrast_146'
parent = root / 'runs/coherent_anatomy_field_143'
dev = root / 'runs/common_support_field_145_dev_eval'
eval132 = root / 'runs/v4_pose_adaptation_132_dev_eval'
eval143 = root / 'runs/coherent_anatomy_field_143_dev_eval'
pose_path = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
out = root / 'runs/field_retention_contrast_146_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms, steps = ('treatment', 'control'), (0, 500, 2000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == protocol.drive.upper() == train.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
done = json.loads((train / 'completed.json').read_text())
config = json.loads((train / 'config.json').read_text())
parent_done = json.loads((parent / 'completed.json').read_text())
parent_config = json.loads((parent / 'config.json').read_text())
dev_done = json.loads((dev / 'completed.json').read_text())
assert sha(train / 'config.json') == done['config_sha256']
assert sha(train / 'draws.jsonl') == done['draws_sha256']
assert sha(train / 'training.jsonl') == done['training_sha256']
assert sha(protocol) == config['protocol_sha256']
assert sha(parent / 'config.json') == parent_done['config_sha256']
assert sha(parent / 'full_step_10000.pt') == parent_done['checkpoint_sha256']['full']['10000'] == config['parent_checkpoint_sha256']
assert sha(parent / 'completed.json') == config['parent_completion_sha256']
assert sha(pose_path) == config['pose_parent_checkpoint_sha256']
assert sha(dev / 'pair_bank.jsonl') == dev_done['pair_bank_sha256']
assert sha(dev / 'config.json') == dev_done['config_sha256']
assert sha(source.parent / 'evaluate_common_support_field_145.py') == dev_done['evaluator_sha256']
assert sha(root / 'runs/common_support_field_145/completed.json') == dev_done['train_completion_sha256']
done132eval = json.loads((eval132 / 'completed.json').read_text())
done143eval = json.loads((eval143 / 'completed.json').read_text())
assert all(sha(eval132 / name) == digest for name, digest in done132eval['output_sha256'].items())
assert sha(eval143 / 'rows.jsonl') == done143eval['rows_sha256']
assert sha(eval143 / 'summary.json') == done143eval['summary_sha256']
assert sha(source.parent / 'evaluate_coherent_anatomy_field_143.py') == done143eval['evaluator_sha256']
assert config['source_sha256'] == done['source_sha256']
assert all(sha(source.parent / name) == digest for name, digest in config['source_sha256'].items())
assert sha(source.parent / 'train_coherent_anatomy_field_143.py') == parent_config['source_sha256']['train_coherent_anatomy_field_143.py']
assert config['checkpoints'] == list(steps) and config['steps'] == done['steps'] == 2000
assert tuple(config['arms']) == arms and config['presentations'] == 6000
assert done['accepted_synthetic_physical_sections'] == 6000
assert done['v4_presentations'] == 4000 and done['v3_presentations'] == 2000
assert all(sha(train / f'{arm}_step_{step:05d}.pt') == done['checkpoint_sha256'][arm][str(step)]
           for arm in arms for step in steps)
assert not any(done.get(key, False) for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used', 'external_pretrained_weights_used'))

pairs = [json.loads(line) for line in (dev / 'pair_bank.jsonl').open()]
records = {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    group = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert len(group) == receipt['physical_sections'] == 256
    records.update({(cohort, row['section_id']): row for row in group if row['eligible']})
    assert sum(row['eligible'] for row in group) == receipt['eligible'] == (248 if cohort == 'v4' else 243)
assert len(pairs) == len(records) == 491
assert {(row['cohort'], row['section_id']) for row in pairs} == set(records)
assert all(row['physical_section_id'] == records[row['cohort'], row['section_id']]['panel_physical_section_id']
           for row in pairs)
train_plans = {row['plan_receipt'] for row in config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for row in records.values()}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans

rows132 = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (eval132 / 'synthetic_rows.jsonl').open()) if row['arm'] == '2000'}
rows143 = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (eval143 / 'rows.jsonl').open())
    if row['step'] == 10000 and row['arm'] == 'full' and row['role'] == 'blind_near'}
assert len(rows132) == len(rows143) == 491

baseline = torch.load(parent / 'full_step_10000.pt', map_location='cpu', weights_only=True)
expected_optimizer = copy.deepcopy(baseline['optimizer'])
for group in expected_optimizer['param_groups']:
    group['lr'] = 5e-5
fields = {}
for step in steps:
    for arm in arms:
        path = train / f'{arm}_step_{step:05d}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step and saved['arm'] == arm and saved['config'] == config
        if step == 0:
            assert same(saved['field'], baseline['field'])
            assert same(saved['optimizer'], expected_optimizer)
            assert same(saved['scaler'], baseline['scaler'])
        field = CoherentAnatomyField143(radius=6).cuda().eval().requires_grad_(False)
        field.load_state_dict(saved['field'], strict=True)
        fields[step, arm] = field
        del saved
del baseline, expected_optimizer
assert same(fields[0, 'treatment'].state_dict(), fields[0, 'control'].state_dict())

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
pose.load_state_dict(torch.load(pose_path, map_location='cpu', weights_only=True)['model'])
axis256 = torch.arange(256, device='cuda') / 256
yy256, xx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((xx256, yy256), -1)
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device='cuda'),
                                  torch.arange(64, device='cuda'), indexing='ij')


def mapping(output, slot, truth, valid, slabs, base, basis, normal):
    valid64 = valid[0, 2::4, 2::4]
    truth64 = truth[0, 2::4, 2::4].float()
    local, _, reachable = dense_local_targets_143(truth, valid,
        base[slot:slot + 1], basis[slot:slot + 1], normal[slot:slot + 1], radius=6)
    x = pixel_x + 64 * local[0, ..., 0]
    y = pixel_y + 64 * local[0, ..., 1]
    z = 3 + local[0, ..., 2] / 500
    inside = (x >= 0) & (x <= 63) & (y >= 0) & (y <= 63) & (z >= 0) & (z <= 6)
    grid = torch.stack((2 * x / 63 - 1, 2 * y / 63 - 1, z / 3 - 1), -1)[None, None]
    support = F.grid_sample(slabs[slot:slot + 1, 1:2].float(), grid,
        align_corners=True)[0, 0, 0] > .5
    global_support = valid64 & inside & support
    estimate = output['offset_chart'][slot].permute(1, 2, 0).float()
    predicted = (base[slot] + torch.einsum('ij,hwj->hwi', basis[slot], estimate[..., :2]) +
                 normal[slot, None, None] * estimate[..., 2:3])
    distance = (predicted - truth64).norm(dim=-1)
    matched = output['match_logits'][slot].argmax(0) != output['match_logits'].shape[1] - 1
    return {'valid_sites': int(valid64.sum()), 'global_supported_sites': int(global_support.sum()),
        'locally_reachable_sites': int((global_support & reachable[0]).sum()),
        'nondustbin_valid_sites': int((valid64 & matched).sum()),
        'nondustbin_global_sites': int((global_support & matched).sum()),
        'scored_le_500_um_valid_sites': int((valid64 & matched & (distance <= 500)).sum()),
        'scored_le_500_um_global_sites': int((global_support & matched & (distance <= 500)).sum()),
        'point_error_valid_mm_sum': float(distance[valid64].sum() / 1000),
        'point_error_global_mm_sum': float(distance[global_support].sum() / 1000)}


def mapping_summary(group):
    keys = ('valid_sites', 'global_supported_sites', 'locally_reachable_sites',
        'nondustbin_valid_sites', 'nondustbin_global_sites',
        'scored_le_500_um_valid_sites', 'scored_le_500_um_global_sites',
        'point_error_valid_mm_sum', 'point_error_global_mm_sum')
    sums = {key: sum(row[key] for row in group) for key in keys}
    valid, global_sites = sums['valid_sites'], sums['global_supported_sites']
    return {'sections': len(group), 'plans': len({row['subject_plan_id'] for row in group}),
        **sums,
        'global_support_coverage_percent': 100 * global_sites / valid if valid else None,
        'hit_global_percent': 100 * sums['scored_le_500_um_global_sites'] / global_sites
            if global_sites else None,
        'hit_valid_percent': 100 * sums['scored_le_500_um_valid_sites'] / valid if valid else None,
        'nondustbin_coverage_global_percent': 100 * sums['nondustbin_global_sites'] / global_sites
            if global_sites else None,
        'nondustbin_coverage_valid_percent': 100 * sums['nondustbin_valid_sites'] / valid
            if valid else None,
        'point_error_valid_mm_mean': sums['point_error_valid_mm_sum'] / valid if valid else None,
        'point_error_global_mm_mean': sums['point_error_global_mm_sum'] / global_sites
            if global_sites else None}


readout = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(source),
    'train_completion_sha256': sha(train / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent / 'full_step_10000.pt'),
    'frozen_145_pair_bank_sha256': sha(dev / 'pair_bank.jsonl'),
    'frozen_145_completion_sha256': sha(dev / 'completed.json'),
    'panel_receipt_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'eval132_completion_sha256': sha(eval132 / 'completed.json'),
    'eval143_completion_sha256': sha(eval143 / 'completed.json'),
    'checkpoint_sha256': {arm: {str(step): sha(train / f'{arm}_step_{step:05d}.pt')
        for step in steps} for arm in arms},
    'arms': arms, 'steps': steps, 'cohorts': {cohort: 248 if cohort == 'v4' else 243
        for cohort in panels},
    'mapping': 'normal full-intensity atlas slabs; non-dustbin CCF error<=500um / identical globally supported sites',
    'features': 'frozen 145 rank-eligible pair bank; exact/wrong all-depth common support; 3x3 source and 3x3x3 atlas erosion; exact CCF reachable; exclude wrong key <1mm; section-equal positive-dot > wrong-dot fraction',
    'blind_near': 'frozen truth-best pose beam branch; not deployable prediction',
    'status': 'synthetic development only; no biological qualification, uncertainty calibration, or public benchmark'}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(readout, indent=2))
mapping_rows, feature_rows = [], []
with torch.inference_mode(), (out / 'mapping_rows.jsonl').open('w') as map_stream, \
        (out / 'feature_rows.jsonl').open('w') as feat_stream:
    for index, pair in enumerate(pairs):
        cohort, section_id = pair['cohort'], pair['section_id']
        record = records[cohort, section_id]
        panel = panels[cohort]
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        predicted = pose.predict(image)
        frozen = rows143[cohort, section_id]
        parent_row = rows132[cohort, section_id]
        branch = pair['blind_near_branch_id']
        assert branch == frozen['blind_near_branch_id'] and branch in parent_row['beam_branch_ids']
        observed = chart256[valid[0]]
        true_rigid = rigid_points_090(state, reflection, observed)[0]
        beam_ids = torch.tensor(parent_row['beam_branch_ids'], device='cuda')
        beam_world = rigid_points_090(predicted['state'][:, beam_ids // 2],
            (beam_ids % 2)[None], observed)[0]
        beam_error = (beam_world - true_rigid).norm(dim=-1).mean(-1) / 1000
        assert int(beam_ids[beam_error.argmin()]) == branch
        near_state = predicted['state'][:, branch // 2]
        near_reflection = reflection.new_full((1,), branch % 2)
        near_mm = float((rigid_points_090(near_state, near_reflection, observed)[0] -
                         true_rigid).norm(dim=-1).mean() / 1000)
        assert abs(near_mm - pair['blind_near_rigid_mm']) < 1e-4
        slab, base, basis, normal = render_coherent_atlas_143(atlas,
            torch.cat((state, near_state)), torch.cat((reflection, near_reflection)),
            offsets.expand(2, -1), weights.expand(2, -1))
        feature = predicted['feature']
        common_row = {'cohort': cohort, 'section_id': section_id,
            'physical_section_id': pair['physical_section_id'],
            'subject_plan_id': pair['subject_plan_id'], 'animal_id': pair['animal_id'],
            'specimen_id': pair['specimen_id'], 'experiment_id': pair['experiment_id'],
            'appearance_mode': pair['appearance_mode'],
            'nearest_cardinal_angle_deg': pair['nearest_cardinal_angle_deg'],
            'blind_near_branch_id': branch, 'blind_near_rigid_mm': near_mm,
            'blind_near_available_1p5mm': pair['blind_near_available_1p5mm'],
            'rank_eligible': pair['rank_eligible'], 'panel_file_sha256': record['sha256']}

        feature_section = {**common_row, 'wrong_rigid_mm': pair['wrong_rigid_mm'],
            'central_support_iou': pair['central_support_iou'],
            'central_common_support_fraction': pair['central_common_support_fraction'],
            'eroded_common_reachable_sites': 0, 'near_wrong_key_excluded_sites': 0,
            'site_count': 0}
        if pair['rank_eligible']:
            wrong = torch.tensor(pair['wrong_state'], device='cuda', dtype=state.dtype)[None]
            pair_slab, pair_base, pair_basis, pair_normal = render_coherent_atlas_143(atlas,
                torch.cat((state, wrong)), reflection.expand(2), offsets.expand(2, -1),
                weights.expand(2, -1))
            common = ((pair_slab[:1, 1:2] > .5) & (pair_slab[1:2, 1:2] > .5)).float()
            atlas_pair = torch.cat((pair_slab[:, :1] * common.expand(2, -1, -1, -1, -1),
                                    common.expand(2, -1, -1, -1, -1)), 1)
            local, _, reachable = dense_local_targets_143(truth, valid, pair_base[:1],
                pair_basis[:1], pair_normal[:1], radius=6)
            local = local[0]
            grid = torch.stack((2 * (pixel_x + local[..., 0] * 64 + .5) / 64 - 1,
                                2 * (pixel_y + local[..., 1] * 64 + .5) / 64 - 1,
                                2 * (3 + local[..., 2] / 500 + .5) / 7 - 1), -1)
            source_interior = F.avg_pool2d(valid[:, None, 2::4, 2::4].float(), 3, 1, 1)[0, 0] == 1
            common_interior = F.avg_pool3d(common, 3, 1, 1)
            supported = F.grid_sample(common_interior, grid[None, None],
                mode='nearest', align_corners=False)[0, 0, 0] == 1
            locations = (reachable[0] & source_interior & supported).nonzero()
            y, x = locations[:, 0], locations[:, 1]
            delta = local[y, x]
            truth_ccf = truth[0, 2::4, 2::4][y, x]
            wrong_ccf = (pair_base[1, y, x] + delta[:, 0, None] * pair_basis[1, :, 0] +
                         delta[:, 1, None] * pair_basis[1, :, 1] +
                         delta[:, 2, None] * pair_normal[1])
            far = (wrong_ccf - truth_ccf).norm(dim=-1) >= 1000
            feature_section['eroded_common_reachable_sites'] = len(locations)
            feature_section['near_wrong_key_excluded_sites'] = int((~far).sum())
            y, x = y[far], x[far]
            feature_section['site_count'] = len(y)
            if len(y):
                sample_grid = grid[y, x][None, None, None].expand(2, -1, -1, -1, -1)
        for step in steps:
            for arm in arms:
                field = fields[step, arm]
                with torch.autocast('cuda', dtype=torch.float16):
                    output = field(image.expand(2, -1, -1, -1), slab, 500.,
                                   source_feature=feature.expand(2, -1, -1, -1))
                for role, slot in (('exact', 0), ('blind_near', 1)):
                    row = {**common_row, 'step': step, 'arm': arm, 'role': role,
                        'fit_energy': float(output['energy'][slot]),
                        **mapping(output, slot, truth, valid, slab, base, basis, normal)}
                    map_stream.write(json.dumps(row, allow_nan=False) + '\n')
                    mapping_rows.append(row)
                if feature_section['site_count']:
                    with torch.autocast('cuda', dtype=torch.float16):
                        image_features = field.source_fine(F.interpolate(image, (128, 128),
                            mode='bilinear', align_corners=False))
                        image_features = image_features + field.shared_source(
                            F.interpolate(feature, (64, 64), mode='bilinear', align_corners=False))
                        image_features = F.normalize(image_features + F.interpolate(
                            field.source_context(image_features), (64, 64),
                            mode='bilinear', align_corners=False), dim=1)
                        atlas_features = F.normalize(field.atlas(atlas_pair), dim=1)
                        keys = F.grid_sample(atlas_features, sample_grid,
                                             mode='bilinear', align_corners=False)
                    query = F.normalize(image_features[0, :, y, x].T.float(), dim=1)
                    keys = F.normalize(keys[:, :, 0, 0].permute(0, 2, 1).float(), dim=-1)
                    positive = (query * keys[0]).sum(-1)
                    wrong_score = (query * keys[1]).sum(-1)
                    margin = positive - wrong_score
                    feature_row = {**feature_section, 'step': step, 'arm': arm,
                        'positive_dot_mean': float(positive.mean()),
                        'wrong_dot_mean': float(wrong_score.mean()),
                        'margin_mean': float(margin.mean()),
                        'positive_gt_wrong_fraction': float((margin > 0).float().mean())}
                else:
                    feature_row = {**feature_section, 'step': step, 'arm': arm,
                        'positive_dot_mean': None, 'wrong_dot_mean': None,
                        'margin_mean': None, 'positive_gt_wrong_fraction': None}
                feat_stream.write(json.dumps(feature_row, allow_nan=False) + '\n')
                feature_rows.append(feature_row)
        if (index + 1) % 128 == 0:
            map_stream.flush()
            feat_stream.flush()
            print(json.dumps({'event': 'development_milestone', 'sections': index + 1}), flush=True)

summary = {'mapping': {}, 'features': {},
    'warning': 'Frozen synthetic DEV and truth-best near branch only; no independent animal or deployment qualification.'}
for cohort in panels:
    summary['mapping'][cohort], summary['features'][cohort] = {}, {}
    for step in steps:
        summary['mapping'][cohort][str(step)], summary['features'][cohort][str(step)] = {}, {}
        for arm in arms:
            summary['mapping'][cohort][str(step)][arm] = {}
            for role in ('exact', 'blind_near'):
                group = [row for row in mapping_rows if row['cohort'] == cohort and
                         row['step'] == step and row['arm'] == arm and row['role'] == role]
                scopes = {'all': group}
                if role == 'blind_near':
                    scopes['near_le_1p5'] = [row for row in group if row['blind_near_available_1p5mm']]
                summary['mapping'][cohort][str(step)][arm][role] = {
                    scope: mapping_summary(items) for scope, items in scopes.items()}
            group = [row for row in feature_rows if row['cohort'] == cohort and
                     row['step'] == step and row['arm'] == arm]
            with_sites = [row for row in group if row['site_count']]
            summary['features'][cohort][str(step)][arm] = {
                'sections': len(group), 'rank_eligible_sections': sum(row['rank_eligible'] for row in group),
                'sections_with_sites': len(with_sites),
                'plans_with_sites': len({row['subject_plan_id'] for row in with_sites}),
                'sites': sum(row['site_count'] for row in with_sites),
                'section_equal_margin_mean': float(np.mean([row['margin_mean'] for row in with_sites]))
                    if with_sites else None,
                'section_equal_positive_gt_wrong_percent': float(100 * np.mean([
                    row['positive_gt_wrong_fraction'] for row in with_sites])) if with_sites else None}

conditions, deltas = {}, {}
for cohort in panels:
    for role, scope in (('exact', 'all'), ('blind_near', 'near_le_1p5')):
        parent_hit = summary['mapping'][cohort]['0']['control'][role][scope]['hit_global_percent']
        treatment_hit = summary['mapping'][cohort]['2000']['treatment'][role][scope]['hit_global_percent']
        key = f'{cohort}_{role}_parent_retention_within_2pp'
        conditions[key] = parent_hit is not None and treatment_hit is not None and treatment_hit >= parent_hit - 2
        deltas[key] = treatment_hit - parent_hit if parent_hit is not None and treatment_hit is not None else None
v4_treat = summary['features']['v4']['2000']['treatment']['section_equal_positive_gt_wrong_percent']
v4_control = summary['features']['v4']['2000']['control']['section_equal_positive_gt_wrong_percent']
v3_treat = summary['features']['v3']['2000']['treatment']['section_equal_positive_gt_wrong_percent']
v3_control = summary['features']['v3']['2000']['control']['section_equal_positive_gt_wrong_percent']
conditions['v4_feature_gain_at_least_5pp'] = v4_treat is not None and v4_control is not None and v4_treat >= v4_control + 5
conditions['v3_feature_loss_at_most_2pp'] = v3_treat is not None and v3_control is not None and v3_treat >= v3_control - 2
deltas['v4_feature_treatment_minus_control_pp'] = v4_treat - v4_control if v4_treat is not None and v4_control is not None else None
deltas['v3_feature_treatment_minus_control_pp'] = v3_treat - v3_control if v3_treat is not None and v3_control is not None else None
summary['decision'] = {'status': 'pass' if all(conditions.values()) else 'fail',
    'pass': all(conditions.values()), 'conditions': conditions, 'deltas_pp': deltas,
    'warning': 'A pass is only a synthetic DEV signal and does not establish blind pose choice or calibrated probabilities.'}
assert len(mapping_rows) == 491 * len(steps) * len(arms) * 2
assert len(feature_rows) == 491 * len(steps) * len(arms)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'),
    'mapping_rows_sha256': sha(out / 'mapping_rows.jsonl'),
    'feature_rows_sha256': sha(out / 'feature_rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'evaluator_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'frozen_145_pair_bank_sha256': sha(dev / 'pair_bank.jsonl'),
    'mapping_rows': len(mapping_rows), 'feature_rows': len(feature_rows),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'decision': summary['decision']['status']}), flush=True)
