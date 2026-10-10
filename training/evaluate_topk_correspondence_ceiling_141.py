"""Frozen top-four atlas-key geometry ceiling on the 140 near-branch DEV panel."""

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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_spatial_fit_139 import AtlasSpatialFit139
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/TOPK_CORRESPONDENCE_CEILING_141_PROTOCOL_20261010.md'
run = root / 'runs/atlas_correspondence_curriculum_140'
prior = root / 'runs/atlas_correspondence_curriculum_140_dev_eval'
out = root / 'runs/topk_correspondence_ceiling_141'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('full_intensity', 'support_only')
device = 'cuda'
side = 256
thresholds = (500, 750, 1500)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert source.drive.upper() == protocol.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
prior_done = json.loads((prior / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(run / 'completed.json') == prior_done['train_completion_sha256']
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(prior / 'config.json') == prior_done['config_sha256']
assert sha(prior / 'rows.jsonl') == prior_done['rows_sha256']
assert sha(prior / 'summary.json') == prior_done['summary_sha256']
assert sha(source / 'evaluate_atlas_correspondence_curriculum_140.py') == prior_done['evaluator_sha256']
assert all(sha(source / name) == digest for name, digest in train_done['source_sha256'].items())
assert not any(train_done.get(key, False) for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used', 'external_pretrained_weights_used'))

frozen = {}
for line in (prior / 'rows.jsonl').open():
    row = json.loads(line)
    if row['step'] != 4000 or row['role'] != 'blind_best_original':
        continue
    key = row['cohort'], row['section_id']
    if row['arm'] == arms[0]:
        assert key not in frozen
        frozen[key] = row
    else:
        previous = frozen[key]
        assert row['branch_id'] == previous['branch_id']
        assert row['original_rigid_mm'] == previous['original_rigid_mm']
assert len(frozen) == 491

panel_records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert len(records) == done['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    assert all(row['provenance']['split'] == 'development' for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
    assert all((cohort, row['section_id']) in frozen for row in panel_records[cohort])
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243
assert sum(frozen[('v4', r['section_id'])]['blind_best_available_1p5mm']
           for r in panel_records['v4']) == 179
assert sum(frozen[('v3', r['section_id'])]['blind_best_available_1p5mm']
           for r in panel_records['v3']) == 197

atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array, annotation
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
heads = {}
for arm in arms:
    path = run / arm / 'head_step_04000.pt'
    assert sha(path) == train_done['checkpoint_sha256'][arm]['4000']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['step'] == 4000 and saved['arm'] == arm
    assert saved['config'] == train_config
    heads[arm] = AtlasSpatialFit139().to(device).eval().requires_grad_(False)
    heads[arm].load_state_dict(saved['head'], strict=True)
    del saved

axis = torch.arange(side, device=device) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
fixed = heads[arms[0]].fixed_query_xy
fixed_x = (fixed[:, 0] * side).long()
fixed_y = (fixed[:, 1] * side).long()
assert torch.equal(fixed, heads[arms[1]].fixed_query_xy)

out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
          'parent_checkpoint_sha256': sha(parent_path),
          'train_completion_sha256': sha(run / 'completed.json'),
          'prior_evaluation_completion_sha256': sha(prior / 'completed.json'),
          'panel_receipt_sha256': {cohort: sha(panel / 'completed.json')
                                   for cohort, panel in panels.items()},
          'cohorts': {cohort: len(records) for cohort, records in panel_records.items()},
          'arms': arms, 'step': 4000, 'threshold_um': thresholds,
          'role': 'oracle truth-best original branch within unchanged blind16; only if original error <=1.5mm',
          'support': 'fractional atlas support >0.5; no label is passed to model',
          'key_lattice': '7x32x32, depth-major then row then column; top4 exclude dustbin',
          'local_offsets': '3x5x5 clamped as frozen head; duplicate keys allowed only in geometry ceiling',
          'gate': 'v4 full-intensity top4 minus top1 >=10 percentage points among globally <=500um sites and top4 >=60% of global ceiling',
          'calibrated': False, 'public_benchmark_used': False,
          'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        completed = 0
        for record in panel_records[cohort]:
            previous = frozen[(cohort, record['section_id'])]
            if not previous['blind_best_available_1p5mm']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                target_state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).to(device).bool()
                dense = torch.from_numpy(arrays['target_centre_um'].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
            prediction = model.predict(image)
            states = full_frame_state_from_components(
                *full_frame_state_to_components(prediction['state']))
            branch = previous['branch_id']
            candidate = states[:, branch // 2:branch // 2 + 1]
            candidate_reflection = torch.tensor([[branch % 2]], device=device)
            observed = chart[valid]
            target = rigid_points_090(target_state, reflection, observed)[0]
            original = rigid_points_090(candidate, candidate_reflection, observed)[0, 0]
            assert abs(float((original - target).norm(dim=-1).mean() / 1000)
                       - previous['original_rigid_mm']) < 1e-4
            truth = dense[fixed_y, fixed_x]
            valid_fixed = valid[fixed_y, fixed_x]
            truth = truth[valid_fixed].float()
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            angle_bin = ('<15' if angle < 15 else '15-30' if angle < 30 else
                         '30-45' if angle < 45 else '>=45')
            assert int(valid.sum()) == record['valid_pixels']
            for arm in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = heads[arm](prediction, image, candidate,
                        candidate_reflection, atlas, offsets, weights,
                        atlas_intensity_enabled=(arm == arms[0]))
                keys = output['coarse_key_world_um'][0, 0].float()
                support = output['coarse_key_support'][0, 0] > .5
                assert keys.shape == (7 * 32 * 32, 3)
                score = output['coarse_logits'][0, 0, valid_fixed].float()
                fine_score = output['fine_logits'][0, 0, valid_fixed].float()
                top = score[:, :-1].topk(4, -1).indices
                depth = top.div(32 * 32, rounding_mode='floor')
                row = top.div(32, rounding_mode='floor').remainder(32)
                column = top.remainder(32)
                lattice = torch.stack((depth, row, column), -1)[:, :, None]
                lattice = lattice + heads[arm].local_offsets[None, None]
                local = (lattice[..., 0].clamp(0, 6) * 32 * 32 +
                         lattice[..., 1].clamp(0, 31) * 32 +
                         lattice[..., 2].clamp(0, 31))
                distances = (keys[local] - truth[:, None, None]).norm(dim=-1)
                distances = distances.masked_fill(~support[local], float('inf'))
                top1_distance = distances[:, 0].amin(-1)
                top4_distance = distances.flatten(1).amin(-1)
                global_distance = torch.cdist(truth[None], keys[None])[0]
                global_distance[:, ~support] = float('inf')
                global_distance = global_distance.amin(-1)
                fine_index = fine_score.argmax(-1)
                fine_world = output['fine_key_world_um'][0, 0, valid_fixed].float()
                fine_support = output['fine_key_support'][0, 0, valid_fixed] > .5
                fine_position = fine_world[torch.arange(len(truth), device=device),
                                           fine_index.clamp_max(74)]
                fine_supported = fine_support[torch.arange(len(truth), device=device),
                                              fine_index.clamp_max(74)]
                scored_distance = (fine_position - truth).norm(dim=-1).masked_fill(
                    (fine_index == 75) | ~fine_supported, float('inf'))
                counts = {'valid_fixed_sites': len(truth),
                          'top4_supported_seed_count': int(support[top].sum()),
                          'top4_seed_count': 4 * len(truth)}
                for threshold in thresholds:
                    label = str(threshold)
                    counts[f'global_le_{label}_um'] = int((global_distance <= threshold).sum())
                    counts[f'top1_le_{label}_um'] = int((top1_distance <= threshold).sum())
                    counts[f'top4_le_{label}_um'] = int((top4_distance <= threshold).sum())
                    counts[f'scored_le_{label}_um'] = int((scored_distance <= threshold).sum())
                assert all(counts[f'top1_le_{t}_um'] <= counts[f'top4_le_{t}_um']
                           <= counts[f'global_le_{t}_um'] for t in thresholds)
                row_out = {'cohort': cohort, 'arm': arm,
                    'section_id': record['section_id'],
                    'physical_section_id': record['panel_physical_section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'animal_id': record['animal_id'],
                    'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'appearance_mode': record['appearance_mode'],
                    'nearest_cardinal_angle_deg': angle,
                    'nearest_cardinal_angle_bin_deg': angle_bin,
                    'observed_valid_tissue_fraction': record['valid_pixels'] / (side * side),
                    'branch_id': branch,
                    'original_rigid_mm': previous['original_rigid_mm'], **counts}
                stream.write(json.dumps(row_out, allow_nan=False) + '\n')
                rows.append(row_out)
            completed += 1
            if completed % 64 == 0:
                stream.flush()
                print(json.dumps({'event': 'cohort_milestone', 'cohort': cohort,
                                  'near_sections': completed}), flush=True)

summary = {}
for cohort in panels:
    for arm in arms:
        for angle_bin in ('all', '<15', '15-30', '30-45', '>=45'):
            group = [row for row in rows if row['cohort'] == cohort and row['arm'] == arm
                     and (angle_bin == 'all' or row['nearest_cardinal_angle_bin_deg'] == angle_bin)]
            total = {field: sum(row[field] for row in group) for field in group[0]
                     if field == 'valid_fixed_sites' or field.endswith('_count')
                     or field.endswith('_um')}
            denominator = total['global_le_500_um']
            gain = (100 * (total['top4_le_500_um'] - total['top1_le_500_um']) / denominator
                    if denominator else None)
            capture = 100 * total['top4_le_500_um'] / denominator if denominator else None
            summary[f'{cohort}:{arm}:{angle_bin}'] = {
                'sections': len(group), **total,
                'top1_over_global500_percent': (100 * total['top1_le_500_um'] / denominator
                                                if denominator else None),
                'top4_over_global500_percent': capture,
                'top4_minus_top1_global500_percentage_points': gain}
gate = summary['v4:full_intensity:all']
summary['decision'] = {'pass': gate['global_le_500_um'] > 0
                       and gate['top4_minus_top1_global500_percentage_points'] >= 10
                       and gate['top4_over_global500_percent'] >= 60,
                       'minimum_gain_percentage_points': 10,
                       'minimum_global_ceiling_capture_percent': 60,
                       'arm': 'full_intensity', 'cohort': 'v4'}
assert len(rows) == 2 * (179 + 197)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'evaluator_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows),
                  'geometry_gate_pass': summary['decision']['pass']}), flush=True)
