"""Frozen 094 match-coordinate versus unshifted-plane oracle diagnostic."""
import hashlib
import json
import math
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GEOMETRY_NULL_096_PROTOCOL_20261005.md'
parent = root / 'runs/spatial_verifier_094_pilot'
prior_eval = root / 'runs/spatial_verifier_094_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
ranking = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
diagnostic = root / 'runs/spatial_match_signal_093_diagnostic'
out = root / 'runs/geometry_null_096_diagnostic'
checkpoint = parent / 'joint_step_20000.pt'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rho(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if math.isfinite(value) else 0.


assert sha(protocol) == '771b3e776699348d9fa5b1c956d28ac5eaf7e3b38e03c8896fd224eef1b75610'
parent_receipt = json.loads((parent / 'completed.json').read_text())
assert sha(checkpoint) == parent_receipt['checkpoint_sha256']['20000'] \
    == '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
prior_receipt = json.loads((prior_eval / 'completed.json').read_text())
assert sha(prior_eval / 'completed.json') == \
    '24c27b5f93e9fbc08e9321ea77d6d246b4372381363c5d1be7057a8d4e793b28'
assert sha(prior_eval / 'config.json') == prior_receipt['config_sha256']
assert sha(prior_eval / 'candidates.jsonl') == prior_receipt['candidates_sha256']
assert sha(prior_eval / 'summary.json') == prior_receipt['summary_sha256']
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'completed.json') == \
    '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
rank_receipt = json.loads((ranking / 'completed.json').read_text())
assert sha(ranking / 'completed.json') == \
    '5eb6dc84b615f3ce01ee9ac84ce124057a42a6dda39dbc2b999c87c1e05b3711'
assert sha(ranking / 'candidates.jsonl') == rank_receipt['candidates_sha256']
diagnostic_receipt = json.loads((diagnostic / 'completed.json').read_text())
assert sha(diagnostic / 'completed.json') == \
    'e982a2f0b47d7a088ceb88eff5fe8505bcbec3df5e8d74e77bd726c64e3ddcca'
assert sha(diagnostic / 'config.json') == diagnostic_receipt['config_sha256']
inspected = set(json.loads((diagnostic / 'config.json').read_text())['section_ids'])
all_records = [row for row in map(json.loads, (panel / 'records.jsonl').open())
               if row['eligible']]
records = [row for row in all_records if row['section_id'] not in inspected]
assert len(all_records) == 246 and len(inspected) == 64 and len(records) == 182
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8

frozen = defaultdict(list)
for row in map(json.loads, (prior_eval / 'candidates.jsonl').open()):
    if row['step'] == 20000 and row['in_fixed_beam'] and row['independent_head_readout']:
        frozen[row['section_id']].append(row)
ranked = defaultdict(dict)
for row in map(json.loads, (ranking / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row['branch_id']
assert set(frozen) == {row['section_id'] for row in records}
for record in records:
    section = record['section_id']
    frozen[section].sort(key=lambda row: row['beam_slot'])
    assert len(frozen[section]) == 14
    assert [row['beam_slot'] for row in frozen[section]] == list(range(14))
    assert [row['branch_id'] for row in frozen[section]] == [ranked[section][i]
                                                             for i in range(14)]
    assert all(row['source_sha256'] == record['sha256'] and row['step'] == 20000
               for row in frozen[section])

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and not saved['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
spatial = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
spatial.load_state_dict(saved['spatial_fit'], strict=True)
del saved

axis = (torch.arange(32, device='cuda') + .5) / 32 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
candidate_rows, section_rows = [], []
max_reproduction_difference = 0.
with torch.inference_mode():
    for record in records:
        section = record['section_id']
        rows = frozen[section]
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        visible = F.interpolate(valid[:, None].float(), (32, 32),
            mode='bilinear', align_corners=False)[:, 0] == 1
        truth32 = F.interpolate(truth.permute(0, 3, 1, 2), (32, 32),
            mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        denominator = visible.sum((-2, -1))[:, None].clamp_min(1)
        prediction = model.predict(image)
        branch = torch.tensor([[row['branch_id'] for row in rows]], device='cuda')
        initial = prediction['state'].gather(1,
            (branch // 2)[..., None].expand(-1, -1, 12))
        actual, null, available_sites = [], [], []
        for start in range(0, 14, 2):
            pair = branch[:, start:start + 2]
            for local, old in enumerate(rows[start:start + 2]):
                assert max(abs(a - b) for a, b in zip(
                    initial[0, start + local].tolist(), old['initial_state'])) < 1e-3
            state = torch.tensor([old['fitted_state'] for old in rows[start:start + 2]],
                                 device='cuda', dtype=initial.dtype)[None]
            matched = spatial.matcher(prediction['feature'], state, pair % 2, atlas,
                offsets, weights, source_shape=(256, 256), match_only=True)
            # Only the coordinate/support outputs are used; this evidence input cannot alter them.
            match_ccf, _, _, available = spatial._verify('fine',
                matched['fine_match_logits'], matched['fine_match_support'],
                prediction['feature'], state.new_zeros(1, 2, 64, 32, 32),
                state, pair % 2, (256, 256))
            available = available >= .5
            support_fraction = available.float().mean((-2, -1))[0].tolist()
            assert all(abs(value - old['atlas_support_fraction']) < 1e-5
                       for value, old in zip(support_fraction, rows[start:start + 2]))
            centre, frame, basis = full_frame_state_to_components(state)
            edges = frame[..., :, :2] @ basis
            chart_x = torch.where((pair % 2).bool()[..., None, None],
                                  255 / 256 - xx, xx)
            chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
            centre_ccf = centre[..., None, None, :] + torch.einsum(
                'bkij,bkhwj->bkhwi', edges, chart - .5)
            match_correct = visible[:, None] & available & (
                (match_ccf - truth32[:, None]).norm(dim=-1) <= 1500)
            null_correct = visible[:, None] & available & (
                (centre_ccf - truth32[:, None]).norm(dim=-1) <= 1500)
            available_sites.extend((available & visible[:, None]).sum((-2, -1)).flatten().tolist())
            actual.extend((match_correct.sum((-2, -1)) / denominator).flatten().tolist())
            null.extend((null_correct.sum((-2, -1)) / denominator).flatten().tolist())
        error = np.array([row['mapped96_error_um'] for row in rows])
        actual, null = np.array(actual), np.array(null)
        difference = actual - null
        reproduction = np.abs(actual - np.array([
            row['actual_correct_fraction'] for row in rows]))
        max_reproduction_difference = max(max_reproduction_difference, float(reproduction.max()))
        assert reproduction.max() < 1e-4
        for slot, old in enumerate(rows):
            candidate_rows.append({'section_id': section,
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'source_sha256': record['sha256'], 'beam_slot': slot,
                'branch_id': old['branch_id'], 'mapped96_error_um': float(error[slot]),
                'visible_sites': int(denominator[0, 0]),
                'atlas_available_visible_sites': int(available_sites[slot]),
                'actual_fraction': float(actual[slot]), 'null_fraction': float(null[slot]),
                'actual_minus_null': float(difference[slot])})
        best, worst = int(error.argmin()), int(error.argmax())
        section_rows.append({'section_id': section,
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'actual_error_rho': rho(actual, -error),
            'null_error_rho': rho(null, -error),
            'incremental_fraction_error_rho': rho(difference, -error),
            'best_actual_fraction': float(actual[best]),
            'best_null_fraction': float(null[best]),
            'worst_actual_fraction': float(actual[worst]),
            'worst_null_fraction': float(null[worst]),
            'mean_actual_minus_null': float(difference.mean()),
            'mean_abs_actual_minus_null': float(np.abs(difference).mean())})

metrics = [key for key in section_rows[0] if key not in
           ('section_id', 'synthetic_subject_plan_id')]
for row in section_rows:
    row['actual_minus_null_error_rho'] = row['actual_error_rho'] - row['null_error_rho']
metrics.append('actual_minus_null_error_rho')
plans = sorted({row['synthetic_subject_plan_id'] for row in section_rows})
by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan
    for row in section_rows), **{metric: float(np.mean([row[metric] for row in section_rows
    if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}}
    for plan in plans}
summary = {'sections': len(section_rows), 'candidates': len(candidate_rows),
    'max_actual_fraction_reproduction_difference': max_reproduction_difference,
    'by_plan': by_plan,
    'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric] for plan in plans]))
                        for metric in metrics}}
prior_summary = json.loads((prior_eval / 'summary.json').read_text())
expected_rho = prior_summary['20000']['independent_candidates']['plan_equal_mean'][
    'independent_actual_inlier_error_spearman']
assert abs(summary['plan_equal_mean']['actual_error_rho'] - expected_rho) < 1e-6
assert len(candidate_rows) == 182 * 14 and len(section_rows) == 182

config = {'protocol_sha256': sha(protocol), 'source_sha256': {name: sha(source / name)
    for name in ('diagnose_geometry_null_096.py', 'evaluate_spatial_verifier_094.py',
                 'spatial_joint_fit_094.py', 'whole_slice_atlas_feedback_083.py',
                 'whole_slice_atlas_feedback_081.py',
                 'arbitrary_plane_full_frame_primitives.py',
                 'arbitrary_plane_one_shot_model.py',
                 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_checkpoint_sha256': sha(checkpoint),
    'parent_completed_sha256': sha(parent / 'completed.json'),
    'prior_eval_completed_sha256': sha(prior_eval / 'completed.json'),
    'prior_eval_candidates_sha256': sha(prior_eval / 'candidates.jsonl'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_candidates_sha256': sha(ranking / 'candidates.jsonl'),
    'diagnostic_config_sha256': sha(diagnostic / 'config.json'),
    'sections': 182, 'fixed_branches_per_section': 14,
    'visible_denominator': 'bilinear valid_mask at 32x32 equals 1',
    'atlas_available': 'any fine match support >= 0.5',
    'correct_tolerance_um': 1500,
    'null': '094 fitted-state plane coordinate with dx=dy=0 and central depth'}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(section_rows), 'candidates': len(candidate_rows),
    'protocol_sha256': sha(protocol), 'parent_checkpoint_sha256': sha(checkpoint),
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json')}, indent=2))
print(json.dumps({'event': 'complete', **summary['plan_equal_mean']}, allow_nan=False))
