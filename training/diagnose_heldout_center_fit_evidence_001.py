"""Frozen 103 DEV candidates: does centered, held-out atlas agreement rank planes?"""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.contextual_matcher_099 import WholeSliceAtlasFeedback099
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve()
protocol = source.parent.parent / 'docs/publication/HELDOUT_CENTER_FIT_EVIDENCE_001_PROTOCOL_20261009.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
descriptor_run = root / 'runs/contextual_match_099_pilot'
development = root / 'runs/cross_candidate_match_103_fresh_development_eval'
panel = root / 'data/cross_candidate_match_103_fresh_dev_panel_001'
out = root / 'runs/heldout_center_fit_evidence_001'
parent_file = parent_run / 'joint_step_20000.pt'
descriptor_file = descriptor_run / 'contextual_step_04000.pt'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
descriptor_receipt = json.loads((descriptor_run / 'completed.json').read_text())
development_receipt = json.loads((development / 'completed.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(parent_file) == parent_receipt['checkpoint_sha256']['20000']
assert sha(descriptor_file) == descriptor_receipt['checkpoint_sha256']['4000']
assert sha(development / 'candidates.jsonl') == development_receipt['candidates_sha256']
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open() if line.strip()]
records = [row for row in records if row['eligible']]
candidate_rows = [json.loads(line) for line in (development / 'candidates.jsonl').open()]
candidate_rows = [row for row in candidate_rows if row['step'] == 0]
assert len(records) == panel_receipt['eligible']
assert len(candidate_rows) == 14 * len(records)
by_section = {row['section_id']: [] for row in records}
for row in candidate_rows:
    by_section[row['section_id']].append(row)
assert all([row['beam_slot'] for row in sorted(by_section[key], key=lambda row: row['beam_slot'])]
           == list(range(14)) for key in by_section)

parent = torch.load(parent_file, map_location='cpu', weights_only=True)
descriptor = torch.load(descriptor_file, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(parent['model'], strict=True)
fitter = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
fitter.load_state_dict(parent['spatial_fit'], strict=True)
matcher = WholeSliceAtlasFeedback099().cuda().eval().requires_grad_(False)
matcher.load_state_dict(descriptor['matcher'], strict=True)
del parent, descriptor
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
axis = torch.arange(32, device='cuda')
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
heldout = ((yy // 2 + xx // 2) % 2) == 1
result_rows = []
with torch.inference_mode():
    for index, record in enumerate(records, 1):
        candidates = sorted(by_section[record['section_id']], key=lambda row: row['beam_slot'])
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        tissue = fitter.validity_head(F.adaptive_avg_pool2d(
            prediction['feature'], (32, 32))).sigmoid()[0, 0]
        centre_cosine, centre_support = [], []
        for start in range(0, 14, 2):
            pair = candidates[start:start + 2]
            state = torch.tensor([row['fitted_state'] for row in pair],
                device='cuda', dtype=image.dtype)[None]
            reflected = torch.tensor([row['branch_id'] % 2 for row in pair],
                device='cuda')[None]
            match = matcher(prediction['feature'], state, reflected, atlas,
                offsets, weights, source_shape=(256, 256), match_only=True)
            support = match['fine_match_support'][0, :, 112]
            cosine = (match['fine_match_logits'][0, :, 112] + 4 * (1 - support)) / \
                matcher.log_temperature.exp().clamp(2, 20)
            centre_cosine.append(cosine)
            centre_support.append(support >= .5)
        centre_cosine = torch.cat(centre_cosine)
        centre_support = torch.cat(centre_support)
        common = centre_support.all(0) & (tissue >= .5) & heldout
        common_sites = int(common.sum())
        site_weight = tissue * common
        block_weight = site_weight.reshape(8, 4, 8, 4).sum((1, 3))
        occupied = block_weight > 0
        block_sum = (centre_cosine * site_weight).reshape(14, 8, 4, 8, 4).sum((2, 4))
        centered = ((block_sum / block_weight.clamp_min(1e-6)) * occupied).sum((1, 2)) / \
            occupied.sum().clamp_min(1)
        tissue_weight = tissue * (tissue >= .5) * heldout
        support_score = (centre_support * tissue_weight).sum((1, 2)) / tissue_weight.sum().clamp_min(1)
        centered = centered.cpu().numpy()
        support_score = support_score.cpu().numpy()
        errors = np.array([row['mapped96_error_um'] for row in candidates])
        old_scores = np.array([row['old094_score'] for row in candidates])
        old_slot = int(old_scores.argmax())
        new_slot = int(centered.argmax()) if common_sites >= 16 else old_slot
        support_slot = int(support_score.argmax())
        valid_score = common_sites >= 16 and np.std(centered) > 1e-8
        result_rows.append({'section_id': record['section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'angle_bin': candidates[0]['angle_bin'],
            'panel_file_sha256': record['sha256'],
            'common_heldout_sites': common_sites,
            'predicted_tissue_sites': int((tissue >= .5).sum()),
            'centered_scores': centered.tolist(),
            'center_support_fractions': support_score.tolist(),
            'mapped96_errors_um': errors.tolist(),
            'old094_scores': old_scores.tolist(),
            'old_slot': old_slot, 'centered_slot': new_slot,
            'support_slot': support_slot,
            'old_error_um': float(errors[old_slot]),
            'centered_error_um': float(errors[new_slot]),
            'support_error_um': float(errors[support_slot]),
            'best14_error_um': float(errors.min()),
            'rho_old': float(spearmanr(old_scores, -errors).statistic) if valid_score else None,
            'rho_centered': float(spearmanr(centered, -errors).statistic) if valid_score else None,
            'rho_support': (float(spearmanr(support_score, -errors).statistic)
                if valid_score and np.std(support_score) > 1e-8 else None)})
        if index % 32 == 0:
            print(json.dumps({'eligible_sections_done': index,
                'eligible_sections_total': len(records)}), flush=True)

plan_ids = sorted({row['synthetic_subject_plan_id'] for row in result_rows})
groups = {'all': result_rows,
    'raw': [row for row in result_rows if row['appearance_mode'] == 'raw'],
    'heavy_oblique': [row for row in result_rows if row['angle_bin'] == '45-54.7']}
summary = {}
for name, group in groups.items():
    plan_groups = [[row for row in group if row['synthetic_subject_plan_id'] == plan]
                   for plan in plan_ids]
    plan_groups = [rows for rows in plan_groups if rows]
    scored = [row for row in group if row['rho_centered'] is not None]
    summary[name] = {'sections': len(group), 'plans': len(plan_groups),
        'fallback_sections': sum(row['common_heldout_sites'] < 16 for row in group),
        'plan_equal_mm': {field: float(np.mean([np.mean([row[field] for row in rows])
            for rows in plan_groups]) / 1000) for field in
            ('old_error_um', 'centered_error_um', 'support_error_um', 'best14_error_um')},
        'mean_within_section_rho': {field: (float(np.mean([
            row[field] for row in scored if row[field] is not None])) if any(
                row[field] is not None for row in scored) else None)
            for field in ('rho_old', 'rho_centered', 'rho_support')},
        'mean_common_heldout_sites': float(np.mean([
            row['common_heldout_sites'] for row in group]))}
all_scores = summary['all']
gain = all_scores['plan_equal_mm']['old_error_um'] - all_scores['plan_equal_mm']['centered_error_um']
rho = all_scores['mean_within_section_rho']
conditions = {'selected_gain_at_least_0.20_mm': gain >= .2,
    'rho_gain_at_least_0.10': rho['rho_centered'] >= rho['rho_old'] + .1,
    'rho_over_support_at_least_0.10': rho['rho_support'] is not None and
        rho['rho_centered'] >= rho['rho_support'] + .1,
    'raw_nonregression_0.20_mm': summary['raw']['plan_equal_mm']['centered_error_um'] <=
        summary['raw']['plan_equal_mm']['old_error_um'] + .2,
    'heavy_oblique_nonregression_0.20_mm': summary['heavy_oblique']['plan_equal_mm']['centered_error_um'] <=
        summary['heavy_oblique']['plan_equal_mm']['old_error_um'] + .2}
summary['criterion'] = {'passed': all(conditions.values()), 'conditions': conditions,
    'selected_gain_mm': gain, 'interpretation': 'reused synthetic development, no weight update'}
out.mkdir(parents=True, exist_ok=False)
with (out / 'sections.jsonl').open('w') as stream:
    for row in result_rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'parent_checkpoint_sha256': sha(parent_file),
    'descriptor_checkpoint_sha256': sha(descriptor_file),
    'development_candidates_sha256': sha(development / 'candidates.jsonl'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'sections': len(result_rows), 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(result_rows),
    'criterion': summary['criterion']}), flush=True)
