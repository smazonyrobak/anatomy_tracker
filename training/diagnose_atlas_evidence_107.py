"""Frozen 106 matcher: can it recognize a correct plane on reused synthetic DEV?"""
import hashlib
import json
import math
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
from training.coarse_pose_updater_106 import CoarsePoseUpdater106
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

panel = root / 'data/v3_pose_capture_confirmation_panel_001'
evaluation = root / 'runs/coarse_pose_updater_106_dev_eval'
out = root / 'runs/atlas_evidence_diagnostic_107'
assert not out.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
pose.load_state_dict(torch.load(root / 'runs/v3_one_pass_pose_capture_pilot_001/joint_step_02000.pt',
    map_location='cpu', weights_only=True)['model'], strict=True)
model = CoarsePoseUpdater106(WholeSliceAtlasFeedback083(),
    SpatialJointFit094(WholeSliceAtlasFeedback083()).validity_head)
model.load_state_dict(torch.load(root / 'runs/coarse_pose_updater_106/pose_step_04000.pt',
    map_location='cpu', weights_only=True)['models']['atlas'], strict=True)
model = model.cuda().eval().requires_grad_(False)
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
records = {row['section_id']: row for row in
    (json.loads(line) for line in (panel / 'records.jsonl').open())}
sections = [json.loads(line) for line in (evaluation / 'sections.jsonl').open()
    if line.strip()]
sections = [row for row in sections if row['step'] == 4000 and row['arm'] == 'atlas']
candidates = [json.loads(line) for line in (evaluation / 'candidates.jsonl').open()
    if line.strip()]
candidates = {(row['section_id'], row['beam_slot']): row for row in candidates
    if row['step'] == 4000 and row['arm'] == 'atlas'}
assert len(sections) == 64 and len(candidates) == 64 * 14
near = [depth * 81 + dy * 9 + dx for depth in (3, 4, 5)
    for dy in (3, 4, 5) for dx in (3, 4, 5)]
rows = []

with torch.inference_mode():
    for index, section in enumerate(sections, 1):
        record = records[section['section_id']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().float()
            target_state = torch.from_numpy(arrays['target_state'][None, None].copy()).cuda()
            target_reflection = torch.from_numpy(arrays['reflection'].reshape(1, 1).copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        feature = pose.predict(image)['feature']
        tissue = F.adaptive_avg_pool2d(model.validity_head(
            F.adaptive_avg_pool2d(feature, (32, 32))).sigmoid(), (16, 16))[:, 0]
        true_tissue = F.adaptive_avg_pool2d(valid[:, None], (16, 16))[:, 0]
        beam = [candidates[(section['section_id'], slot)] for slot in range(14)]
        states = torch.cat((torch.tensor([row['input_state'] for row in beam],
            device='cuda', dtype=target_state.dtype)[None], target_state), 1)
        reflection = torch.cat((torch.tensor([row['branch_id'] % 2 for row in beam],
            device='cuda', dtype=torch.long)[None], target_reflection), 1)
        candidate_scores = []
        for start in range(0, 15, 2):
            match = model.matcher(feature, states[:, start:start + 2],
                reflection[:, start:start + 2], atlas, offsets, weights,
                source_shape=(256, 256), match_only=True)
            logits = match['coarse_match_logits']
            support = match['coarse_match_support']
            near_logmass = torch.logsumexp(logits[:, :, near], 2) - \
                torch.logsumexp(logits, 2)
            measurements = {'near_logmass': near_logmass,
                'center_logit': logits[:, :, 364],
                'peak_logit': logits.amax(2),
                'support_any': support.amax(2)}
            for local in range(logits.shape[1]):
                item = {}
                for name, value in measurements.items():
                    for mask_name, mask in (('blind', tissue), ('true_mask', true_tissue)):
                        item[f'{name}_{mask_name}'] = float((value[:, local] * mask).sum() /
                            mask.sum().clamp_min(1e-6))
                candidate_scores.append(item)
        assert len(candidate_scores) == 15
        for slot, item in enumerate(candidate_scores):
            source = beam[slot] if slot < 14 else None
            rows.append({'section_id': section['section_id'], 'slot': slot,
                'oracle_true_plane_diagnostic_only': slot == 14,
                'input_rigid_error_mm': source['input_rigid_error_um'] / 1000
                if source else 0., 'prior_score': source['prior_score'] if source else None,
                **item})
        if index % 16 == 0:
            print(json.dumps({'sections_done': index, 'sections_total': 64}), flush=True)

score_names = [key for key in rows[0] if key.endswith(('_blind', '_true_mask'))]
summary = {'role': 'exploratory, reused synthetic DEV; true plane is diagnostic only',
    'sections': 64, 'blind_candidates_per_section': 14, 'scores': {}}
for name in score_names:
    grouped = [[row for row in rows if row['section_id'] == section['section_id']]
        for section in sections]
    rho = [float(spearmanr([row[name] for row in group[:14]],
        [-row['input_rigid_error_mm'] for row in group[:14]]).statistic)
        for group in grouped]
    oracle = np.array([group[14][name] for group in grouped])
    prior = np.array([group[section['prior_slot']][name]
        for group, section in zip(grouped, sections)])
    best_blind = np.array([max(row[name] for row in group[:14]) for group in grouped])
    summary['scores'][name] = {'mean_within_beam_rho': float(np.nanmean(rho)),
        'oracle_above_prior_fraction': float(np.mean(oracle > prior)),
        'oracle_above_all_blind_fraction': float(np.mean(oracle > best_blind)),
        'oracle_minus_prior_median': float(np.median(oracle - prior)),
        'oracle_minus_best_blind_median': float(np.median(oracle - best_blind))}

out.mkdir(parents=True)
with (out / 'scores.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'panel_completion_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
    'evaluation_completion_sha256': hashlib.sha256(
        (evaluation / 'completed.json').read_bytes()).hexdigest(),
    'training_completion_sha256': hashlib.sha256(
        (root / 'runs/coarse_pose_updater_106/completed.json').read_bytes()).hexdigest(),
    'scores_sha256': hashlib.sha256((out / 'scores.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'no_new_training': True, 'uses_reused_dev': True,
    'true_plane_never_selectable': True}, indent=2))
print(json.dumps(summary), flush=True)
