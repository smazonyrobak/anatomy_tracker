"""Read-only 092 local-match diagnostic on eight frozen DEV sections per plan.

Question: does a model-generated candidate's actual 2-D↔3-D match correctness
identify the physically better plane, and does learned reliability recognize
those correct matches? No score is trained or promoted here.
"""
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
from scipy.stats import rankdata, spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.spatial_joint_fit_092 import SpatialJointFit092
from training.whole_slice_atlas_feedback_083 import (
    WholeSliceAtlasFeedback083, synthetic_match_targets,
)

train = root / 'runs/spatial_joint_fit_092_pilot'
evaluation = root / 'runs/spatial_joint_fit_092_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/spatial_match_signal_093_diagnostic'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def corr(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.


training_receipt = json.loads((train / 'completed.json').read_text())
evaluation_receipt = json.loads((evaluation / 'completed.json').read_text())
assert sha(train / 'joint_step_06000.pt') == training_receipt['checkpoint_sha256']['6000']
assert sha(evaluation / 'candidates.jsonl') == evaluation_receipt['candidates_sha256']
assert sha(evaluation / 'summary.json') == evaluation_receipt['summary_sha256']
assert sha(panel / 'records.jsonl') == '481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a'
all_records = [r for r in map(json.loads, (panel / 'records.jsonl').open())
               if r['eligible']]
by_plan = defaultdict(list)
for record in all_records:
    by_plan[record['synthetic_subject_plan_id']].append(record)
assert len(by_plan) == 8
rng = np.random.default_rng(20261004093)
records = []
for plan in sorted(by_plan):
    pool = sorted(by_plan[plan], key=lambda x: x['section_id'])
    records.extend(pool[i] for i in sorted(rng.choice(len(pool), 8, replace=False)))
chosen_ids = {r['section_id'] for r in records}
candidates = defaultdict(list)
for row in map(json.loads, (evaluation / 'candidates.jsonl').open()):
    if row['step'] == 6000 and row['in_fixed_beam'] and row['section_id'] in chosen_ids:
        candidates[row['section_id']].append(row)
assert len(records) == len(candidates) == 64
assert all(len(candidates[r['section_id']]) == 14 for r in records)

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'script_sha256': sha(__file__),
    'checkpoint_sha256': sha(train / 'joint_step_06000.pt'),
    'training_completed_sha256': sha(train / 'completed.json'),
    'evaluation_completed_sha256': sha(evaluation / 'completed.json'),
    'evaluation_candidates_sha256': sha(evaluation / 'candidates.jsonl'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'sample_seed': 20261004093, 'sections_per_plan': 8,
    'section_ids': [r['section_id'] for r in records],
    'planes_per_section': 14, 'fine_correct_distance_um': 1500,
    'coarse_correct_distance_um': 2000,
    'score_use': 'diagnostic only; no reranking or checkpoint selection',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
spatial = SpatialJointFit092(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
checkpoint = torch.load(train / 'joint_step_06000.pt',
                        map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
spatial.load_state_dict(checkpoint['spatial_fit'], strict=True)
del checkpoint
rows = []
reliability_values, correct_values = [], []
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as stream:
    for record in records:
        group = candidates[record['section_id']]
        group.sort(key=lambda x: x['beam_slot'])
        assert [r['beam_slot'] for r in group] == list(range(14))
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        feature = model.predict(image)['feature']
        states = torch.tensor([r['initial_state'] for r in group],
                              device='cuda', dtype=image.dtype)[None]
        reflections = torch.tensor([r['reflection'] for r in group],
                                   device='cuda', dtype=torch.long)[None]
        for start in range(0, 14, 2):
            state = states[:, start:start + 2]
            reflected = reflections[:, start:start + 2]
            result = spatial(feature, state, reflected, atlas, offsets, weights)
            assert torch.allclose(result['state'][0],
                torch.tensor([r['fitted_state'] for r in group[start:start + 2]],
                    device='cuda', dtype=image.dtype), atol=.05, rtol=1e-5)
            for scale, size, tolerance, fit_state in (
                ('coarse', 16, 2000, state),
                ('fine', 32, 1500, result['state'])):
                labels = synthetic_match_targets(target, valid, fit_state, reflected)
                truth = F.interpolate(target.permute(0, 3, 1, 2),
                    (size, size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                visible = F.interpolate(valid[:, None].float(), (size, size),
                    mode='bilinear', align_corners=False)[:, 0] == 1
                logits = result[f'{scale}_match_logits']
                support = result[f'{scale}_match_support']
                peak = logits.masked_fill(support < .5, -1e4).argmax(2)
                peak_supported = support.gather(2, peak[:, :, None]).squeeze(2) >= .5
                at_truth = support.gather(
                    2, labels[f'{scale}_index'][:, :, None]).squeeze(2) >= .5
                in_window = labels[f'{scale}_mask'] & at_truth
                distance = (result[f'{scale}_match_ccf_um'] - truth[:, None]).norm(dim=-1)
                correct = visible[:, None] & peak_supported & (distance <= tolerance)
                reliability = result[f'{scale}_reliability_logit'].sigmoid()
                for offset, candidate in enumerate(group[start:start + 2]):
                    v = visible[0]
                    selected = v & peak_supported[0, offset]
                    row = {'section_id': record['section_id'],
                        'plan_id': record['synthetic_subject_plan_id'],
                        'appearance_mode': record['appearance_mode'],
                        'branch_id': candidate['branch_id'],
                        'beam_slot': candidate['beam_slot'],
                        'scale': scale,
                        'mapped96_error_um': candidate['mapped96_error_um'],
                        'visible_sites': int(v.sum()),
                        'supported_sites': int(selected.sum()),
                        'in_window_supported_fraction': float(
                            in_window[0, offset].sum() / v.sum().clamp_min(1)),
                        'correct_match_fraction': float(
                            correct[0, offset].sum() / v.sum().clamp_min(1)),
                        'reliability_visible_mean': float(
                            reliability[0, offset][v].mean()),
                        'reliability_supported_mean': float(
                            reliability[0, offset][selected].mean())
                            if selected.any() else 0.,
                        'fit_energy': candidate['fit_energy'],
                        'atlas_support_fraction': candidate['atlas_support_fraction']}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    if scale == 'fine':
                        reliability_values.append(reliability[0, offset][v].cpu().numpy())
                        correct_values.append(correct[0, offset][v].cpu().numpy())
    stream.flush()

summaries = {}
for scale in ('coarse', 'fine'):
    group = defaultdict(list)
    for row in rows:
        if row['scale'] == scale:
            group[row['section_id']].append(row)
    result = []
    for candidates in group.values():
        error = np.array([r['mapped96_error_um'] for r in candidates])
        result.append({'plan': candidates[0]['plan_id'],
            **{name: corr(np.array([r[name] for r in candidates]), -error)
               for name in ('in_window_supported_fraction',
                            'correct_match_fraction',
                            'reliability_visible_mean',
                            'reliability_supported_mean',
                            'atlas_support_fraction')},
            'correct_at_best': candidates[int(error.argmin())]['correct_match_fraction'],
            'correct_at_worst': candidates[int(error.argmax())]['correct_match_fraction']})
    plans = {r['plan'] for r in result}
    summaries[scale] = {key: float(np.mean([
        np.mean([r[key] for r in result if r['plan'] == plan])
        for plan in plans])) for key in result[0] if key != 'plan'}
positive = np.concatenate(correct_values)
confidence = np.concatenate(reliability_values)
count = int(positive.sum())
ranks = rankdata(confidence)
summaries['fine']['site_reliability_auroc'] = float(
    (ranks[positive].sum() - count * (count + 1) / 2)
    / (count * (len(positive) - count)))
summaries['fine']['site_correct_fraction'] = float(positive.mean())
(out / 'summary.json').write_text(json.dumps(summaries, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(records), 'candidate_scale_rows': len(rows),
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(records),
    'fine': summaries['fine'], 'coarse': summaries['coarse']}), flush=True)
