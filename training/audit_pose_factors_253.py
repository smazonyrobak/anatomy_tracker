"""Frozen 247 image-only pose span/centre readout on the 248 synthetic DEV panel."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
TRAIN = ROOT / 'runs/anchor_only_blind_feedback_247_train_001'
BASELINE = ROOT / 'runs/one_shot_v7_heldout_dev_panel_248_readout_001'
OUT = ROOT / 'runs/pose_factor_audit_253_001'
PROTOCOL = Path('docs/publication/POSE_FACTOR_AUDIT_253_PROTOCOL_20261012.md')
STEP = 6000
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


panel_done = json.loads((PANEL / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
baseline_done = json.loads((BASELINE / 'completed.json').read_text())
assert sha(PANEL / 'records.jsonl') == panel_done['records_sha256']
assert sha(PANEL / 'protocol.json') == panel_done['protocol_sha256']
assert sha(TRAIN / f'joint_step_{STEP:05d}.pt') == train_done['checkpoint_sha256'][str(STEP)]
assert sha(BASELINE / 'rows.jsonl') == baseline_done['rows_sha256']
assert sha(PANEL / 'completed.json') == baseline_done['panel_sha256']['completed.json']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [r for r in records if r['eligible'] and r['valid_pixels'] > 0]
frozen = {r['physical_section_id']: r for r in
          map(json.loads, (BASELINE / 'rows.jsonl').open(encoding='utf8'))}
assert len(records) == 256 and len(eligible) == len(frozen) == 227
assert {r['provenance']['physical_section_id'] for r in eligible} == set(frozen)

model = OneShotJointSliceModel(modes=0, normal_anchor_count=64,
    shared_anchor_pose=True, atlas_conditioning=True,
    vector_refinement=True, fitted_ranking=True).cuda().eval()
checkpoint = torch.load(TRAIN / f'joint_step_{STEP:05d}.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == STEP and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        identity = record['provenance']['physical_section_id']
        old = frozen[identity]
        assert sha(PANEL / record['file']) == record['sha256'] == old['panel_file_sha256']
        assert record['animal_id'] == old['animal_id'] and record['section_id'] == old['section_id']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        prediction = model.predict(image)
        modes = prediction['log_mass'][0].topk(4).indices.repeat_interleave(2)
        reflections = torch.tensor([0, 1] * 4, device='cuda')
        prior = prediction['log_mass'][0, modes] + torch.where(reflections.bool(),
            F.logsigmoid(prediction['reflection_logit'][0, modes]),
            F.logsigmoid(-prediction['reflection_logit'][0, modes]))
        selected = int(prior.argmax())
        mode, reflection = int(modes[selected]), int(reflections[selected])
        assert mode == old['prior_top1_mode'] and reflection == old['prior_top1_reflection']
        true_c, true_f, true_b = full_frame_state_to_components(truth_state)
        pred_c, pred_f, pred_b = full_frame_state_to_components(prediction['state'][:, mode])
        true_span = torch.linalg.svdvals(true_b)[0].cpu().numpy() / 1000
        pred_span = torch.linalg.svdvals(pred_b)[0].cpu().numpy() / 1000
        centre_delta = ((pred_c - true_c)[0] / 1000).cpu().numpy()
        axis_deg = float(torch.rad2deg(torch.acos((pred_f[0, :, 2] * true_f[0, :, 2]).sum().abs().clamp_max(1))))
        assert abs(axis_deg - old['prior_top1_normal_axis_error_deg']) < .02
        row = {'physical_section_id': identity, 'animal_id': record['animal_id'],
            'section_id': record['section_id'], 'panel_file_sha256': record['sha256'],
            'selected_mode': mode, 'selected_reflection': reflection,
            'true_major_fov_mm': float(true_span[0]), 'true_minor_fov_mm': float(true_span[1]),
            'pred_major_fov_mm': float(pred_span[0]), 'pred_minor_fov_mm': float(pred_span[1]),
            'major_fov_abs_error_mm': float(abs(pred_span[0] - true_span[0])),
            'minor_fov_abs_error_mm': float(abs(pred_span[1] - true_span[1])),
            'major_fov_ratio': float(pred_span[0] / true_span[0]),
            'minor_fov_ratio': float(pred_span[1] / true_span[1]),
            'centre_error_mm': float(np.linalg.norm(centre_delta)),
            'centre_abs_ap_mm': float(abs(centre_delta[0])),
            'centre_abs_dv_mm': float(abs(centre_delta[1])),
            'centre_abs_ml_mm': float(abs(centre_delta[2])),
            'frozen_axis_error_deg': old['prior_top1_normal_axis_error_deg'],
            'frozen_rigid_error_mm': old['prior_top1_rigid_um'] / 1000}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows)}), flush=True)

metrics = ('true_major_fov_mm', 'true_minor_fov_mm', 'pred_major_fov_mm',
    'pred_minor_fov_mm', 'major_fov_abs_error_mm', 'minor_fov_abs_error_mm',
    'major_fov_ratio', 'minor_fov_ratio', 'centre_error_mm',
    'centre_abs_ap_mm', 'centre_abs_dv_mm', 'centre_abs_ml_mm',
    'frozen_axis_error_deg', 'frozen_rigid_error_mm')
plans = sorted({r['animal_id'] for r in rows})
plan_means = {plan: {key: float(np.mean([r[key] for r in rows if r['animal_id'] == plan]))
                     for key in metrics} for plan in plans}
axis_good = [r for r in rows if r['frozen_axis_error_deg'] <= 15]
axis_good_rigid_bad = [r for r in axis_good if r['frozen_rigid_error_mm'] >= 2]
summary = {'cases': len(rows), 'synthetic_plans': len(plans),
    'case_mean': {key: float(np.mean([r[key] for r in rows])) for key in metrics},
    'case_median': {key: float(np.median([r[key] for r in rows])) for key in metrics},
    'plan_equal_mean': {key: float(np.mean([plan_means[p][key] for p in plans])) for key in metrics},
    'plan_means': plan_means,
    'true_minor_fov_le_1p2mm': {'count': sum(r['true_minor_fov_mm'] <= 1.2 for r in rows),
        'percent': 100 * np.mean([r['true_minor_fov_mm'] <= 1.2 for r in rows])},
    'pred_minor_fov_le_1p2mm': {'count': sum(r['pred_minor_fov_mm'] <= 1.2 for r in rows),
        'percent': 100 * np.mean([r['pred_minor_fov_mm'] <= 1.2 for r in rows])},
    'axis_le_15deg_and_rigid_ge_2mm': {'count': len(axis_good_rigid_bad),
        'of_axis_le_15deg': len(axis_good), 'of_all_cases': len(rows),
        'conditional_percent': 100 * len(axis_good_rigid_bad) / len(axis_good),
        'all_case_percent': 100 * len(axis_good_rigid_bad) / len(rows)}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'source_sha256': {name: sha(Path('training') / name) for name in
        ('audit_pose_factors_253.py', 'arbitrary_plane_one_shot_model.py',
         'arbitrary_plane_full_frame_primitives.py')},
    'protocol_sha256': sha(PROTOCOL),
    'checkpoint_sha256': sha(TRAIN / f'joint_step_{STEP:05d}.pt'),
    'panel_completed_sha256': sha(PANEL / 'completed.json'),
    'panel_records_sha256': sha(PANEL / 'records.jsonl'),
    'frozen_248_readout_completed_sha256': sha(BASELINE / 'completed.json'),
    'frozen_248_rows_sha256': sha(BASELINE / 'rows.jsonl'),
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'cases': len(rows), 'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows),
    'axis_le_15deg_and_rigid_ge_2mm': len(axis_good_rigid_bad)}), flush=True)
