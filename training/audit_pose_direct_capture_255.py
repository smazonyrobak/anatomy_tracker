"""Read-only full-branch coordinate capture audit of frozen 254 on v7 DEV."""
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
TRAIN = ROOT / 'runs/pose_direct_full_pose_254_train_001'
READOUT = ROOT / 'runs/pose_direct_full_pose_254_readout_001'
OUT = ROOT / 'runs/pose_direct_capture_255_001'
DOC = Path('docs/publication/POSE_DIRECT_CAPTURE_255_PROTOCOL_20261012.md')
CHECKPOINT = TRAIN / 'joint_step_03000.pt'
SIDE = 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
readout_done = json.loads((READOUT / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == train_done['checkpoint_sha256']['3000']
assert hashlib.sha256((READOUT / 'rows.jsonl').read_bytes()).hexdigest() == readout_done['rows_sha256']
assert readout_done['checkpoint_sha256'] == train_done['checkpoint_sha256']['3000']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
readout = {r['physical_section_id']: r for r in
           (json.loads(line) for line in (READOUT / 'rows.jsonl').open(encoding='utf8'))}
eligible = [r for r in records if r['eligible'] and r['valid_pixels'] > 0]
assert len(records) == panel_done['physical_sections'] == 256
assert len(eligible) == panel_done['eligible'] == len(readout) == 227
assert {r['provenance']['physical_section_id'] for r in eligible} == set(readout)
assert all(hashlib.sha256((PANEL / r['file']).read_bytes()).hexdigest() == r['sha256'] for r in records)

model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=False,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().eval()
saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved
yy, xx = torch.meshgrid(torch.arange(SIDE, device='cuda') / SIDE,
                        torch.arange(SIDE, device='cuda') / SIDE, indexing='ij')
full_chart = torch.stack((xx, yy), -1)

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
        prediction = model.predict(image)
        centre, frame, basis = full_frame_state_to_components(prediction['state'][0])
        true_axis = full_frame_state_to_components(truth)[1][0, :, 2]
        axis_error = torch.rad2deg(torch.acos((frame[:, :, 2] @ true_axis).abs().clamp_max(1)))
        chart = full_chart[valid]
        chart_both = chart[None].expand(2, -1, -1).clone()
        chart_both[1, :, 0] = (SIDE - 1) / SIDE - chart_both[1, :, 0]
        physical = centre[:, None, None, :] + torch.einsum(
            'kij,spj->kspi', frame[:, :, :2] @ basis, chart_both - .5)
        error = (physical - target[valid][None, None]).norm(dim=-1).mean(-1) / 1000
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        scores = (prediction['log_mass'][..., None] + reflection_log)[0]
        selected = int(scores.flatten().argmax())
        best_all = int(error.flatten().argmin())
        top4 = prediction['log_mass'][0].topk(4).indices
        top4_best = int(error[top4].flatten().argmin())
        near3 = ((model.normal_anchor_frames[:, :, 2] @ true_axis).abs()).topk(3).indices
        near3_best = int(error[near3].flatten().argmin())
        old = readout[record['provenance']['physical_section_id']]
        assert old['panel_file_sha256'] == record['sha256']
        assert abs(float(error.flatten()[selected]) - old['selected_rigid_mm']) < 1e-4
        assert abs(float(error[selected // 2].min()) - old['reflection_oracle_rigid_mm']) < 1e-4
        assert abs(float(axis_error[selected // 2]) - old['selected_axis_error_deg']) < 1e-3
        row = {
            'physical_section_id': record['provenance']['physical_section_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
            'panel_file_sha256': record['sha256'],
            'selected_rigid_mm': float(error.flatten()[selected]),
            'top4x2_oracle_rigid_mm': float(error[top4].min()),
            'all128_oracle_rigid_mm': float(error.min()),
            'true_near3x2_oracle_rigid_mm': float(error[near3].min()),
            'selected_axis_error_deg': float(axis_error[selected // 2]),
            'all128_best_axis_error_deg': float(axis_error[best_all // 2]),
            'selected_mode': selected // 2, 'selected_reflection': selected % 2,
            'all128_best_mode': best_all // 2, 'all128_best_reflection': best_all % 2,
            'all128_best_prior_rank': int((scores.flatten() > scores.flatten()[best_all]).sum() + 1),
            'top4_modes': top4.tolist(), 'true_near3_modes': near3.tolist(),
            'top4_best_branch_local_index': top4_best,
            'near3_best_branch_local_index': near3_best,
        }
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

metrics = ('selected_rigid_mm', 'top4x2_oracle_rigid_mm', 'all128_oracle_rigid_mm',
           'true_near3x2_oracle_rigid_mm', 'selected_axis_error_deg',
           'all128_best_axis_error_deg')
animals = sorted({r['animal_id'] for r in rows})
assert len(animals) == panel_done['synthetic_subjects'] == 8
subject_means = {animal: {key: float(np.mean([r[key] for r in rows if r['animal_id'] == animal]))
                          for key in metrics} for animal in animals}
equal = {key: float(np.mean([subject_means[animal][key] for animal in animals]))
         for key in metrics}
summary = {
    'status': 'truth-assisted candidate-capture diagnostic only; not a deployable prediction',
    'cases': len(rows), 'synthetic_subjects': len(animals),
    'subject_equal_mean': equal, 'subject_means': subject_means,
    'selected_to_top4x2_gap_mm': equal['selected_rigid_mm'] - equal['top4x2_oracle_rigid_mm'],
    'selected_to_all128_gap_mm': equal['selected_rigid_mm'] - equal['all128_oracle_rigid_mm'],
    'all128_capture_counts': {str(limit): sum(r['all128_oracle_rigid_mm'] <= limit for r in rows)
                              for limit in (1, 2, 3)},
    'top4x2_capture_counts': {str(limit): sum(r['top4x2_oracle_rigid_mm'] <= limit for r in rows)
                              for limit in (1, 2, 3)},
    'all128_best_prior_rank_median': float(np.median([r['all128_best_prior_rank'] for r in rows])),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(rows),
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'training_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'readout_completed_sha256': hashlib.sha256((READOUT / 'completed.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows)}), flush=True)
