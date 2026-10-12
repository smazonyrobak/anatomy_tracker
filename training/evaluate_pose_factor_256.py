"""Frozen v7 DEV readout of direct-anchor physical-factor training."""
import hashlib
import json
import math
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
BASELINE = ROOT / 'runs/one_shot_v7_heldout_dev_panel_248_readout_001'
TRAIN_254 = ROOT / 'runs/pose_direct_full_pose_254_train_001'
READOUT_254 = ROOT / 'runs/pose_direct_full_pose_254_readout_001'
CAPTURE_255 = ROOT / 'runs/pose_direct_capture_255_001'
TRAIN = ROOT / 'runs/pose_factor_256_train_001'
OUT = ROOT / 'runs/pose_factor_256_readout_001'
DOC = Path('docs/publication/POSE_FACTOR_256_READOUT_PROTOCOL_20261012.md')
CHECKPOINT = TRAIN / 'joint_step_03000.pt'
SIDE = 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
base_done = json.loads((BASELINE / 'completed.json').read_text())
train254_done = json.loads((TRAIN_254 / 'completed.json').read_text())
old_done = json.loads((READOUT_254 / 'completed.json').read_text())
capture_done = json.loads((CAPTURE_255 / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
config = json.loads((TRAIN / 'config.json').read_text())
assert train_done['draws_sha256'] == train254_done['draws_sha256']
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((BASELINE / 'rows.jsonl').read_bytes()).hexdigest() == base_done['rows_sha256']
assert hashlib.sha256((READOUT_254 / 'rows.jsonl').read_bytes()).hexdigest() == old_done['rows_sha256']
assert hashlib.sha256((CAPTURE_255 / 'rows.jsonl').read_bytes()).hexdigest() == capture_done['rows_sha256']
assert hashlib.sha256((TRAIN / 'training.jsonl').read_bytes()).hexdigest() == train_done['training_sha256']
assert hashlib.sha256((TRAIN / 'config.json').read_bytes()).hexdigest() == train_done['config_sha256']
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == train_done['checkpoint_sha256']['3000']
assert hashlib.sha256(Path('docs/publication/POSE_FACTOR_256_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
assert hashlib.sha256(Path('training/arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest() == config['source_sha256']['arbitrary_plane_one_shot_model.py']
assert hashlib.sha256(Path('training/arbitrary_plane_full_frame_primitives.py').read_bytes()).hexdigest() == config['source_sha256']['arbitrary_plane_full_frame_primitives.py']

records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
base_rows = {r['physical_section_id']: r for r in
             (json.loads(line) for line in (BASELINE / 'rows.jsonl').open(encoding='utf8'))}
old_rows = {r['physical_section_id']: r for r in
            (json.loads(line) for line in (READOUT_254 / 'rows.jsonl').open(encoding='utf8'))}
capture_rows = {r['physical_section_id']: r for r in
                (json.loads(line) for line in (CAPTURE_255 / 'rows.jsonl').open(encoding='utf8'))}
eligible = [r for r in records if r['eligible'] and r['valid_pixels'] > 0]
assert len(records) == panel_done['physical_sections'] == 256
assert len(eligible) == panel_done['eligible'] == len(base_rows) == len(old_rows) == len(capture_rows) == 227
assert {r['provenance']['physical_section_id'] for r in eligible} == set(base_rows) == set(old_rows) == set(capture_rows)
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
            true_reflection = int(arrays['reflection'])
        prediction = model.predict(image)
        centre, frame, basis = full_frame_state_to_components(prediction['state'][0])
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
        true_axis = true_frame[0, :, 2]
        axis_error = torch.rad2deg(torch.acos((frame[:, :, 2] @ true_axis).abs().clamp_max(1)))
        chart = full_chart[valid]
        chart_both = chart[None].expand(2, -1, -1).clone()
        chart_both[1, :, 0] = (SIDE - 1) / SIDE - chart_both[1, :, 0]
        physical = centre[:, None, None, :] + torch.einsum(
            'kij,spj->kspi', frame[:, :, :2] @ basis, chart_both - .5)
        error = (physical - target[valid][None, None]).norm(dim=-1).mean(-1) / 1000
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        selected = int((prediction['log_mass'][..., None] + reflection_log).flatten().argmax())
        top4 = prediction['log_mass'][0].topk(4).indices
        fov = torch.linalg.svdvals(basis[selected // 2]) / 1000
        true_fov = torch.linalg.svdvals(true_basis[0]) / 1000
        raster_span = (frame[selected // 2, :, :2] @ basis[selected // 2]).norm(dim=-2) / 1000
        true_raster_span = (true_frame[0, :, :2] @ true_basis[0]).norm(dim=-2) / 1000
        identity = record['provenance']['physical_section_id']
        old, base, capture = old_rows[identity], base_rows[identity], capture_rows[identity]
        assert old['panel_file_sha256'] == base['panel_file_sha256'] == capture['panel_file_sha256'] == record['sha256']
        row = {
            'physical_section_id': identity, 'animal_id': record['animal_id'],
            'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
            'section_id': record['section_id'], 'panel_file_sha256': record['sha256'],
            'appearance_mode': record['appearance_mode'],
            'cut_nearest_principal_deg': base['cut_nearest_principal_axis_deg'],
            'valid_fraction': record['valid_pixels'] / SIDE**2,
            'selected_axis_error_deg': float(axis_error[selected // 2]),
            'selected_rigid_mm': float(error.flatten()[selected]),
            'top4x2_oracle_rigid_mm': float(error[top4].min()),
            'all128_oracle_rigid_mm': float(error.min()),
            'reflection_oracle_rigid_mm': float(error[selected // 2].min()),
            'centre_error_mm': float((centre[selected // 2] - true_centre[0]).norm() / 1000),
            'fov_mm': fov.tolist(), 'true_fov_mm': true_fov.tolist(),
            'minor_fov_le_1p2_mm': bool(fov.min() <= 1.2),
            'raster_span_log_ratio': float((raster_span.log() - true_raster_span.log()).abs().mean()),
            'selected_anchor': selected // 2, 'selected_reflection': selected % 2,
            'true_reflection': true_reflection,
            'baseline247_axis_error_deg': base['prior_top1_normal_axis_error_deg'],
            'baseline247_rigid_mm': base['prior_top1_rigid_um'] / 1000,
            'old254_axis_error_deg': old['selected_axis_error_deg'],
            'old254_rigid_mm': old['selected_rigid_mm'],
            'old254_all128_oracle_rigid_mm': capture['all128_oracle_rigid_mm'],
        }
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

metrics = ('selected_axis_error_deg', 'selected_rigid_mm', 'top4x2_oracle_rigid_mm',
           'all128_oracle_rigid_mm', 'reflection_oracle_rigid_mm', 'centre_error_mm',
           'raster_span_log_ratio', 'baseline247_axis_error_deg', 'baseline247_rigid_mm',
           'old254_axis_error_deg', 'old254_rigid_mm', 'old254_all128_oracle_rigid_mm')
animals = sorted({r['animal_id'] for r in rows})
assert len(animals) == panel_done['synthetic_subjects'] == 8
subject_means = {animal: {key: float(np.mean([r[key] for r in rows if r['animal_id'] == animal]))
                          for key in metrics} for animal in animals}
equal = {key: float(np.mean([subject_means[animal][key] for animal in animals]))
         for key in metrics}
normal_ce = [json.loads(line)['normal_ce'] for line in (TRAIN / 'training.jsonl').open(encoding='utf8')]
assert len(normal_ce) == train_done['batches'] == 3000
narrow = sum(r['minor_fov_le_1p2_mm'] for r in rows)
gates = {'selected_rigid': equal['selected_rigid_mm'] <= 4.0,
         'axis': equal['selected_axis_error_deg'] <= 51.0,
         'minor_fov_collapse': narrow <= 10}
strata = {}
for label, group in {
    **{f'cut_{low}_{high}': [r for r in rows if low <= r['cut_nearest_principal_deg'] < high]
       for low, high in ((0, 15), (15, 30), (30, 45), (45, 60))},
    **{f'visible_{low}_{high}': [r for r in rows if low <= r['valid_fraction'] < high]
       for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01))},
}.items():
    strata[label] = {'cases': len(group),
                     'case_mean_selected_rigid_mm': float(np.mean([r['selected_rigid_mm'] for r in group])) if group else None,
                     'case_mean_axis_deg': float(np.mean([r['selected_axis_error_deg'] for r in group])) if group else None,
                     'case_mean_all128_oracle_mm': float(np.mean([r['all128_oracle_rigid_mm'] for r in group])) if group else None}
summary = {
    'status': 'synthetic DEV pose-factor mechanism only; map/refiner/scorer untrained; not deployable',
    'cases': len(rows), 'synthetic_subjects': len(animals),
    'subject_equal_mean': equal, 'subject_means': subject_means, 'strata': strata,
    'last500_train_normal_ce': float(np.mean(normal_ce[-500:])),
    'uniform_normal_ce': math.log(64),
    'minor_fov_le_1p2_mm_cases': narrow,
    'all128_capture_le_1mm_cases': sum(r['all128_oracle_rigid_mm'] <= 1 for r in rows),
    'selected_to_all128_gap_mm': equal['selected_rigid_mm'] - equal['all128_oracle_rigid_mm'],
    'reflection_accuracy': float(np.mean([r['selected_reflection'] == r['true_reflection'] for r in rows])),
    'gates': gates, 'continuation_gate': all(gates.values()),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(rows),
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'training_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'readout254_completed_sha256': hashlib.sha256((READOUT_254 / 'completed.json').read_bytes()).hexdigest(),
    'capture255_completed_sha256': hashlib.sha256((CAPTURE_255 / 'completed.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows),
                  'continuation_gate': summary['continuation_gate']}), flush=True)
