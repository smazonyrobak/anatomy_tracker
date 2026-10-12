"""Frozen 248 DEV readout of the fresh direct full-pose 254 checkpoint."""
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
NORMAL_HEADS = ROOT / 'runs/pose_normal_heads_252_readout_001'
TRAIN = ROOT / 'runs/pose_direct_full_pose_254_train_001'
OUT = ROOT / 'runs/pose_direct_full_pose_254_readout_001'
DOC = Path('docs/publication/POSE_DIRECT_FULL_POSE_254_READOUT_PROTOCOL_20261012.md')
STEP, SIDE = 3000, 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
baseline_done = json.loads((BASELINE / 'completed.json').read_text())
heads_done = json.loads((NORMAL_HEADS / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
config = json.loads((TRAIN / 'config.json').read_text())
checkpoint_file = TRAIN / f'joint_step_{STEP:05d}.pt'
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((BASELINE / 'rows.jsonl').read_bytes()).hexdigest() == baseline_done['rows_sha256']
assert hashlib.sha256((NORMAL_HEADS / 'rows.jsonl').read_bytes()).hexdigest() == heads_done['rows_sha256']
assert hashlib.sha256((TRAIN / 'training.jsonl').read_bytes()).hexdigest() == train_done['training_sha256']
assert hashlib.sha256((TRAIN / 'config.json').read_bytes()).hexdigest() == train_done['config_sha256']
assert hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() == train_done['checkpoint_sha256'][str(STEP)]
assert hashlib.sha256(Path('docs/publication/POSE_DIRECT_FULL_POSE_254_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
assert hashlib.sha256(Path('training/arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest() == config['source_sha256']['arbitrary_plane_one_shot_model.py']
assert hashlib.sha256(Path('training/arbitrary_plane_full_frame_primitives.py').read_bytes()).hexdigest() == config['source_sha256']['arbitrary_plane_full_frame_primitives.py']

records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
baseline_rows = {r['physical_section_id']: r for r in
                 (json.loads(line) for line in (BASELINE / 'rows.jsonl').open(encoding='utf8'))}
head_rows = {r['physical_section_id']: r for r in
             (json.loads(line) for line in (NORMAL_HEADS / 'rows.jsonl').open(encoding='utf8'))}
eligible = [r for r in records if r['eligible'] and r['valid_pixels'] > 0]
assert len(records) == panel_done['physical_sections'] == 256
assert len(eligible) == panel_done['eligible'] == len(baseline_rows) == len(head_rows) == 227
assert {r['provenance']['physical_section_id'] for r in eligible} == set(baseline_rows) == set(head_rows)
assert all(hashlib.sha256((PANEL / r['file']).read_bytes()).hexdigest() == r['sha256'] for r in records)

model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=False,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().eval()
saved = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
assert saved['step'] == STEP and not saved['calibrated']
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
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        choice = int((prediction['log_mass'][..., None] + reflection_log).flatten(1).argmax())
        mode, reflection = divmod(choice, 2)
        centre, frame, basis = full_frame_state_to_components(prediction['state'][:, mode])
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
        normal_error = torch.rad2deg(torch.acos((frame[0, :, 2] * true_frame[0, :, 2])
                                                 .sum().abs().clamp_max(1)))
        chart = full_chart[valid]
        reflected_chart = chart[None].expand(2, -1, -1).clone()
        reflected_chart[1, :, 0] = (SIDE - 1) / SIDE - reflected_chart[1, :, 0]
        rigid = centre[0] + torch.einsum('ij,spj->spi', frame[0, :, :2] @ basis[0],
                                        reflected_chart - .5)
        rigid_errors = (rigid - target[valid][None]).norm(dim=-1).mean(-1) / 1000
        fov = torch.linalg.svdvals(basis[0]) / 1000
        true_fov = torch.linalg.svdvals(true_basis[0]) / 1000
        identity = record['provenance']['physical_section_id']
        old, normal_head = baseline_rows[identity], head_rows[identity]
        assert old['panel_file_sha256'] == normal_head['panel_file_sha256'] == record['sha256']
        row = {'physical_section_id': identity, 'animal_id': record['animal_id'],
               'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
               'section_id': record['section_id'], 'panel_file_sha256': record['sha256'],
               'appearance_mode': record['appearance_mode'],
               'cut_nearest_principal_deg': old['cut_nearest_principal_axis_deg'],
               'valid_fraction': record['valid_pixels'] / SIDE**2,
               'selected_axis_error_deg': float(normal_error),
               'selected_rigid_mm': float(rigid_errors[reflection]),
               'reflection_oracle_rigid_mm': float(rigid_errors.min()),
               'centre_error_mm': float((centre[0] - true_centre[0]).norm() / 1000),
               'fov_mm': fov.tolist(), 'true_fov_mm': true_fov.tolist(),
               'fov_mean_abs_error_mm': float((fov - true_fov).abs().mean()),
               'minor_fov_le_1p2_mm': bool(fov.min() <= 1.2),
               'selected_anchor': mode, 'selected_reflection': reflection,
               'true_reflection': true_reflection,
               'baseline247_axis_error_deg': old['prior_top1_normal_axis_error_deg'],
               'baseline247_rigid_mm': old['prior_top1_rigid_um'] / 1000,
               'normal_only251_axis_error_deg': normal_head['axis_error_deg']['251']}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

metrics = ('selected_axis_error_deg', 'selected_rigid_mm', 'reflection_oracle_rigid_mm',
           'centre_error_mm', 'fov_mean_abs_error_mm', 'baseline247_axis_error_deg',
           'baseline247_rigid_mm', 'normal_only251_axis_error_deg')
animals = sorted({r['animal_id'] for r in rows})
assert len(animals) == panel_done['synthetic_subjects'] == 8
subject_means = {animal: {key: float(np.mean([r[key] for r in rows if r['animal_id'] == animal]))
                          for key in metrics} for animal in animals}
equal = {key: float(np.mean([subject_means[animal][key] for animal in animals]))
         for key in metrics}
train_ce = [json.loads(line)['normal_ce'] for line in (TRAIN / 'training.jsonl').open(encoding='utf8')]
assert len(train_ce) == train_done['batches'] == STEP
last500 = float(np.mean(train_ce[-500:]))
gates = {'normal_ce': last500 <= 3.659,
         'axis': equal['selected_axis_error_deg'] <= 42.78,
         'selected_rigid': equal['selected_rigid_mm'] <= 3.75}
summary = {
    'status': 'synthetic DEV pose mechanism only; map/refiner/scorer untrained; not deployable or calibrated',
    'cases': len(rows), 'synthetic_subjects': len(animals),
    'subject_equal_mean': equal, 'subject_means': subject_means,
    'normal_ce_last500': last500, 'uniform_normal_ce': math.log(64),
    'gates': gates, 'continuation_gate': all(gates.values()),
    'minor_fov_le_1p2_mm_cases': sum(r['minor_fov_le_1p2_mm'] for r in rows),
    'reflection_accuracy': float(np.mean([r['selected_reflection'] == r['true_reflection']
                                           for r in rows])),
    'axis_le_15_rigid_ge_2mm_cases': sum(r['selected_axis_error_deg'] <= 15 and
                                        r['selected_rigid_mm'] >= 2 for r in rows),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'checkpoint_step': STEP,
    'checkpoint_sha256': hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
    'training_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'baseline_completed_sha256': hashlib.sha256((BASELINE / 'completed.json').read_bytes()).hexdigest(),
    'normal_heads_completed_sha256': hashlib.sha256((NORMAL_HEADS / 'completed.json').read_bytes()).hexdigest(),
    'readout_protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows),
                  'continuation_gate': summary['continuation_gate']}), flush=True)
