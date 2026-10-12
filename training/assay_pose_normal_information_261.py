"""Read-only normal-axis collapse and image-pairing assay for 260."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
TRAIN = ROOT / 'runs/dense_physical_pose_260_train_001'
PRIOR = ROOT / 'runs/dense_physical_pose_260_readout_001'
OUT = ROOT / 'runs/pose_normal_information_261_dev_001'
DOC = Path('docs/publication/POSE_NORMAL_INFORMATION_261_PROTOCOL_20261012.md')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
prior_done = json.loads((PRIOR / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((PRIOR / 'rows.jsonl').read_bytes()).hexdigest() == prior_done['rows_sha256']
assert hashlib.sha256((TRAIN / 'joint_step_03000.pt').read_bytes()).hexdigest() == train_done['checkpoint_sha256']['3000']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [record for record in records if record['eligible'] and record['valid_pixels'] > 0]
prior = {row['physical_section_id']: row for row in
         (json.loads(line) for line in (PRIOR / 'rows.jsonl').open(encoding='utf8'))
         if row['batch'] == 3000}
assert len(eligible) == len(prior) == 227
assert {record['provenance']['physical_section_id'] for record in eligible} == set(prior)

model = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
                              dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
saved = torch.load(TRAIN / 'joint_step_03000.pt', map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved
OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        predicted = model.dense_coordinate_plane(model.predict(image), source_side=192)
        _, frame, _ = full_frame_state_to_components(predicted)
        _, true_frame, _ = full_frame_state_to_components(truth)
        p = frame[0, :, 2].cpu().numpy()
        t = true_frame[0, :, 2].cpu().numpy()
        angle = float(np.rad2deg(np.arccos(np.clip(abs(p @ t), 0, 1))))
        identity = record['provenance']['physical_section_id']
        assert abs(angle - prior[identity]['fitted_axis_deg']) < .001
        row = {'physical_section_id': identity, 'animal_id': record['animal_id'],
               'panel_file_sha256': record['sha256'], 'predicted_normal': p.tolist(),
               'true_normal': t.tolist(), 'axis_deg': angle}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

p = np.asarray([row['predicted_normal'] for row in rows])
t = np.asarray([row['true_normal'] for row in rows])
animals = np.asarray([row['animal_id'] for row in rows])
groups = [np.flatnonzero(animals == animal) for animal in sorted(set(animals))]
eigenvalues = {'predicted': np.linalg.eigvalsh(p.T @ p / len(p)).tolist(),
               'true': np.linalg.eigvalsh(t.T @ t / len(t)).tolist()}
matched = float(np.mean([np.abs((p[g] * t[g]).sum(-1)).mean() for g in groups]))
rng = np.random.default_rng(2026101261)
null = []
for _ in range(1024):
    null.append(float(np.mean([np.abs((p[g] * t[rng.permutation(g)]).sum(-1)).mean()
                               for g in groups])))
summary = {'cases': len(rows), 'synthetic_plans': len(groups),
           'axis_scatter_eigenvalues_ascending': eigenvalues,
           'matched_equal_plan_abs_cosine': matched,
           'permuted_equal_plan_abs_cosine_mean': float(np.mean(null)),
           'matched_minus_permuted': matched - float(np.mean(null)),
           'normal_information_threshold_met': matched - float(np.mean(null)) >= .05,
           'permutation_draws': len(null),
           'equal_plan_axis_error_deg': float(np.mean([np.mean([rows[i]['axis_deg'] for i in g]) for g in groups]))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'train_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'prior_completed_sha256': hashlib.sha256((PRIOR / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
