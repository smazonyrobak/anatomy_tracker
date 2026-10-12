"""Check a fixed atlas-size FOV candidate with no truth-dependent inference input."""
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
PRIOR = ROOT / 'runs/spatial_candidate_fov_263_dev_001'
OUT = ROOT / 'runs/spatial_candidate_atlas_fov_264_dev_001'
DOC = Path('docs/publication/SPATIAL_CANDIDATE_ATLAS_FOV_264_PROTOCOL_20261012.md')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
prior_done = json.loads((PRIOR / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((TRAIN / 'joint_step_03000.pt').read_bytes()).hexdigest() == train_done['checkpoint_sha256']['3000']
assert hashlib.sha256((PRIOR / 'rows.jsonl').read_bytes()).hexdigest() == prior_done['rows_sha256']
records = [row for line in (PANEL / 'records.jsonl').open(encoding='utf8')
           if (row := json.loads(line))['eligible'] and row['valid_pixels'] > 0]
prior = {row['physical_section_id']: row for line in (PRIOR / 'rows.jsonl').open(encoding='utf8')
         if (row := json.loads(line))}
assert len(records) == len(prior) == 227

model = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
                              dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
saved = torch.load(TRAIN / 'joint_step_03000.pt', map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
del saved
anchor_model = OneShotJointSliceModel(modes=0, normal_anchor_count=64)
anchors = anchor_model.normal_anchor_frames[:, :, 2].cuda()
del anchor_model
offsets = torch.tensor((-2000., 0., 2000.), device='cuda')
atlas_span = torch.tensor((16000., 24000.), device='cuda')

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in records:
        identity = record['provenance']['physical_section_id']
        assert record['sha256'] == prior[identity]['panel_file_sha256']
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        predicted = model.dense_coordinate_plane(model.predict(image), source_side=192)
        centre, _, _ = full_frame_state_to_components(predicted)
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
        normal = torch.rad2deg(torch.acos((anchors @ true_frame[0, :, 2]).abs().clamp_max(1)))
        centres = centre[0, None, None] + anchors[:, None] * offsets[None, :, None]
        centre_mm = torch.linalg.vector_norm(centres - true_centre[0], dim=-1) / 1000
        true_span = torch.linalg.svdvals(true_basis)[0].sort().values
        span_log = float((atlas_span.log() - true_span.log()).abs().amax())
        correctable = bool(((normal[:, None] <= 15) & (centre_mm <= 3)).any() and span_log <= .7)
        row = {'physical_section_id': identity, 'animal_id': record['animal_id'],
               'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
               'visible_fraction': record['valid_pixels'] / 192**2,
               'prior_correctable': prior[identity]['correctable'], 'correctable': correctable,
               'nearest_centre_mm': float(centre_mm.min()),
               'fixed_span_log_ratio': span_log}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

plans = sorted({row['animal_id'] for row in rows})
plan_fractions = {plan: float(np.mean([row['correctable'] for row in rows if row['animal_id'] == plan]))
                  for plan in plans}
low = [row for row in rows if row['visible_fraction'] < .1]
summary = {'cases': len(rows), 'correctable_cases': sum(row['correctable'] for row in rows),
           'correctable_case_fraction': float(np.mean([row['correctable'] for row in rows])),
           'correctable_equal_plan_fraction': float(np.mean(list(plan_fractions.values()))),
           'plan_correctable_fractions': plan_fractions,
           'plans_at_least_85_percent': sum(value >= .85 for value in plan_fractions.values()),
           'low_visible_cases': len(low),
           'low_visible_correctable_fraction': float(np.mean([row['correctable'] for row in low])),
           'fixed_span_fail_cases': sum(row['fixed_span_log_ratio'] > .7 for row in rows),
           'centre_bounded_fail_cases': sum(row['nearest_centre_mm'] > 3 for row in rows),
           'gate': bool(np.mean([row['correctable'] for row in rows]) >= .9 and
                        sum(value >= .85 for value in plan_fractions.values()) >= 7 and
                        np.mean([row['correctable'] for row in low]) >= .5),
           'interpretation': 'truth-measured geometric reachability only; selection and biology unknown'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'train_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'prior_completed_sha256': hashlib.sha256((PRIOR / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'gate': summary['gate'], 'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
