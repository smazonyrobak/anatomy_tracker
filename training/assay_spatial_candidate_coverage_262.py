"""Measure truth-only reachability of a fixed, inference-available pose grid."""
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
OUT = ROOT / 'runs/spatial_candidate_coverage_262_dev_001'
DOC = Path('docs/publication/SPATIAL_CANDIDATE_COVERAGE_262_PROTOCOL_20261012.md')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((TRAIN / 'joint_step_03000.pt').read_bytes()).hexdigest() == train_done['checkpoint_sha256']['3000']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [record for record in records if record['eligible'] and record['valid_pixels'] > 0]
assert len(eligible) == 227 and len({record['animal_id'] for record in eligible}) == 8
assert len({record['provenance']['physical_section_id'] for record in eligible}) == 227

model = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
                              dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
saved = torch.load(TRAIN / 'joint_step_03000.pt', map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved
anchor_model = OneShotJointSliceModel(modes=0, normal_anchor_count=64)
anchors = anchor_model.normal_anchor_frames[:, :, 2].cuda()
del anchor_model

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        predicted = model.dense_coordinate_plane(model.predict(image), source_side=192)
        predicted_centre, _, predicted_basis = full_frame_state_to_components(predicted)
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
        normal_deg = torch.rad2deg(torch.acos((anchors @ true_frame[0, :, 2]).abs().clamp_max(1)))
        centres = predicted_centre[0, None, None] + anchors[:, None] * torch.tensor(
            (-2000., 0., 2000.), device='cuda')[None, :, None]
        centre_mm = torch.linalg.vector_norm(centres - true_centre[0], dim=-1) / 1000
        predicted_span = torch.linalg.svdvals(predicted_basis)[0].sort().values
        true_span = torch.linalg.svdvals(true_basis)[0].sort().values
        span_log = (torch.log(predicted_span[None] * torch.tensor((1., 1.8), device='cuda')[:, None])
                    - torch.log(true_span[None])).abs().amax(-1)
        jointly = ((normal_deg[:, None, None] <= 15) & (centre_mm[:, :, None] <= 3)
                   & (span_log[None, None, :] <= .7))
        row = {'physical_section_id': record['provenance']['physical_section_id'],
               'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
               'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
               'appearance_mode': record['appearance_mode'],
               'visible_fraction': record['valid_pixels'] / 192**2,
               'panel_file_sha256': record['sha256'],
               'correctable': bool(jointly.any()),
               'normal_within_15': bool((normal_deg <= 15).any()),
               'centre_within_3': bool((centre_mm <= 3).any()),
               'span_within_log_0p7': bool((span_log <= .7).any()),
               'nearest_normal_deg': float(normal_deg.min()),
               'nearest_centre_mm': float(centre_mm.min()),
               'nearest_span_log_ratio': float(span_log.min()),
               'predicted_minor_span_mm': float(predicted_span[0] / 1000),
               'true_minor_span_mm': float(true_span[0] / 1000)}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

plans = sorted({row['animal_id'] for row in rows})
plan_fractions = {plan: float(np.mean([row['correctable'] for row in rows if row['animal_id'] == plan]))
                  for plan in plans}
summary = {'cases': len(rows), 'plans': len(plans),
           'correctable_cases': sum(row['correctable'] for row in rows),
           'correctable_case_fraction': float(np.mean([row['correctable'] for row in rows])),
           'correctable_equal_plan_fraction': float(np.mean(list(plan_fractions.values()))),
           'plan_correctable_fractions': plan_fractions,
           'plans_at_least_60_percent': sum(value >= .6 for value in plan_fractions.values()),
           'component_fails': {key: sum(not row[key] for row in rows) for key in
                               ('normal_within_15', 'centre_within_3', 'span_within_log_0p7')},
           'median_nearest_normal_deg': float(np.median([row['nearest_normal_deg'] for row in rows])),
           'median_nearest_centre_mm': float(np.median([row['nearest_centre_mm'] for row in rows])),
           'median_nearest_span_log_ratio': float(np.median([row['nearest_span_log_ratio'] for row in rows])),
           'appearance': {mode: {'cases': sum(row['appearance_mode'] == mode for row in rows),
                                 'correctable_fraction': float(np.mean([row['correctable'] for row in rows
                                                                         if row['appearance_mode'] == mode]))}
                          for mode in sorted({row['appearance_mode'] for row in rows})},
           'low_visibility': {label: {'cases': sum(low <= row['visible_fraction'] < high for row in rows),
                                     'correctable_fraction': float(np.mean([row['correctable'] for row in rows
                                                                             if low <= row['visible_fraction'] < high]))}
                              for label, low, high in (('below_0p1', 0, .1), ('0p1_to_0p3', .1, .3),
                                                       ('0p3_to_1', .3, 1.01))
                              if any(low <= row['visible_fraction'] < high for row in rows)},
           'gate': bool(np.mean([row['correctable'] for row in rows]) >= .7 and
                        sum(value >= .6 for value in plan_fractions.values()) >= 6),
           'interpretation': 'optimistic geometry-only reachability; not blind selection or image evidence'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'train_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'gate': summary['gate'], 'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
