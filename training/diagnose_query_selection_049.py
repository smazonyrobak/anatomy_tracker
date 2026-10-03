"""Measure whether pose-fitting query points actually land on surviving tissue."""
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.pose_feedback_global_041 import PoseFeedbackGlobal041

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [record for record in records if record['eligible']]
torch.set_num_threads(4)
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                             vector_refinement=True, candidate_ranking=True,
                             fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(root / 'runs/one_shot_exposure_019/joint_step_18000.pt',
                                map_location='cpu', weights_only=True)['model'])
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt',
                                  map_location='cpu', weights_only=True)['head'])
positions = torch.arange(256, device='cuda').reshape(16, 16)
rows = []
with torch.inference_mode():
    for record in eligible:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = pose.predict(image)
        source = torch.cat((F.adaptive_avg_pool2d(prediction['feature'], 16),
                            F.adaptive_avg_pool2d(image, 16)), 1)
        visibility = coarse.visibility(coarse.query(source)).flatten()
        tissue = valid[8::16, 8::16].flatten()
        quadrant = torch.cat([positions[y:y + 8, x:x + 8].flatten()[
            visibility[positions[y:y + 8, x:x + 8].flatten()].topk(2).indices]
            for y in (0, 8) for x in (0, 8)])
        global8 = visibility.topk(8).indices
        global16 = visibility.topk(16).indices
        rows.append({'identity': record['synthetic_subject_plan_id'],
                     'appearance': record['appearance_mode'],
                     'valid_cells': int(tissue.sum()),
                     'quadrant_valid': int(tissue[quadrant].sum()),
                     'global8_valid': int(tissue[global8].sum()),
                     'global16_valid': int(tissue[global16].sum())})

identities = sorted({row['identity'] for row in rows})
summary = {'eligible': len(rows), 'identities': len(identities)}
for field in ('valid_cells', 'quadrant_valid', 'global8_valid', 'global16_valid'):
    summary[field + '_identity_equal_mean'] = float(np.mean([np.mean([row[field]
        for row in rows if row['identity'] == identity]) for identity in identities]))
for field in ('quadrant_valid', 'global8_valid'):
    summary[field + '_at_least_3_fraction'] = float(np.mean([row[field] >= 3 for row in rows]))
    summary[field + '_at_least_6_fraction'] = float(np.mean([row[field] >= 6 for row in rows]))
print(json.dumps(summary, indent=2))
