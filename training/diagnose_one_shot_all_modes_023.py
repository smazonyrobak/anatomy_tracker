"""Frozen 019 all-32 versus prior-top-eight rigid pose capture."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
out = root / 'runs/one_shot_all_modes_023'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 18000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved
flags = torch.tensor([0, 1], device='cuda')
out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        choice = prior.topk(8).indices
        index = valid.flatten().nonzero().flatten()
        chart = torch.stack((index.remainder(256), index.div(256, rounding_mode='floor')), -1).float() / 256
        target = reference.reshape(-1, 3)[index]
        state = prediction['state'][0]
        centre, frame, basis = full_frame_state_to_components(state)
        axes = frame[:, :, :2] @ basis
        chart = chart[None, None].expand(16, 2, -1, -1).clone()
        chart[:, 1, :, 0] = 255 / 256 - chart[:, 1, :, 0]
        rigid = centre[:, None, None] + torch.einsum('mij,mrnj->mrni', axes, chart - .5)
        physical = (rigid - target[None, None]).norm(dim=-1).mean(-1).flatten()
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                           'section_id', 'sha256')}
        row.update(top8=choice.tolist(), prior=prior.tolist(), rigid_um=physical.tolist(),
                   best8_um=float(physical[choice].min()), best32_um=float(physical.min()))
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

animals = sorted({row['animal_id'] for row in rows})
summary = {'cases': len(rows), 'identities': len(animals)}
for name in ('best8_um', 'best32_um'):
    values = np.asarray([row[name] for row in rows])
    summary[name] = {'case_mean': float(values.mean()),
                     'identity_equal_mean': float(np.mean([
                         np.mean([row[name] for row in rows if row['animal_id'] == animal])
                         for animal in animals])),
                     'below_250_um': float((values < 250).mean()),
                     'below_500_um': float((values < 500).mean()),
                     'below_1000_um': float((values < 1000).mean())}
summary['all32_improves_over_top8_fraction'] = float(np.mean([
    row['best32_um'] + 1e-4 < row['best8_um'] for row in rows]))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
