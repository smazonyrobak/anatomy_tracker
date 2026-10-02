"""Physical tissue-to-atlas DEV error for the adapted spatial pose candidate."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_real_coordinate_pose_014'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/one_shot_real_coordinate_pose_014_mapped_eval'
steps = (0, 10000, 12000)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 12000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(records) == 256 and len({r['animal_id'] for r in records}) == 8
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction = model.predict(image)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            prior_choice = int(prior.argmax())
            dense_state = model.dense_coordinate_plane(prediction)
            extended = {**prediction, 'state': torch.cat((prediction['state'],
                                                         dense_state[:, None], truth[:, None]), 1)}
            chosen = torch.tensor([[prior_choice // 2, 16, 17]], device='cuda')
            flags = torch.tensor([[prior_choice % 2, 0, reflection]], device='cuda')
            mapped = model.map(extended, offsets, chosen, flags, (side, side), atlas,
                               weights)['centre_surface_ccf_ap_dv_ml_um'][0]
            error = (mapped[:, valid] - reference[valid][None]).norm(dim=-1).mean(-1)
            row = {'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id', 'sha256')},
                   'prior_mapped_um': float(error[0]),
                   'dense_mapped_um': float(error[1]),
                   'true_mapped_um': float(error[2]),
                   'valid_fraction': float(valid.float().mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'eligible': sum(row['step'] == step for row in rows)}), flush=True)

summary = []
for step in steps:
    group = [row for row in rows if row['step'] == step]
    identities = sorted({row['animal_id'] for row in group})
    summary.append({'step': step, 'rows': len(group), 'identities': len(identities),
                    'identity_equal_mean': {name: float(np.mean([
                        np.mean([row[name] for row in group if row['animal_id'] == identity])
                        for identity in identities]))
                        for name in ('prior_mapped_um', 'dense_mapped_um', 'true_mapped_um')}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows),
    'checkpoint_sha256': {str(step): hashlib.sha256(
        (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
