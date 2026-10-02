"""Measure whether the direct pose prior puts a useful branch in its top-k set."""
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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/one_shot_on_policy_atlas_rank_009_development_eval'
out = root / 'runs/one_shot_rank009_prior_capture_diagnostic'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
evaluated = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
reference = {(row['arm'], row['sha256']): row for row in evaluated
             if row['set'] == 'synthetic' and row['eligible']}
out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
rows = []

with torch.inference_mode():
    for arm in ('direct', 'atlas'):
        path = root / f'runs/one_shot_on_policy_atlas_rank_009/{arm}/joint_step_08000.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                       candidate_ranking=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        for record in records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            prediction = model.predict(image)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            ordered = prior.argsort(descending=True).cpu().numpy()
            error = np.asarray(reference[arm, record['sha256']]['branch_tissue_um']).reshape(-1)
            rows.append({'arm': arm, 'animal_id': record['animal_id'],
                         'section_id': record['section_id'], 'sha256': record['sha256'],
                         **{f'top_{k}_oracle_um': float(error[ordered[:k]].min())
                            for k in (1, 2, 4, 8, 16, 32)}})
        del model, checkpoint

summary = []
for arm in ('direct', 'atlas'):
    selected = [row for row in rows if row['arm'] == arm]
    animals = sorted({row['animal_id'] for row in selected})
    summary.append({'arm': arm, 'scored_sections': len(selected),
                    'synthetic_identities': len(animals),
                    'identity_equal_mean_mm': {f'top_{k}': float(np.mean([
                        np.mean([row[f'top_{k}_oracle_um'] for row in selected
                                 if row['animal_id'] == animal]) for animal in animals]) / 1000)
                        for k in (1, 2, 4, 8, 16, 32)}})
(out / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'evaluation_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps(summary), flush=True)
