"""Check whether true-mask post-fit agreement separates current top-eight poses."""
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

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/one_shot_on_policy_atlas_rank_009_development_eval'
out = root / 'runs/one_shot_rank009_fitted_evidence_diagnostic'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
evaluated = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
reference = {row['sha256']: row for row in evaluated
             if row['arm'] == 'atlas' and row['set'] == 'synthetic' and row['eligible']}
checkpoint_path = root / 'runs/one_shot_on_policy_atlas_rank_009/atlas/joint_step_08000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
torch.set_num_threads(4)
out.mkdir(parents=True, exist_ok=False)
rows = []

with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        ordered = prior.topk(8).indices
        physical = np.asarray(reference[record['sha256']]['branch_tissue_um']).reshape(-1)
        fit, learned, displacement = [], [], []
        for group in ordered.split(4):
            mapped = model.map(prediction, offsets, (group // 2)[None], (group % 2)[None],
                               (256, 256), atlas, weights)
            fit.extend(model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0].tolist())
            learned.extend(mapped['fit_energy'][0].tolist())
            displacement.extend(mapped['local_displacement_um'][0].square().sum(1)
                                .mean((-2, -1)).sqrt().tolist())
        errors = physical[ordered.cpu().numpy()]
        row = {'animal_id': record['animal_id'], 'section_id': record['section_id'],
               'sha256': record['sha256'], 'valid_pixels': record['valid_pixels'],
               'candidate_branches': ordered.tolist(), 'physical_error_um': errors.tolist(),
               'oracle_mask_fit': fit, 'learned_energy': learned,
               'local_rms_um': displacement,
               'prior_top1_um': float(errors[0]),
               'top8_oracle_um': float(errors.min()),
               'fit_selected_um': float(errors[np.argmin(fit)]),
               'learned_selected_um': float(errors[np.argmin(learned)])}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'scored': len(rows)}), flush=True)

identities = sorted({row['animal_id'] for row in rows})
summary = {'scored': len(rows), 'synthetic_identities': len(identities),
           'identity_equal_mean_mm': {key: float(np.mean([np.mean([
               row[key] for row in rows if row['animal_id'] == identity])
               for identity in identities]) / 1000)
               for key in ('prior_top1_um', 'top8_oracle_um', 'fit_selected_um',
                           'learned_selected_um')}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'evaluation_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'uses_oracle_synthetic_valid_mask': True, 'deployable': False}, indent=2))
print(json.dumps(summary), flush=True)
