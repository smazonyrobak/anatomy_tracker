"""Can atlas-conditioned fitting rank actual predicted planes on frozen synthetic development?"""
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
import torch.nn.functional as F
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

RUN = ROOT / 'runs/one_shot_joint_physical_pose_002'
PANEL = ROOT / 'data/one_shot_native256_synthetic_dev_001'
EVALUATION = ROOT / 'runs/one_shot_joint_physical_pose_002_development_eval'
OUT = ROOT / 'runs/one_shot_native256_candidate_fit_001'
CHECKPOINT = RUN / 'joint_step_08000.pt'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 8000
assert json.loads((EVALUATION / 'completed.json').read_text())['rows'] == 960
reference = [row for row in map(json.loads, (EVALUATION / 'rows.jsonl').read_text().splitlines())
             if row['step'] == 8000 and row['set'] == 'synthetic' and row['eligible']]
records = {row['panel_physical_section_id']: row for row in
           map(json.loads, (PANEL / 'records.jsonl').read_text().splitlines())}
assert len(reference) == 90 and len(records) == 128
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(atlas_conditioning=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
OUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            'panel_records_sha256': hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest(),
            'geometry_rows_sha256': hashlib.sha256((EVALUATION / 'rows.jsonl').read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': 90, 'branches': 16, 'fit_mask': 'synthetic known-valid tissue (optimistic diagnostic only)',
            'selection': 'prior argmax versus raw fitted atlas-mismatch argmin; no threshold tuned',
            'scope': 'four held-out synthetic deformation plans, not biological animals or calibration',
            'public_benchmark_used': False}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w') as stream:
    for index, truth in enumerate(reference):
        record = records[truth['panel_physical_section_id']]
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)
        assert torch.allclose(prior.cpu(), torch.tensor(truth['branch_log_mass']), atol=1e-4)
        fit = torch.empty(8, 2, device='cuda')
        local = torch.empty(8, 2, device='cuda')
        for mode in range(8):
            mapped = model.map(prediction, offsets,
                               torch.tensor([[mode, mode]], device='cuda'),
                               torch.tensor([[0, 1]], device='cuda'),
                               (256, 256), atlas, weights)
            fit[mode] = model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0]
            local[mode] = ((mapped['local_displacement_um'][0].square().sum(1).sqrt()
                            * valid[0]).sum((-2, -1)) / valid[0].sum())
        teacher = {**prediction, 'state': prediction['state'].clone()}
        teacher['state'][:, 0] = state
        mapped = model.map(teacher, offsets, torch.zeros((1, 1), device='cuda', dtype=torch.long),
                           torch.tensor([[reflection]], device='cuda'), (256, 256), atlas, weights)
        true_fit = float(model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0, 0])
        errors = np.asarray(truth['branch_tissue_um']).reshape(-1)
        scores = fit.flatten().cpu().numpy()
        prior_choice = int(prior.flatten().argmax())
        fit_choice = int(fit.flatten().argmin())
        oracle_choice = int(errors.argmin())
        row = {'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
               'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
               'panel_physical_section_id': record['panel_physical_section_id'],
               'appearance_mode': record['appearance_mode'], 'valid_pixels': record['valid_pixels'],
               'input_sha256': record['sha256'], 'prior_choice': prior_choice,
               'fit_choice': fit_choice, 'oracle_choice': oracle_choice,
               'prior_tissue_um': float(errors[prior_choice]),
               'fit_tissue_um': float(errors[fit_choice]),
               'oracle_tissue_um': float(errors[oracle_choice]),
               'prior_fit': float(scores[prior_choice]), 'minimum_fit': float(scores[fit_choice]),
               'oracle_fit': float(scores[oracle_choice]), 'true_pose_fit': true_fit,
               'fit_error_rank_correlation': float(spearmanr(scores, errors).statistic),
               'branch_fit': fit.cpu().tolist(), 'branch_local_magnitude_um': local.cpu().tolist(),
               'branch_tissue_um': truth['branch_tissue_um']}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if (index + 1) % 15 == 0:
            stream.flush()
            print(json.dumps({'cases': index + 1, 'prior_um': float(np.mean([r['prior_tissue_um'] for r in rows])),
                              'fit_um': float(np.mean([r['fit_tissue_um'] for r in rows]))}), flush=True)
animals = sorted({row['animal_id'] for row in rows})
summary = {'cases': len(rows), 'identities': len(animals),
           'identity_equal_mean': {key: float(np.mean([np.mean([row[key] for row in rows
                                            if row['animal_id'] == animal]) for animal in animals]))
                                   for key in ('prior_tissue_um', 'fit_tissue_um', 'oracle_tissue_um',
                                               'prior_fit', 'minimum_fit', 'oracle_fit', 'true_pose_fit',
                                               'fit_error_rank_correlation')},
           'fit_beats_prior': sum(row['fit_tissue_um'] < row['prior_tissue_um'] for row in rows),
           'fit_worse_than_prior': sum(row['fit_tissue_um'] > row['prior_tissue_um'] for row in rows),
           'oracle_fit_rank_mean': float(np.mean([np.argsort(np.asarray(row['branch_fit']).reshape(-1)).tolist().index(
               row['oracle_choice']) + 1 for row in rows]))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'cases': len(rows), 'checkpoint_sha256': protocol['checkpoint_sha256'],
    'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'source_sha256': protocol['source_sha256'], 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
