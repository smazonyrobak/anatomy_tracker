"""Test whether atlas fitting can rank the 16-mode model's actual proposals."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_pose_capture_003'
evaluation = root / 'runs/one_shot_pose_capture_003_development_eval'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
out = root / 'runs/one_shot_pose_capture_003_fit_diagnostic'
checkpoint_path = run / 'joint_step_40000.pt'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 40000
assert json.loads((evaluation / 'completed.json').read_text())['rows'] == 1728
truth_rows = [row for row in map(json.loads, (evaluation / 'rows.jsonl').open())
              if row['step'] == 40000 and row['set'] == 'synthetic' and row['eligible']]
records = {row['panel_physical_section_id']: row for row in
           map(json.loads, (panel / 'records.jsonl').open())}
assert len(truth_rows) == 90 and len(records) == 128
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint

out.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'geometry_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
            'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': 90, 'branches': 32,
            'fit_mask': 'known synthetic tissue; optimistic diagnostic unavailable at deployment',
            'selection': 'image-prior argmax, raw atlas-fit argmin, frozen match-energy argmin; no tuned fusion',
            'scope': 'held-out synthetic deformation plans, not independent biological animals',
            'public_benchmark_used': False}
(out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for index, truth in enumerate(truth_rows):
        record = records[truth['panel_physical_section_id']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
        prediction = model.predict(image)
        prior = prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)
        assert torch.allclose(prior.cpu(), torch.tensor(truth['branch_log_mass']), atol=1e-4)
        fit = torch.empty(16, 2, device='cuda')
        energy = torch.empty(16, 2, device='cuda')
        local = torch.empty(16, 2, device='cuda')
        for mode in range(16):
            mapped = model.map(prediction, offsets,
                               torch.tensor([[mode, mode]], device='cuda'),
                               torch.tensor([[0, 1]], device='cuda'),
                               (256, 256), atlas, weights)
            fit[mode] = model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0]
            energy[mode] = mapped['fit_energy'][0]
            local[mode] = ((mapped['local_displacement_um'][0].square().sum(1).sqrt()
                            * valid[0]).sum((-2, -1)) / valid[0].sum())
        teacher = {**prediction, 'state': prediction['state'].clone()}
        teacher['state'][:, 0] = state
        mapped = model.map(teacher, offsets, torch.zeros((1, 1), device='cuda', dtype=torch.long),
                           torch.tensor([[reflection]], device='cuda'), (256, 256), atlas, weights)
        true_fit = float(model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0, 0])
        errors = np.asarray(truth['branch_tissue_um']).reshape(-1)
        raw = fit.flatten().cpu().numpy()
        learned = energy.flatten().cpu().numpy()
        prior_choice = int(prior.flatten().argmax())
        fit_choice = int(raw.argmin())
        energy_choice = int(learned.argmin())
        oracle_choice = int(errors.argmin())
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                            'section_id', 'panel_physical_section_id',
                                            'appearance_mode', 'valid_pixels')}
        row.update(prior_choice=prior_choice, fit_choice=fit_choice,
                   energy_choice=energy_choice, oracle_choice=oracle_choice,
                   prior_tissue_um=float(errors[prior_choice]),
                   fit_tissue_um=float(errors[fit_choice]),
                   energy_tissue_um=float(errors[energy_choice]),
                   oracle_tissue_um=float(errors[oracle_choice]),
                   true_pose_fit=true_fit,
                   fit_error_rank_correlation=float(spearmanr(raw, errors).statistic),
                   energy_error_rank_correlation=float(spearmanr(learned, errors).statistic),
                   branch_fit=fit.cpu().tolist(), branch_energy=energy.cpu().tolist(),
                   branch_local_magnitude_um=local.cpu().tolist(),
                   branch_tissue_um=truth['branch_tissue_um'])
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if (index + 1) % 15 == 0:
            stream.flush()
            print(json.dumps({'cases': index + 1}), flush=True)
animals = sorted({row['animal_id'] for row in rows})
metrics = ('prior_tissue_um', 'fit_tissue_um', 'energy_tissue_um', 'oracle_tissue_um',
           'true_pose_fit', 'fit_error_rank_correlation', 'energy_error_rank_correlation')
summary = {'cases': len(rows), 'identities': len(animals),
           'identity_equal_mean': {key: float(np.mean([np.mean([row[key] for row in rows
                                      if row['animal_id'] == animal]) for animal in animals]))
                                   for key in metrics},
           'fit_beats_prior': sum(row['fit_tissue_um'] < row['prior_tissue_um'] for row in rows),
           'fit_worse_than_prior': sum(row['fit_tissue_um'] > row['prior_tissue_um'] for row in rows),
           'energy_beats_prior': sum(row['energy_tissue_um'] < row['prior_tissue_um'] for row in rows),
           'energy_worse_than_prior': sum(row['energy_tissue_um'] > row['prior_tissue_um'] for row in rows)}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'cases': len(rows),
    'checkpoint_sha256': protocol['checkpoint_sha256'],
    'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'source_sha256': protocol['source_sha256'], 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
