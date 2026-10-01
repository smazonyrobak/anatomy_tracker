"""Test whether the learned match energy recognizes the true plane on held-out synthetic sections."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_native256_synthetic_dev_001'
evaluation = root / 'runs/one_shot_candidate_fit_energy_001_development_eval'
run = root / 'runs/one_shot_candidate_fit_energy_001'
out = root / 'runs/one_shot_candidate_energy_truth_001'
steps = (1000, 4000)
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
previous = {(row['step'], row['panel_physical_section_id']): row for row in
            map(json.loads, (evaluation / 'rows.jsonl').open()) if row['set'] == 'synthetic' and row['eligible']}
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
out.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint_path = run / f'joint_step_{step:05d}.pt'
        checkpoint_hashes[str(step)] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = OneShotJointSliceModel(atlas_conditioning=True, fit_quality=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = torch.tensor([[int(arrays['reflection'])]], device='cuda')
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction = model.predict(image)
            states = prediction['state'].clone()
            states[:, 0] = state
            mapped = model.map({**prediction, 'state': states}, offsets,
                               torch.zeros((1, 1), device='cuda', dtype=torch.long),
                               reflection, (256, 256), atlas, weights)
            true_energy = float(mapped['fit_energy'][0, 0])
            predicted = previous[(step, record['panel_physical_section_id'])]
            predicted_energies = np.asarray(predicted['branch_energy']).ravel()
            row = {'step': step, 'panel_physical_section_id': record['panel_physical_section_id'],
                   'animal_id': record['animal_id'], 'true_energy': true_energy,
                   'minimum_predicted_energy': float(predicted_energies.min()),
                   'prior_predicted_energy': float(predicted_energies[predicted['prior_choice']]),
                   'true_below_all_predicted': bool(true_energy < predicted_energies.min())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        print(json.dumps({'step': step, 'rows': len(rows)}), flush=True)
        del model
summary = [{'step': step, 'rows': sum(row['step'] == step for row in rows),
            'true_below_all_predicted': sum(row['true_below_all_predicted'] for row in rows if row['step'] == step),
            'mean_true_energy': float(np.mean([row['true_energy'] for row in rows if row['step'] == step])),
            'mean_minimum_predicted_energy': float(np.mean([row['minimum_predicted_energy']
                                                           for row in rows if row['step'] == step]))}
           for step in steps]
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'public_benchmark_used': False,
}, indent=2))
