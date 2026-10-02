"""Frozen synthetic-DEV atlas-point retrieval diagnostic; not a final test."""
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.atlas_point_proposal_024 import AtlasPointProposal024, atlas_cubes, observed_patches

run = root / 'runs/atlas_point_proposal_024'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_point_proposal_024_development_eval'
steps = (0, 1000, 2000, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 3000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [record for record in records if record['eligible']]
assert len(records) == 256 and len(eligible) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = F.avg_pool3d(torch.from_numpy(atlas_array)[None].cuda(), 4, 4)
support = (atlas[0, 1].cpu().numpy() > .5)
support_flat = np.flatnonzero(support)
rng = np.random.default_rng(2026102401)
selected = rng.choice(support_flat, 100000, replace=False)
bank_coords = (np.array(np.unravel_index(selected, support.shape)).T.astype('float32') + .375) * 100
bank = torch.from_numpy(bank_coords).cuda()
bank_tree = cKDTree(bank_coords)
out.mkdir(parents=True, exist_ok=False)
np.save(out / 'bank_ccf_ap_dv_ml_um.npy', bank_coords, allow_pickle=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


models, banks, checkpoints = [], [], {}
for step in steps:
    path = run / f'point_step_{step:05d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasPointProposal024().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    models.append(model)
    banks.append(torch.empty((len(bank), 64), device='cuda'))
    checkpoints[str(step)] = sha(path)
del checkpoint
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'bank_ccf_sha256': sha(out / 'bank_ccf_ap_dv_ml_um.npy'),
          'bank_seed': 2026102401, 'bank_positions': len(bank), 'bank_grid_um': 100,
          'query_points_per_section': 32, 'query_selection': 'fixed seeded GT-valid diagnostic; not inference',
          'checkpoints_sha256': checkpoints,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_atlas_point_proposal_024.py',
                             'atlas_point_proposal_024.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
started = time.perf_counter()
with torch.inference_mode():
    for start in range(0, len(bank), 64):
        end = min(start + 64, len(bank))
        cube = atlas_cubes(atlas, bank[start:end])
        for model, embeddings in zip(models, banks):
            embeddings[start:end] = F.normalize(model.atlas(cube), dim=-1)
        if end % 10000 < 64:
            print(json.dumps({'bank_encoded': end, 'seconds': time.perf_counter() - started}), flush=True)
    rows = []
    with (out / 'rows.jsonl').open('w') as stream:
        for number, record in enumerate(eligible, 1):
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                valid = arrays['valid_mask'].reshape(-1)
                rng = np.random.default_rng(int(record['sha256'][:16], 16))
                chosen = rng.choice(np.flatnonzero(valid), 32, replace=False)
                xy = np.stack((chosen // 256, chosen % 256), -1)
                truth = arrays['target_centre_um'][xy[:, 0], xy[:, 1]].copy()
                inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            points = torch.from_numpy(xy.copy()).cuda().float()
            query = observed_patches(inputs, torch.zeros(32, device='cuda', dtype=torch.long), points)
            truth_gpu = torch.from_numpy(truth).cuda()
            nearest = bank_tree.query(truth, k=1)[0]
            for step, model, embeddings in zip(steps, models, banks):
                descriptor = F.normalize(model.image(query), dim=-1)
                top = (descriptor @ embeddings.T).topk(16, -1).indices
                top_distance = (bank[top] - truth_gpu[:, None]).norm(dim=-1).min(-1).values
                good = (top_distance < 500).nonzero().flatten().tolist()
                triple = any(all((truth_gpu[a] - truth_gpu[b]).norm() >= 1000 and
                                 (points[a] - points[b]).norm() >= 16 for a, b in
                                 itertools.combinations(indices, 2))
                             for indices in itertools.combinations(good, 3))
                row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                           'section_id', 'synthetic_subject_plan_id', 'sha256')}
                row.update(step=step, query_pixel_yx=xy.tolist(), true_ccf_um=truth.tolist(),
                           top16_bank_indices=top.cpu().tolist(),
                           min_top16_um=top_distance.cpu().tolist(),
                           nearest_bank_um=nearest.tolist(), three_separated=bool(triple))
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
            if number % 40 == 0:
                stream.flush()
                print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
summaries = []
for step in steps:
    selected_rows = [row for row in rows if row['step'] == step]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in selected_rows})
    by_subject = {subject: [row for row in selected_rows
                            if row['synthetic_subject_plan_id'] == subject] for subject in subjects}
    metrics = {'step': step, 'sections': len(selected_rows), 'synthetic_subjects': len(subjects)}
    for threshold in (250, 500, 1000):
        metrics[f'recall16_{threshold}um'] = float(np.mean([
            np.mean([np.mean(np.array(row['min_top16_um']) < threshold) for row in grouped])
            for grouped in by_subject.values()]))
        metrics[f'bank_coverage_{threshold}um'] = float(np.mean([
            np.mean([np.mean(np.array(row['nearest_bank_um']) < threshold) for row in grouped])
            for grouped in by_subject.values()]))
    metrics['three_separated_fraction'] = float(np.mean([
        np.mean([row['three_separated'] for row in grouped]) for grouped in by_subject.values()]))
    summaries.append(metrics)
(out / 'summary.json').write_text(json.dumps({'rows': len(rows), 'checkpoints': summaries,
    'promising_gate': any(row['recall16_500um'] >= .20 and
                          row['three_separated_fraction'] >= .50 for row in summaries[1:]),
    'scope': 'GT-valid point diagnostic on held-out synthetic deformation plans, not deployable pose accuracy',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'),
    'bank_sha256': sha(out / 'bank_ccf_ap_dv_ml_um.npy'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'summaries': summaries,
                  'seconds': time.perf_counter() - started}), flush=True)
