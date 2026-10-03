"""Frozen 030-bank lookup using the updated image query encoders."""
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

from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches

train = root / 'runs/frozen_bank_global_query_055_pilot'
bank_dir = root / 'runs/atlas_global_patch_030'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/frozen_bank_global_query_055_development_eval'
steps = (0, 500, 1500)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
receipt = json.loads((train / 'completed.json').read_text())
assert receipt['batches'] == 1500 and receipt['accepted_synthetic'] == 1500
bank_receipt = json.loads((bank_dir / 'completed.json').read_text())
for name, expected in bank_receipt['files_sha256'].items():
    assert sha(bank_dir / name) == expected
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
assert bank.shape == (2061824, 128) and positions.shape == (4027, 3)
models = []
for step in steps:
    checkpoint = torch.load(train / f'query_step_{step:05d}.pt', map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasOrientedPatch025().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    models.append(model)
del checkpoint
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185 and len({row['animal_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
config = {'steps': steps, 'training_completed_sha256': sha(train / 'completed.json'),
          'bank_completed_sha256': sha(bank_dir / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'checkpoint_sha256': {str(step): sha(train / f'query_step_{step:05d}.pt') for step in steps},
          'query_points_per_section': 32, 'query_selection': 'fixed seeded GT-valid diagnostic, not inference',
          'query_parities': ['original', 'horizontally_flipped'], 'retrieval_top_k': 16,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_frozen_bank_global_query_055.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_yx = np.stack((chosen // 256, chosen % 256), -1)
            truth = arrays['target_centre_um'][query_yx[:, 0], query_yx[:, 1]].copy()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                torch.from_numpy(query_yx.copy()).cuda().float())
        query = torch.stack((query, query.flip(-1)), 1).flatten(0, 1)
        truth_gpu = torch.from_numpy(truth).cuda()
        for step, model in zip(steps, models):
            descriptor = F.normalize(model.shared(model.image_stem(query)), dim=-1)
            score = descriptor.half() @ bank.T
            value, flat = score.reshape(32, -1).topk(16, -1)
            parity = flat // len(bank)
            bank_index = flat % len(bank)
            distance = (positions[bank_index // 512] - truth_gpu[:, None]).norm(dim=-1)
            error = distance.cpu().numpy()
            matched = np.flatnonzero(error.min(axis=1) < 500)
            separated = np.linalg.norm(truth[matched, None] - truth[None, matched], axis=-1)
            triple = False
            for first in range(len(matched)):
                for second in range(first + 1, len(matched)):
                    if separated[first, second] > 1000 and np.any(
                        (separated[first, second + 1:] > 1000) &
                        (separated[second, second + 1:] > 1000)):
                        triple = True
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                               'section_id', 'synthetic_subject_plan_id',
                                               'appearance_mode', 'sha256')}
            row.update(step=step, query_pixel_yx=query_yx.tolist(), true_ccf_um=truth.tolist(),
                       top16_bank_index=bank_index.cpu().tolist(), top16_parity=parity.cpu().tolist(),
                       top16_score=value.float().cpu().tolist(), top16_distance_um=error.tolist(),
                       three_separated_matches=triple)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records)}), flush=True)

summary = {'sections': 185, 'synthetic_subjects': 8}
for step in steps:
    group = [row for row in rows if row['step'] == step]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in group})
    recall1 = float(np.mean([np.mean([np.mean(np.asarray(row['top16_distance_um'])[:, 0] < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    recall16 = float(np.mean([np.mean([np.mean(np.asarray(row['top16_distance_um']).min(axis=1) < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    triple = float(np.mean([np.mean([row['three_separated_matches'] for row in group
        if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    modes = {mode: float(np.mean([np.mean([np.mean(np.asarray(row['top16_distance_um']).min(axis=1) < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in group)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summary[str(step)] = {'point_recall1_500um': recall1, 'point_recall16_500um': recall16,
                          'three_separated_section_fraction': triple,
                          'mode_point_recall16_500um': modes,
                          'global_progress_gate': bool(step > 0 and recall1 >= .03 and recall16 >= .25 and
                                                       min(modes.values()) >= .5 * recall16)}
summary['any_global_progress_gate'] = any(summary[str(step)]['global_progress_gate'] for step in steps[1:])
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'any_global_progress_gate':
                  summary['any_global_progress_gate']}), flush=True)
