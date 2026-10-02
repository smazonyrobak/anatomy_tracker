"""Frozen unknown-plane lookup of observed patches in an atlas-only orientation bank."""
import hashlib
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

from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches

bank_dir = root / 'runs/atlas_global_patch_030'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/atlas_global_patch_030_development_eval'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

with (bank_dir / 'completed.json').open() as stream:
    bank_completion = json.load(stream)
for name, expected in bank_completion['files_sha256'].items():
    with (bank_dir / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
with (bank_dir / 'config.json').open() as stream:
    bank_config = json.load(stream)
with checkpoint.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == bank_config['checkpoint_sha256']
positions = np.load(bank_dir / 'positions_um.npy', allow_pickle=False)
bank_array = np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)
assert bank_array.shape == (2061824, 128) and len(positions) == 4027
bank = torch.from_numpy(bank_array).cuda()
positions_gpu = torch.from_numpy(positions).cuda()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = AtlasOrientedPatch025().cuda().eval()
model.load_state_dict(saved['model'], strict=True)
del saved

records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185
out.mkdir(parents=True, exist_ok=False)
config = {'bank_completed_sha256': hashlib.sha256((bank_dir / 'completed.json').read_bytes()).hexdigest(),
          'panel_completed_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
          'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
          'checkpoint_sha256': bank_config['checkpoint_sha256'],
          'query_points_per_section': 32, 'query_selection': 'fixed seeded GT-valid; diagnostic only',
          'query_parities': ['original', 'horizontally_flipped'],
          'retrieval_top_k': 16, 'distance_threshold_um': 500,
          'minimum_three_match_separation_um': 1000,
          'gate': {'point_recall16_500um': .10, 'three_separated_section_fraction': .20,
                   'minimum_mode_relative_point_recall': .5},
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('evaluate_atlas_global_patch_030.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        with path.open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            selected = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_xy = np.stack((selected // 256, selected % 256), -1)
            truth = arrays['target_centre_um'][query_xy[:, 0], query_xy[:, 1]].copy()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                torch.from_numpy(query_xy.copy()).cuda().float())
        query = torch.stack((query, query.flip(-1)), 1).flatten(0, 1)
        descriptor = F.normalize(model.shared(model.image_stem(query)), dim=-1)
        score = descriptor.half() @ bank.T
        value, flat = score.reshape(32, -1).topk(16, -1)
        parity = flat // len(bank)
        bank_index = flat % len(bank)
        position_index = bank_index // 512
        error = (positions_gpu[position_index] - torch.from_numpy(truth).cuda()[:, None]).norm(dim=-1)
        error_np = error.cpu().numpy()
        matches = np.flatnonzero(error_np.min(axis=1) < 500)
        separation = np.linalg.norm(truth[matches, None] - truth[None, matches], axis=-1)
        triple = False
        for first in range(len(matches)):
            for second in range(first + 1, len(matches)):
                if separation[first, second] > 1000 and np.any(
                    (separation[first, second + 1:] > 1000) &
                    (separation[second, second + 1:] > 1000)):
                    triple = True
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(query_pixel_yx=query_xy.tolist(), true_ccf_um=truth.tolist(),
                   top16_bank_index=bank_index.cpu().tolist(), top16_parity=parity.cpu().tolist(),
                   top16_score=value.float().cpu().tolist(), top16_distance_um=error_np.tolist(),
                   three_separated_matches=triple)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 25 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records),
                              'seconds': time.perf_counter() - started}), flush=True)

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
point_recall = float(np.mean([np.mean([np.mean(np.min(row['top16_distance_um'], axis=1) < 500)
    for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
section_triples = float(np.mean([np.mean([row['three_separated_matches']
    for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
mode_recall = {mode: float(np.mean([np.mean([np.mean(np.min(row['top16_distance_um'], axis=1) < 500)
    for row in rows if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
    for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                  row['appearance_mode'] == mode for row in rows)]))
    for mode in ('raw', 'exact_black', 'imperfect_brush')}
gate = point_recall >= .10 and section_triples >= .20 and min(mode_recall.values()) >= .5 * point_recall
summary = {'sections': len(rows), 'synthetic_subjects': len(subjects),
           'point_recall16_500um': point_recall,
           'point_recall1_500um': float(np.mean([np.mean([np.mean(np.asarray(row['top16_distance_um'])[:, 0] < 500)
               for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])),
           'three_separated_section_fraction': section_triples, 'mode_point_recall16_500um': mode_recall,
           'necessary_gate': gate,
           'scope': 'oracle GT-valid query selection; bank truth-independent; no learned pose or GUI inference',
           'calibrated': False, 'public_benchmark_used': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
completion = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in ('config.json', 'rows.jsonl', 'summary.json')}
(out / 'completed.json').write_text(json.dumps({'files_sha256': completion, 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', **summary, 'seconds': time.perf_counter() - started}), flush=True)
