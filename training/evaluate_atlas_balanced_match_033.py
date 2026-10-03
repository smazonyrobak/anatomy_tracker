"""Fixed-panel evaluation of the continued contextual match/pose model."""
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
from training.atlas_learned_match_032 import AtlasLearnedMatch032

bank_dir = root / 'runs/atlas_global_patch_030'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
descriptor_checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
run = root / 'runs/atlas_balanced_match_033'
out = root / 'runs/atlas_balanced_match_033_development_eval'
steps = (3000, 5000, 7000, 9000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

assert json.loads((run / 'completed.json').read_text())['last_batch'] == 9000
assert hashlib.sha256((bank_dir / 'completed.json').read_bytes()).hexdigest() == json.loads((run / 'config.json').read_text())['atlas_bank_completed_sha256']
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
axis_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
axis_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
saved = torch.load(descriptor_checkpoint, map_location='cpu', weights_only=True)
descriptor = AtlasOrientedPatch025().cuda().eval()
descriptor.load_state_dict(saved['model'], strict=True)
del saved
models, checkpoints = [], {}
for step in steps:
    path = run / f'match_step_{step:05d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasLearnedMatch032().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    models.append(model)
    checkpoints[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
    del checkpoint

records = {record['sha256']: record for record in map(json.loads, (panel / 'records.jsonl').open())
           if record['eligible']}
retrieved_rows = list(map(json.loads, (retrieval / 'rows.jsonl').open()))
assert len(records) == len(retrieved_rows) == 185
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
          'retrieval_completed_sha256': hashlib.sha256((retrieval / 'completed.json').read_bytes()).hexdigest(),
          'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
          'checkpoint_sha256': checkpoints, 'steps': steps,
          'query_selection': 'fixed synthetic GT-valid positions from 030; diagnostic only',
          'gate': {'available_match_recall500_at_least': .5,
                   'unavailable_false_match_rate_at_most': .2,
                   'weighted_fit_rigid_mean_um_below': 2500},
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('evaluate_atlas_balanced_match_033.py',
                                         'atlas_learned_match_032.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, retrieved in enumerate(retrieved_rows, 1):
        record = records[retrieved['sha256']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            pixel = torch.from_numpy(np.asarray(retrieved['query_pixel_yx'])[None].copy()).cuda()
            truth_query = torch.from_numpy(np.asarray(retrieved['true_ccf_um'])[None].copy()).cuda()
            valid = arrays['valid_mask'].copy()
            truth_surface = arrays['target_centre_um'].copy()
        query = extract_patches(inputs, torch.zeros(32, device='cuda', dtype=torch.long), pixel[0].float())
        query = torch.stack((query, query.flip(-1)), 1).flatten(0, 1)
        feature = F.normalize(descriptor.shared(descriptor.image_stem(query)), dim=-1).reshape(1, 32, 2, 128)
        index = torch.tensor(retrieved['top16_bank_index'], device='cuda')[None]
        parity = torch.tensor(retrieved['top16_parity'], device='cuda')[None]
        image_feature = torch.gather(feature, 2, parity[..., None].expand(-1, -1, -1, 128))
        atlas_feature = bank[index].float()
        score = torch.tensor(retrieved['top16_score'], device='cuda')[None]
        location = positions[index // 512]
        sign = 1 - 2 * parity
        u = axis_u[index % 512] * sign[..., None]
        v = axis_v[index % 512]
        rank = torch.arange(16, device='cuda')[None, None].expand(1, 32, -1) / 15
        distance = (location - truth_query[:, :, None]).norm(dim=-1)
        available = distance.min(-1).values < 500
        y, x = np.nonzero(valid)
        for step, model in zip(steps, models):
            logits = model(image_feature, atlas_feature, score, pixel.float(), location, u, v, rank)
            choice = logits.argmax(-1)
            chosen_distance = distance.gather(-1, choice.clamp_max(15)[..., None]).squeeze(-1)
            correct = (choice < 16) & (chosen_distance < 500)
            false_match = (choice < 16) & ~available
            grouped_choice = torch.where(torch.logsumexp(logits[..., :16], -1) > logits[..., 16],
                                         logits[..., :16].argmax(-1), 16)
            grouped_distance = distance.gather(-1, grouped_choice.clamp_max(15)[..., None]).squeeze(-1)
            grouped_correct = (grouped_choice < 16) & (grouped_distance < 500)
            grouped_false = (grouped_choice < 16) & ~available
            coefficients = model.fit_plane(logits, pixel.float(), location)[0].cpu().numpy()
            mapped = (coefficients[0] + (x[:, None] / 256 - .5) * coefficients[1]
                      + (y[:, None] / 256 - .5) * coefficients[2])
            rigid_error = float(np.linalg.norm(mapped - truth_surface[y, x], axis=-1).mean())
            row = {key: retrieved[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
            row.update(step=step, available_count=int(available.sum()),
                       correct_available_count=int((correct & available).sum()),
                       false_unavailable_count=int(false_match.sum()),
                       grouped_correct_available_count=int((grouped_correct & available).sum()),
                       grouped_false_unavailable_count=int(grouped_false.sum()),
                       unavailable_count=int((~available).sum()),
                       selected_match_rank=choice[0].cpu().tolist(),
                       grouped_selected_match_rank=grouped_choice[0].cpu().tolist(),
                       selected_distance_um=chosen_distance[0].cpu().tolist(),
                       match_logits=logits[0].cpu().tolist(),
                       affine_coefficient_normalized_xy_um=coefficients.tolist(),
                       rigid_mean_error_um=rigid_error)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(retrieved_rows),
                              'seconds': time.perf_counter() - started}), flush=True)

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summaries = []
for step in steps:
    selected = [row for row in rows if row['step'] == step]
    recall = float(np.mean([sum(row['correct_available_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['available_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects]))
    false_rate = float(np.mean([sum(row['false_unavailable_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['unavailable_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects]))
    mean_error = float(np.mean([np.mean([row['rigid_mean_error_um'] for row in selected
        if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    grouped_recall = float(np.mean([sum(row['grouped_correct_available_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['available_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects]))
    grouped_false_rate = float(np.mean([sum(row['grouped_false_unavailable_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['unavailable_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects]))
    summaries.append({'step': step, 'sections': len(selected),
                      'available_match_recall500': recall,
                      'unavailable_false_match_rate': false_rate,
                      'grouped_available_match_recall500_secondary': grouped_recall,
                      'grouped_unavailable_false_match_rate_secondary': grouped_false_rate,
                      'weighted_fit_rigid_mean_um': mean_error,
                      'necessary_gate': recall >= .5 and false_rate <= .2 and mean_error < 2500})
summary = {'rows': len(rows), 'sections_per_checkpoint': len(retrieved_rows),
           'synthetic_subjects': len(subjects), 'checkpoints': summaries,
           'any_continuation_gate': any(row['necessary_gate'] for row in summaries[1:]),
           'scope': 'oracle tissue-query selection; frozen atlas bank and descriptor; synthetic pose stage only',
           'calibrated': False, 'public_benchmark_used': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'files_sha256': {
    name: hashlib.sha256((out / name).read_bytes()).hexdigest()
    for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'rows': len(rows), 'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', **summary, 'seconds': time.perf_counter() - started}), flush=True)
