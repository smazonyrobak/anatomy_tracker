"""Read-only rotation-group pooled 025 descriptors on fixed oracle-plane DEV."""
import hashlib
import json
import math
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
prior = root / 'runs/atlas_oriented_patch_025_development_eval/summary.json'
out = root / 'runs/atlas_group_pool_029'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
model = AtlasOrientedPatch025().cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pooled(patches, stem, step_degrees, offset_degrees=0):
    output = []
    for start in range(0, len(patches), 64):
        chunk = patches[start:start + 64]
        aggregate = 0
        for angle in range(0, 360, step_degrees):
            radians = math.radians(angle + offset_degrees)
            cosine, sine = math.cos(radians), math.sin(radians)
            theta = chunk.new_tensor([[cosine, -sine, 0], [sine, cosine, 0]])[None].expand(len(chunk), -1, -1)
            rotated = F.grid_sample(chunk, F.affine_grid(theta, chunk.shape, align_corners=True),
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
            aggregate = aggregate + F.normalize(model.shared(stem(rotated)), dim=-1)
        output.append(F.normalize(aggregate, dim=-1))
    return torch.cat(output)


baseline = next(row for row in json.loads(prior.read_text())['checkpoints'] if row['step'] == 3000)
out.mkdir(parents=True, exist_ok=False)
config = {'checkpoint_sha256': sha(checkpoint),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'baseline_summary_sha256': sha(prior),
          'baseline_recall1_500um': baseline['recall1_500um'],
          'baseline_recall16_500um': baseline['recall16_500um'],
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('diagnose_atlas_group_pool_029.py', 'atlas_oriented_patch_025.py')},
          'cases': len(records), 'query_points_per_case': 32, 'atlas_candidates_per_case': 288,
          'groups_degrees': [90, 30], 'atlas_half_shift_degrees': [45, 15],
          'truth_usage': 'oracle deformed CCF surface and GT-valid query selection; diagnostic only',
          'weights_updated': False, 'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = np.arange(8, 256, 16)
gy, gx = np.meshgrid(axis, axis, indexing='ij')
regular = np.stack((gy.flatten(), gx.flatten()), -1)
started = time.perf_counter()
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_xy = np.stack((chosen // 256, chosen % 256), -1)
            atlas_xy = np.concatenate((regular, query_xy))
            truth = torch.from_numpy(arrays['target_centre_um'][query_xy[:, 0],
                                                              query_xy[:, 1]].copy()).cuda()
            atlas_ccf = torch.from_numpy(arrays['target_centre_um'][atlas_xy[:, 0],
                                                                  atlas_xy[:, 1]].copy()).cuda()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        rendered = atlas_surface_image(atlas, centre, state, offsets, weights)
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                torch.from_numpy(query_xy.copy()).cuda().float())
        atlas_xy_gpu = torch.from_numpy(atlas_xy.copy()).cuda().float()
        key = torch.cat([extract_patches(rendered,
            torch.zeros(min(64, len(atlas_xy) - start), device='cuda', dtype=torch.long),
            atlas_xy_gpu[start:start + 64]) for start in range(0, len(atlas_xy), 64)])
        outcomes = {}
        for step, half in ((90, 45), (30, 15)):
            query_feature = pooled(query, model.image_stem, step)
            atlas_feature = pooled(key, model.atlas_stem, step)
            shifted_feature = pooled(key, model.atlas_stem, step, half)
            for name, feature in ((f'group{360 // step}', atlas_feature),
                                  (f'group{360 // step}_halfshift', shifted_feature)):
                top = (query_feature @ feature.T).topk(16, -1).indices
                distance = (atlas_ccf[top] - truth[:, None]).norm(dim=-1)
                outcomes[name] = {'recall1_500um': float((distance[:, 0] < 500).float().mean()),
                                  'recall16_500um': float((distance.min(-1).values < 500).float().mean())}
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row['conditions'] = outcomes
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {}
for condition in ('group4', 'group4_halfshift', 'group12', 'group12_halfshift'):
    result = {}
    for field in ('recall1_500um', 'recall16_500um'):
        result[field] = float(np.mean([np.mean([row['conditions'][condition][field] for row in rows
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    result['mode_recall1_500um'] = {mode: float(np.mean([
        np.mean([row['conditions'][condition]['recall1_500um'] for row in rows
                 if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in rows)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summary[condition] = result
gate = (summary['group12']['recall1_500um'] >= .8 and
        summary['group12']['recall1_500um'] - summary['group12_halfshift']['recall1_500um'] <= .1 and
        all(min(summary[name]['mode_recall1_500um'].values()) >= .5 * summary[name]['recall1_500um']
            for name in ('group12', 'group12_halfshift')))
(out / 'summary.json').write_text(json.dumps({'cases': len(rows), 'synthetic_subjects': len(subjects),
    'conditions': summary, 'pooled_bank_gate': bool(gate),
    'baseline_025': {'recall1_500um': baseline['recall1_500um'],
                     'recall16_500um': baseline['recall16_500um']},
    'scope': 'oracle deformed atlas plane; no unknown-plane localization',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'pooled_bank_gate': bool(gate),
                  'summary': summary, 'seconds': time.perf_counter() - started}), flush=True)
