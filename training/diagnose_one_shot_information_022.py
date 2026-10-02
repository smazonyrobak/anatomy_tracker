"""Frozen 019 fitted-plane local image-information comparison at two resolutions."""
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
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
control = root / 'runs/one_shot_exposure_019_candidate_rank_diagnostic/rows.jsonl'
out = root / 'runs/one_shot_information_022'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
controls = {row['sha256']: row for row in map(json.loads, control.open())}
assert len(records) == len(controls) == 185
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 18000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved


def disagreement(image, valid, mapped, weights, size):
    count = mapped['coordinates'].shape[1]
    masses = weights.expand(count, -1)
    rendered = render_finite_thickness_coordinate_grid(
        atlas, mapped['coordinates'][0], (0., 0., 0.), (25., 25., 25.), masses)
    support = rendered[:, 1:2].clamp(0, 1)
    target = rendered[:, :1] / support.clamp_min(1e-4)
    source = F.interpolate(image[:, :1], (size, size), mode='area').expand(count, -1, -1, -1)
    truth = F.interpolate(valid[:, None], (size, size), mode='area').expand(count, -1, -1, -1)
    predicted = mapped['correspondence_logit'][0].sigmoid()
    a = source - F.avg_pool2d(source, 9, 1, 4)
    b = target - F.avg_pool2d(target, 9, 1, 4)
    cross = F.avg_pool2d(a * b, 9, 1, 4)
    variance = F.avg_pool2d(a.square(), 9, 1, 4) * F.avg_pool2d(b.square(), 9, 1, 4)
    agreement = (cross / (variance + 1e-5).sqrt()).abs().clamp(0, 1)
    return [(((1 - agreement) * tissue * support + tissue * (1 - support)).sum((1, 2, 3))
             / tissue.sum((1, 2, 3)).clamp_min(1)).tolist()
            for tissue in (truth, predicted)]


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().float()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        choice = prior.topk(8).indices
        control_row = controls[record['sha256']]
        assert choice.tolist() == control_row['choice']
        state = prediction['state'][:, choice // 2]
        reflected = (choice % 2)[None]
        selected = {**prediction, 'state': state,
                    'log_mass': prediction['log_mass'][:, choice // 2],
                    'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
        index = torch.arange(8, device='cuda')[None]
        first = model.map(selected, offsets, index, reflected, (64, 64), atlas, weights,
                          return_refinement_feature=True, feature_side=64,
                          source_shape=(256, 256))
        refined_state, _, _ = model.refine(first['refinement_feature'], state)
        refined = {**selected, 'state': refined_state}
        scores = {'96_oracle': [], '96_predicted': [],
                  '256_oracle': [], '256_predicted': []}
        for size in (96, 256):
            chunk = 8 if size == 96 else 2
            for start in range(0, 8, chunk):
                mapped = model.map(refined, offsets, index[:, start:start + chunk],
                                   reflected[:, start:start + chunk], (size, size), atlas, weights,
                                   feature_side=size if size == 96 else None,
                                   source_shape=(256, 256) if size == 96 else None)
                oracle, predicted = disagreement(image, valid, mapped, weights, size)
                scores[f'{size}_oracle'].extend(oracle)
                scores[f'{size}_predicted'].extend(predicted)
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                           'section_id', 'sha256')}
        row.update(valid_fraction=float(valid.mean()), choice=choice.tolist(),
                   physical_mapped_um=control_row['physical_mapped_um'],
                   fitted_score=control_row['fitted_score'], disagreement=scores)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'cases': number}), flush=True)

summary = []
animals = sorted({row['animal_id'] for row in rows})
for name in ('fitted', '96_oracle', '96_predicted', '256_oracle', '256_predicted', 'best8'):
    if name == 'fitted':
        chosen = [int(np.argmax(row['fitted_score'])) for row in rows]
    elif name == 'best8':
        chosen = [int(np.argmin(row['physical_mapped_um'])) for row in rows]
    else:
        chosen = [int(np.argmin(row['disagreement'][name])) for row in rows]
    errors = [row['physical_mapped_um'][index] for row, index in zip(rows, chosen)]
    best = [int(np.argmin(row['physical_mapped_um'])) for row in rows]
    summary.append({'method': name,
                    'identity_equal_mapped_um': float(np.mean([
                        np.mean([error for row, error in zip(rows, errors)
                                 if row['animal_id'] == animal]) for animal in animals])),
                    'case_mean_mapped_um': float(np.mean(errors)),
                    'top1_oracle_fraction': float(np.mean(np.asarray(chosen) == best))})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'control_rows_sha256': hashlib.sha256(control.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'cases': len(rows)}), flush=True)
