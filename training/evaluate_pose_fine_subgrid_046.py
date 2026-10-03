"""Matched synthetic DEV key and continuous-point comparison across 046 checkpoints."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_oriented_patch_025 import extract_patches
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046
from training.pose_fine_patch_045 import atlas_key_patches

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
run = root / 'runs/pose_fine_subgrid_046_pilot'
out = root / 'runs/pose_fine_subgrid_046_development_eval'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
steps = (0, 500, 1000, 1500, 2000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [record for record in records if record['eligible']]
assert len(eligible) == 177 and len({row['synthetic_subject_plan_id'] for row in eligible}) == 8
assert json.loads((run / 'completed.json').read_text())['updates'] == 2000
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                             vector_refinement=True, candidate_ranking=True,
                             fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(coarse_path, map_location='cpu', weights_only=True)['head'])
models = {}
for step in steps:
    checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
                            map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = PoseFineMatch046().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    models[step] = model
del checkpoint

out.mkdir(parents=True, exist_ok=False)
config = {'steps': list(steps), 'beam': 8, 'shortlist': 32,
          'unsupported_error_um': 20000,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'model_source_sha256': hashlib.sha256(Path(__file__).with_name(
              'pose_fine_match_046.py').read_bytes()).hexdigest(),
          'patch_source_sha256': hashlib.sha256(Path(__file__).with_name(
              'pose_fine_patch_045.py').read_bytes()).hexdigest(),
          'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
          'train_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'coarse_sha256': hashlib.sha256(coarse_path.read_bytes()).hexdigest(),
          'checkpoints_sha256': {str(step): hashlib.sha256(
              (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
          'scope': 'synthetic DEV only; physically best-eight is oracle diagnostic',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
started = time.perf_counter()
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(eligible, 1):
        path = panel / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = pose.predict(image)
        prior_score = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = prior_score.topk(8, -1).indices
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(state)
        ids = valid.flatten().nonzero().flatten()
        chart = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')),
                             -1).float()[None, None] / 256
        chart = chart.expand(1, 8, -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / 256 - chart[..., 0], chart[..., 0])
        plane = centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
            frame[..., :, :2] @ basis, chart - .5)
        physical = (plane - target.reshape(-1, 3)[ids][None, None]).norm(dim=-1).mean(-1)[0]
        best = int(physical.argmin())
        coarse_result = coarse(prediction, image, state, reflection, atlas, offsets, weights)
        valid_query = valid[8::16, 8::16].reshape(256).nonzero().flatten()
        truth = target[8::16, 8::16].reshape(256, 3)[valid_query]
        qy, qx = torch.meshgrid(torch.arange(8, 256, 16, device='cuda'),
                                torch.arange(8, 256, 16, device='cuda'), indexing='ij')
        query_xy = torch.stack((qy, qx), -1).reshape(256, 2)[valid_query]
        query_patch = extract_patches(image,
            torch.zeros(len(valid_query), device='cuda', dtype=torch.long), query_xy.float())
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
               'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(valid_query_cells=len(valid_query), prior_top1_um=float(physical[0]),
                   prior_best8_um=float(physical[best]), prior_best8_index=best)
        for label, candidate in (('top1', 0), ('best8', best)):
            top = coarse_result['logits'][0, candidate, valid_query].topk(32)
            world = coarse_result['key_world'][0, candidate, top.indices]
            support = coarse_result['key_support'][0, candidate, top.indices] > .1
            distance = (world - truth[:, None]).norm(dim=-1)
            rotation = frame[0, candidate]
            relative = (world - coarse_result['query_base'][0, candidate, valid_query, None]) @ rotation
            for threshold in (500, 1500):
                row[f'{label}_available_{threshold}um'] = int(
                    ((distance <= threshold) & support).any(-1).sum())
            raw = {step: [] for step in steps}
            corrected = {step: [] for step in steps}
            for first in range(0, len(valid_query), 4):
                last = min(first + 4, len(valid_query))
                keys = world[first:last].reshape(-1, 3)
                patch = atlas_key_patches(atlas, keys,
                    state[0, candidate][None].expand(len(keys), -1),
                    reflection[0, candidate][None].expand(len(keys)),
                    offsets[0][None].expand(len(keys), -1),
                    weights[0][None].expand(len(keys), -1))
                for step, model in models.items():
                    scores, local_offset = model(query_patch[first:last], patch,
                        relative[first:last], top.values[first:last])
                    scores = scores.masked_fill(~support[first:last], -1e4)
                    index = scores.argmax(-1)
                    key = world[first:last].gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
                    shift = local_offset.gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
                    selected_support = support[first:last].gather(1, index[:, None])[:, 0]
                    raw_error = (key - truth[first:last]).norm(dim=-1)
                    corrected_error = (key + shift @ rotation.T - truth[first:last]).norm(dim=-1)
                    raw[step].extend(torch.where(selected_support, raw_error,
                        raw_error.new_full(raw_error.shape, 20000)).cpu().tolist())
                    corrected[step].extend(torch.where(selected_support, corrected_error,
                        corrected_error.new_full(corrected_error.shape, 20000)).cpu().tolist())
            for step in steps:
                row[f'{label}_step{step}_raw_um'] = raw[step]
                row[f'{label}_step{step}_corrected_um'] = corrected[step]
                for name, values in (('raw', raw[step]), ('corrected', corrected[step])):
                    for threshold in (500, 1500):
                        row[f'{label}_step{step}_{name}_recall_{threshold}um'] = float(
                            np.mean(np.asarray(values) <= threshold))
                    row[f'{label}_step{step}_{name}_mean_um'] = float(np.mean(values))
                row[f'{label}_step{step}_harmful_correction_fraction'] = float(np.mean(
                    np.asarray(corrected[step]) > np.asarray(raw[step]) + 250))
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number in (48, 96, 144, 177):
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}),
                  flush=True)

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'synthetic_eligible_sections': len(rows), 'synthetic_subjects': len(identities),
           'seconds': time.perf_counter() - started,
           'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
           'mean_valid_query_cells': float(np.mean([row['valid_query_cells'] for row in rows]))}
for label in ('top1', 'best8'):
    summary[label] = {}
    for step in steps:
        for name in ('raw', 'corrected'):
            for metric in ('recall_500um', 'recall_1500um', 'mean_um'):
                field = f'{label}_step{step}_{name}_{metric}'
                summary[label][f'step{step}_{name}_{metric}'] = float(np.mean([
                    np.mean([row[field] for row in rows if row['synthetic_subject_plan_id'] == identity])
                    for identity in identities]))
        field = f'{label}_step{step}_harmful_correction_fraction'
        summary[label][f'step{step}_harmful_correction_fraction'] = float(np.mean([
            np.mean([row[field] for row in rows if row['synthetic_subject_plan_id'] == identity])
            for identity in identities]))
    summary[label]['by_appearance_corrected_500um'] = {appearance: {str(step): float(np.mean([
        np.mean([row[f'{label}_step{step}_corrected_recall_500um'] for row in rows
                 if row['synthetic_subject_plan_id'] == identity and
                    row['appearance_mode'] == appearance])
        for identity in identities if any(row['synthetic_subject_plan_id'] == identity and
            row['appearance_mode'] == appearance for row in rows)])) for step in steps}
        for appearance in ('raw', 'exact_black', 'imperfect_brush')}
summary['stage_promising'] = {str(step): (
    summary['best8'][f'step{step}_corrected_recall_500um'] >= .25 and
    summary['best8'][f'step{step}_corrected_recall_1500um'] >= .60 and
    all(summary['best8']['by_appearance_corrected_500um'][appearance][str(step)] >=
        summary['best8']['by_appearance_corrected_500um'][appearance]['0'] - .05
        for appearance in ('raw', 'exact_black', 'imperfect_brush')))
    for step in steps[1:]}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    name + '_sha256': hashlib.sha256((out / (name + ('.jsonl' if name == 'rows'
        else '.json'))).read_bytes()).hexdigest() for name in ('config', 'rows', 'summary')
} | {'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'stage_promising': summary['stage_promising'],
                  'seconds': summary['seconds']}), flush=True)
