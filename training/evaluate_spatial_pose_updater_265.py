"""Blind frozen-DEV readout of spatial pose capture and matched support control."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.spatial_pose_candidate_265 import (
    BANK, SpatialPoseUpdater, candidate_bank, five_point_error,
    render_coarse, render_exact,
)

PARENT = ROOT / 'runs/dense_physical_pose_260_train_001'
TRAIN = ROOT / 'runs/spatial_pose_updater_265_train_001'
FRESH = ROOT / 'data/spatial_pose_fresh_dev_265_001'
FROZEN = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/spatial_pose_updater_265_readout_001'
DOC = Path('docs/publication/SPATIAL_POSE_UPDATER_265_PROTOCOL_20261012.md')
STEPS, SIDE, TOP_K, CHUNK = (1000, 2000, 3000), 192, 32, 128
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

train_done = json.loads((TRAIN / 'completed.json').read_text())
train_config = json.loads((TRAIN / 'config.json').read_text())
parent_done = json.loads((PARENT / 'completed.json').read_text())
fresh_done = json.loads((FRESH / 'completed.json').read_text())
frozen_done = json.loads((FROZEN / 'completed.json').read_text())
assert hashlib.sha256((TRAIN / 'config.json').read_bytes()).hexdigest() == train_done['config_sha256']
assert hashlib.sha256((TRAIN / 'training.jsonl').read_bytes()).hexdigest() == train_done['training_sha256']
assert hashlib.sha256((TRAIN / 'draws.jsonl').read_bytes()).hexdigest() == train_done['draws_sha256']
assert hashlib.sha256(DOC.read_bytes()).hexdigest() == train_config['protocol_sha256']
assert hashlib.sha256((FRESH / 'records.jsonl').read_bytes()).hexdigest() == fresh_done['records_sha256']
assert hashlib.sha256((FROZEN / 'records.jsonl').read_bytes()).hexdigest() == frozen_done['records_sha256']
assert hashlib.sha256((PARENT / 'joint_step_03000.pt').read_bytes()).hexdigest() == parent_done['checkpoint_sha256']['3000']
assert all(hashlib.sha256((TRAIN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
           == train_done['checkpoint_sha256'][str(step)] for step in STEPS)
assert all(hashlib.sha256((Path('training') / name).read_bytes()).hexdigest() == digest
           for name, digest in train_config['source_sha256'].items())
fresh_records = [row for line in (FRESH / 'records.jsonl').open(encoding='utf8')
                 if (row := json.loads(line))['eligible'] and row['valid_pixels'] > 0]
frozen_records = [row for line in (FROZEN / 'records.jsonl').open(encoding='utf8')
                  if (row := json.loads(line))['eligible'] and row['valid_pixels'] > 0]
assert len(frozen_records) == 227 and len({row['animal_id'] for row in frozen_records}) == 8
assert {row['provenance']['physical_section_id'] for row in fresh_records}.isdisjoint(
       {row['provenance']['physical_section_id'] for row in frozen_records})
assert {row['animal_id'] for row in fresh_records} == {row['animal_id'] for row in frozen_records}

context = load_streaming_synthetic_v7_64(device='cuda')
parent = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
    dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
parent.load_state_dict(torch.load(PARENT / 'joint_step_03000.pt',
                                   map_location='cpu', weights_only=True)['model'], strict=True)
anchor_model = OneShotJointSliceModel(modes=0, normal_anchor_count=64)
anchors = anchor_model.normal_anchor_frames.cuda()
del anchor_model
models = {'full': SpatialPoseUpdater().cuda().eval(),
          'support': SpatialPoseUpdater().cuda().eval()}
all_ids = torch.arange(BANK, device='cuda')
axis = torch.arange(SIDE, device='cuda') / SIDE
yy, xx = torch.meshgrid(axis, axis, indexing='ij')


def rigid_error(states, flags, target, valid):
    centre, frame, basis = full_frame_state_to_components(states)
    edges = frame[:, :, :2] @ basis
    x = torch.where(flags[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((x.expand(-1, SIDE, SIDE), yy.expand(len(states), -1, -1)), -1) - .5
    points = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi', edges, chart)
    return ((points - target[None]).norm(dim=-1) * valid).sum((1, 2)) / valid.sum().clamp_min(1) / 1000


def run_case(directory, record, step, arms):
    assert hashlib.sha256((directory / record['file']).read_bytes()).hexdigest() == record['sha256']
    with np.load(directory / record['file'], allow_pickle=False) as arrays:
        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().float()
        true_flag = torch.from_numpy(np.asarray(arrays['reflection']).reshape(1).copy()).cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    predicted = parent.dense_coordinate_plane(parent.predict(image), source_side=SIDE)
    centre = full_frame_state_to_components(predicted)[0][0]
    bank, flags, _ = candidate_bank(centre, anchors)
    full_scores = {arm: torch.empty(BANK, device='cuda') for arm in arms}
    supports = torch.empty(BANK, device='cuda')
    queries = {arm: models[arm].encode_image(image) for arm in arms}
    for start in range(0, BANK, CHUNK):
        ids = all_ids[start:start + CHUNK]
        rendered = render_coarse(context['atlas'], centre, anchors, ids)
        supports[ids] = (rendered[:, 1] > .1).float().mean((1, 2))
        for arm in arms:
            pair = rendered if arm == 'full' else torch.cat((
                torch.zeros_like(rendered[:, :1]), rendered[:, 1:2]), 1)
            full_scores[arm][ids] = models[arm](queries[arm], pair)[0]
    true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
    true_normal = true_frame[0, :, 2]
    true_spans = torch.linalg.svdvals(true_basis[0]).sort().values
    parent_axis = float(torch.rad2deg(torch.acos(
        (full_frame_state_to_components(predicted)[1][0, :, 2] @ true_normal).abs().clamp_max(1))))
    parent_rigid = float(rigid_error(predicted, torch.zeros_like(true_flag), target, valid)[0])
    bank_error = five_point_error(bank, flags, truth, true_flag)
    cut = float(torch.rad2deg(torch.acos(true_normal.abs().max().clamp_max(1))))
    common = {'panel': 'fresh' if directory == FRESH else 'frozen_248', 'batch': step,
        'physical_section_id': record['provenance']['physical_section_id'],
        'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
        'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
        'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
        'valid_fraction': record['valid_pixels'] / SIDE**2,
        'cut_nearest_principal_deg': cut, 'parent260_rigid_mm': parent_rigid,
        'parent260_axis_deg': parent_axis, 'bank_nearest_five_mm': float(bank_error.min())}
    rows = []
    for arm in arms:
        ids = full_scores[arm].topk(TOP_K).indices
        states, selected_flags = bank[ids], flags[ids]
        initial_five = bank_error[ids]
        start_centre, start_frame, start_basis = full_frame_state_to_components(states)
        start_normal = start_frame[:, :, 2]
        normal_deg = torch.rad2deg(torch.acos((start_normal @ true_normal).abs().clamp_max(1)))
        centre_mm = (start_centre - true_centre).norm(dim=-1) / 1000
        spans = torch.linalg.svdvals(start_basis).sort().values
        span_log = (spans.log() - true_spans.log()).abs().amax(-1)
        reachable = (normal_deg <= 15) & (centre_mm <= 3) & (span_log <= .7)
        for _ in range(2):
            rendered = render_exact(context['atlas'], states, selected_flags,
                                    support_only=arm == 'support')
            _, update, _ = models[arm](queries[arm], rendered)
            states = compose_full_frame_state(states, update)
        final_render = render_exact(context['atlas'], states, selected_flags,
                                    support_only=arm == 'support')
        final_score = models[arm](queries[arm], final_render)[0]
        chosen = final_score.argmax()
        corrected_rigid = rigid_error(states, selected_flags, target, valid)
        corrected_five = five_point_error(states, selected_flags, truth, true_flag)
        corrected_frame = full_frame_state_to_components(states)[1]
        corrected_axis = torch.rad2deg(torch.acos((corrected_frame[:, :, 2] @ true_normal).abs().clamp_max(1)))
        rows.append({**common, 'arm': arm,
            'top_k': TOP_K, 'topk_pre_residual_reach': bool(reachable.any()),
            'topk_reachable_count': int(reachable.sum()),
            'topk_best_initial_five_mm': float(initial_five.min()),
            'topk_mean_atlas_support_fraction': float(supports[ids].mean()),
            'topk_zero_atlas_support_count': int((supports[ids] < .01).sum()),
            'topk_best_corrected_five_mm': float(corrected_five.min()),
            'topk_best_corrected_rigid_mm': float(corrected_rigid.min()),
            'selected_five_mm': float(corrected_five[chosen]),
            'selected_rigid_mm': float(corrected_rigid[chosen]),
            'selected_axis_deg': float(corrected_axis[chosen]),
            'selected_atlas_support_fraction': float(supports[ids[chosen]]),
            'selected_candidate_index': int(ids[chosen])})
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return [{**row, 'inference_seconds_both_arms': seconds,
             'peak_gpu_mib_both_arms': peak} for row in rows]


OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for step in STEPS:
        saved = torch.load(TRAIN / f'joint_step_{step:05d}.pt',
                           map_location='cpu', weights_only=True)
        assert saved['batch'] == step and not saved['calibrated']
        for arm in models:
            models[arm].load_state_dict(saved[arm], strict=True)
        for record in fresh_records:
            case_rows = run_case(FRESH, record, step, tuple(models))
            rows.extend(case_rows)
            for row in case_rows:
                stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'event': 'fresh_checkpoint', 'batch': step,
                          'cases': len(fresh_records)}), flush=True)

    plans = sorted({row['animal_id'] for row in fresh_records})
    fresh_means = {(step, arm, key): float(np.mean([
        np.mean([row[key] for row in rows if row['panel'] == 'fresh'
                 and row['batch'] == step and row['arm'] == arm and row['animal_id'] == plan])
        for plan in plans])) for step in STEPS for arm in models
        for key in ('selected_rigid_mm', 'selected_axis_deg',
                    'parent260_rigid_mm', 'parent260_axis_deg')}
    chosen = {arm: min(STEPS, key=lambda step: fresh_means[(step, arm, 'selected_rigid_mm')])
              for arm in models}
    frozen_steps = sorted(set(chosen.values()))
    for step in frozen_steps:
        saved = torch.load(TRAIN / f'joint_step_{step:05d}.pt',
                           map_location='cpu', weights_only=True)
        for arm in models:
            models[arm].load_state_dict(saved[arm], strict=True)
        for record in frozen_records:
            case_rows = run_case(FROZEN, record, step, tuple(models))
            rows.extend(case_rows)
            for row in case_rows:
                stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'event': 'frozen_checkpoint', 'batch': step,
                          'cases': len(frozen_records)}), flush=True)

keys = ('selected_rigid_mm', 'selected_axis_deg', 'topk_best_corrected_rigid_mm',
        'topk_best_corrected_five_mm', 'topk_pre_residual_reach', 'parent260_rigid_mm',
        'parent260_axis_deg', 'bank_nearest_five_mm')
summary = {'fresh_cases': len(fresh_records), 'frozen_cases': len(frozen_records),
    'fresh_equal_plan': {f'{step}_{arm}': {key: fresh_means[(step, arm, key)] for key in
        ('selected_rigid_mm', 'selected_axis_deg', 'parent260_rigid_mm', 'parent260_axis_deg')}
        for step in STEPS for arm in models},
    'chosen_batch': chosen, 'frozen': {}}
for step in frozen_steps:
    for arm in models:
        selected = [row for row in rows if row['panel'] == 'frozen_248'
                    and row['batch'] == step and row['arm'] == arm]
        equal = {key: float(np.mean([np.mean([row[key] for row in selected
            if row['animal_id'] == plan]) for plan in plans])) for key in keys}
        per_plan = {plan: {'selected_rigid_mm': float(np.mean([
            row['selected_rigid_mm'] for row in selected if row['animal_id'] == plan])),
            'parent260_rigid_mm': float(np.mean([row['parent260_rigid_mm']
                                                for row in selected if row['animal_id'] == plan]))}
            for plan in plans}
        appearance = {mode: {'cases': len(group := [row for row in selected
            if row['appearance_mode'] == mode]),
            'selected_rigid_mm': float(np.mean([row['selected_rigid_mm'] for row in group]))}
            for mode in ('raw', 'exact_black', 'imperfect_brush')}
        visibility = {f'{low}_{high}': {'cases': len(group := [row for row in selected
            if low <= row['valid_fraction'] < high]),
            'selected_rigid_mm': float(np.mean([row['selected_rigid_mm'] for row in group])) if group else None}
            for low, high in ((0., .1), (.1, .3), (.3, 1.01))}
        cut = {f'{low}_{high}': {'cases': len(group := [row for row in selected
            if low <= row['cut_nearest_principal_deg'] < high]),
            'selected_rigid_mm': float(np.mean([row['selected_rigid_mm'] for row in group])) if group else None}
            for low, high in ((0, 15), (15, 30), (30, 45), (45, 60))}
        summary['frozen'][f'{step}_{arm}'] = {'equal_plan': equal, 'per_plan': per_plan,
            'appearance': appearance, 'visibility': visibility, 'cut': cut,
            'case_mean_inference_seconds_both_arms': float(np.mean([
                row['inference_seconds_both_arms'] for row in selected])),
            'peak_gpu_mib_max': float(max(row['peak_gpu_mib_both_arms'] for row in selected))}

batch = chosen['full']
full = summary['frozen'][f'{batch}_full']
paired_control = summary['frozen'][f'{batch}_support']
control = summary['frozen'][f'{chosen["support"]}_support']
summary['paired_same_batch'] = {
    'batch': batch,
    'full_equal_plan': full['equal_plan'],
    'support_equal_plan': paired_control['equal_plan'],
    'support_minus_full_selected_rigid_mm':
        paired_control['equal_plan']['selected_rigid_mm'] - full['equal_plan']['selected_rigid_mm'],
}
summary['independently_chosen_support'] = {
    'batch': chosen['support'], 'equal_plan': control['equal_plan']}
f = full['equal_plan']
c = control['equal_plan']
summary['gate_components'] = {
    'selected_rigid_improvement': f['parent260_rigid_mm'] - f['selected_rigid_mm'] >= .5,
    'axis_improvement': f['parent260_axis_deg'] - f['selected_axis_deg'] >= 15,
    'plans_improved': sum(full['per_plan'][plan]['selected_rigid_mm'] <
                          full['per_plan'][plan]['parent260_rigid_mm'] for plan in plans) >= 7,
    'topk_residual_reach': f['topk_pre_residual_reach'] >= .75,
    'best_of_k_rigid': f['topk_best_corrected_rigid_mm'] <= 1.25,
    'atlas_intensity_gain': c['selected_rigid_mm'] - f['selected_rigid_mm'] >= .25,
    'fresh_direction': fresh_means[(batch, 'full', 'selected_rigid_mm')] <
        fresh_means[(batch, 'full', 'parent260_rigid_mm')] and
        fresh_means[(batch, 'full', 'selected_axis_deg')] <
        fresh_means[(batch, 'full', 'parent260_axis_deg')],
}
summary['gate'] = all(summary['gate_components'].values())
summary['scope'] = 'synthetic development only; reused 248 plans; no biological calibration or GUI readiness'
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'fresh_cases': len(fresh_records), 'frozen_cases': len(frozen_records),
    'train_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'fresh_completed_sha256': hashlib.sha256((FRESH / 'completed.json').read_bytes()).hexdigest(),
    'frozen_completed_sha256': hashlib.sha256((FROZEN / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'gate': summary['gate'], 'calibrated': False,
    'biological_validation': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'chosen_batch': chosen,
                  'gate': summary['gate']}), flush=True)
