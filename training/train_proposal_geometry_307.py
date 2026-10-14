"""Short matched TRAIN-only test of one versus two poses per normal anchor."""

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import CODE_SHA256 as V10_SHA, FOV_SCALE, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, BATCH, UPDATES = 20261014307, 192, 2, 800
STEPS = (0, 400, 800)
PARENT = ROOT / 'runs/joint_spatial_305_acquired_rehearsal_001/intervention/pose_local_00400.pt'
OLDER_PANEL = ROOT / 'runs/fov_balance_v10_short_pose_001/panel_draws.jsonl'
RUN = ROOT / 'runs/proposal_geometry_307_001'
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/PROPOSAL_GEOMETRY_307_PROTOCOL_20261014.md'
RESUME = RUN.exists()
assert SOURCE.drive.upper() == PARENT.drive.upper() == RUN.drive.upper() == 'I:'
assert FOV_SCALE == .72
with PARENT.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert parent_sha == '56d10febb1a6e6592adf18b9b4c7fe1fe0beae2a9179d9ddaffb78eefb82e692'

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
v9 = load_streaming_synthetic_v7_64('cuda')
v10 = {**v9, 'subjects': copy.deepcopy(v9['subjects']),
    'nuisance': v9['nuisance'].copy(), 'provenance': copy.deepcopy(v9['provenance'])}
v10['provenance']['virtual_subjects'] = v10['subjects']
for subject in v10['subjects']:
    bounds = np.asarray(subject['bounds_um'])
    middle = bounds.mean(0)
    subject['bounds_um'] = (middle + FOV_SCALE * (bounds - middle)).tolist()
v10['nuisance'][:, 2:4] *= FOV_SCALE
v10['provenance']['fov_balance_v10'] = {
    'code_sha256': V10_SHA, 'fov_scale': FOV_SCALE,
    'geometry': 'scale conservative subject box and TRAIN acquisition span lengths about their original centres',
    'unchanged': 'full-sphere normals, placement branches, PSF, local deformation, appearance, damage, optional brush'}

older_ids = {row['provenance']['physical_section_id'] for line in OLDER_PANEL.open(encoding='utf8')
    if (row := json.loads(line))['accepted']}
used_ids = set(older_ids)
if RESUME:
    for name in ('panel_draws.jsonl', 'control/draws.jsonl'):
        used_ids.update(json.loads(line)['provenance']['physical_section_id']
            for line in (RUN / name).open(encoding='utf8'))
panel_rng = np.random.default_rng(SEED + 1)
RUN.mkdir(parents=True, exist_ok=RESUME)
panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True) if RESUME else {}
with (RUN / 'panel_draws.jsonl').open('a' if RESUME else 'w', encoding='utf8') as panel_draws, torch.inference_mode():
    for version, context in (() if RESUME else (('v9', v9), ('v10', v10))):
        cases = []
        for partition in ('natural', 'sparse', 'full'):
            bases = np.repeat(np.arange(56, 64), 8) if partition == 'natural' else panel_rng.integers(56, 64, 16)
            for base in bases:
                accepted = False
                while not accepted:
                    variant = int(panel_rng.integers(VARIANTS))
                    seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
                    draw = (sample_postartifact_gauge_v1(context, [int(base) * VARIANTS + variant], seed, SIDE)
                            if version == 'v9' else
                            sample_fov_balance_v10(context, [int(base) * VARIANTS + variant], seed, SIDE))
                    record = draw['provenance'][0]
                    section_id = record['physical_section_id']
                    assert section_id not in used_ids and record['split'] == 'train'
                    used_ids.add(section_id)
                    fraction = float(draw['valid_mask'][0].float().mean())
                    accepted = bool(draw['eligible'][0]) and (
                        partition == 'natural' or
                        (partition == 'sparse' and fraction < .15) or
                        (partition == 'full' and fraction >= .35))
                    panel_draws.write(json.dumps({'version': version, 'partition': partition,
                        'base': int(base), 'variant': variant, 'seed': seed, 'accepted': accepted,
                        'visible_fraction': fraction, 'provenance': record}, allow_nan=False) + '\n')
                    if accepted:
                        pixels = torch.multinomial(draw['valid_mask'][0].flatten().float(), 128,
                            replacement=True, generator=torch.Generator(device='cuda').manual_seed(seed))
                        chart = torch.stack((pixels.remainder(SIDE),
                            pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                        cases.append({'inputs': draw['inputs'][0].cpu(), 'chart': chart.cpu(),
                            'truth': draw['centre'][0].reshape(-1, 3)[pixels].cpu(),
                            'state': draw['state'][0].cpu(), 'reflection': draw['reflection'][0].cpu(),
                            'meta': {'version': version, 'partition': partition, 'base': int(base),
                                'physical_section_id': section_id, 'visible_fraction': fraction,
                                'sampling_branch': record['sampling_branch'],
                                'framing': record['framing']['kind']}})
        assert len(cases) == 96
        panels[version] = cases
if not RESUME:
    torch.save(panels, RUN / 'panels.pt')

schedule_rng = np.random.default_rng(SEED + 2)
schedule = schedule_rng.integers(56, size=(UPDATES, BATCH))
with (RUN / 'schedule.jsonl').open('a' if RESUME else 'w', encoding='utf8') as schedule_file:
    for step, bases in (() if RESUME else enumerate(schedule, 1)):
        schedule_file.write(json.dumps({'step': step, 'synthetic_bases': bases.tolist(),
            'versions': ['v9', 'v10']}) + '\n')
config = {'candidate': 307, 'seed': SEED, 'batches_per_arm': UPDATES,
    'synthetic_sections_per_batch': BATCH, 'readouts': STEPS,
    'gradient_bases': list(range(56)), 'inner_panel_bases': list(range(56, 64)),
    'panel_cases_per_version': {'natural': 64, 'sparse': 16, 'full': 16},
    'parent_sha256': parent_sha,
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'v10_generator_sha256': V10_SHA,
    'panel_sha256': hashlib.sha256((RUN / 'panels.pt').read_bytes()).hexdigest(),
    'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.jsonl').read_bytes()).hexdigest(),
    'v9_input_sha256': v9['provenance']['input_sha256'],
    'no_acquired_gradient_or_development_data': True,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
if RESUME:
    frozen_config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
    for key in ('parent_sha256', 'protocol_sha256', 'panel_sha256',
                'panel_draws_sha256', 'schedule_sha256', 'v10_generator_sha256'):
        assert config[key] == frozen_config[key]
    config = frozen_config
else:
    (RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
all_readouts = {}
if RESUME:
    control_rows = [json.loads(line) for line in (RUN / 'control/readouts.jsonl').open(encoding='utf8')]
    assert len(control_rows) == len(STEPS) * 2 * 96
    assert (RUN / 'control/pose_anchor_00800.pt').exists()
    control_readouts = {}
    for step in STEPS:
        metrics = {}
        for version in panels:
            metrics[version] = {}
            for partition in ('natural', 'sparse', 'full'):
                group = [row for row in control_rows if row['step'] == step
                    and row['version'] == version and row['partition'] == partition]
                metrics[version][partition] = {
                    'cases': len(group),
                    'all_capture_le1p5mm': sum(row['all_best_site_mm'] <= 1.5 for row in group),
                    'top16_capture_le1p5mm': sum(row['top16_best_site_mm'] <= 1.5 for row in group),
                    'first_mean_site_mm': sum(row['first_site_mm'] for row in group) / len(group)}
        control_readouts[str(step)] = metrics
    all_readouts['control'] = control_readouts
    assert not any((RUN / 'intervention').iterdir())
for arm, slots in ((('intervention', 2),) if RESUME else (('control', 1), ('intervention', 2))):
    path = RUN / arm
    path.mkdir(exist_ok=RESUME)
    parent, parent_config = load_one_shot_checkpoint(PARENT, device='cuda')
    assert parent_config['candidate'] == 305
    if slots == 1:
        model = parent
    else:
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
            atlas_conditioning=True, vector_refinement=True,
            spatial_feedback=True, anchor_hypotheses=2).cuda()
        source_state = parent.state_dict()
        expanded = model.state_dict()
        for key in expanded:
            if expanded[key].shape == source_state[key].shape:
                expanded[key] = source_state[key].clone()
            else:
                assert key in ('anchor_pose.2.weight', 'anchor_pose.2.bias',
                               'anchor_global.2.weight', 'anchor_global.2.bias')
                expanded[key] = source_state[key].reshape(64, 18, *source_state[key].shape[1:]
                    ).repeat_interleave(2, 0).reshape_as(expanded[key]).clone()
        logits = expanded['anchor_pose.2.bias'].view(64, 2, 18)
        logits[:, :, 16] -= math.log(2)
        jitter = torch.Generator(device='cuda').manual_seed(SEED + 3)
        logits[:, 1, :3] += .08 * torch.randn((64, 3), device='cuda', generator=jitter)
        logits[:, 1, 5] += .05 * torch.randn((64,), device='cuda', generator=jitter)
        model.load_state_dict(expanded)
    del parent
    model.requires_grad_(False)
    model.anchor_pose.requires_grad_(True)
    model.anchor_global.requires_grad_(True)
    model.eval()
    parameters = list(model.anchor_pose.parameters()) + list(model.anchor_global.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + (10 if slots == 1 else 11))
    site_rng = torch.Generator(device='cuda').manual_seed(SEED + (20 if slots == 1 else 21))
    started = time.perf_counter()
    arm_readouts = {}
    with (path / 'draws.jsonl').open('w', encoding='utf8') as draws, \
            (path / 'training.jsonl').open('w', encoding='utf8') as training, \
            (path / 'readouts.jsonl').open('w', encoding='utf8') as readouts_file:
        for step in range(UPDATES + 1):
            if step:
                inputs, states, reflections, charts, truths = [], [], [], [], []
                for index, base in enumerate(schedule[step - 1]):
                    version = 'v9' if index == 0 else 'v10'
                    context = v9 if index == 0 else v10
                    accepted = False
                    while not accepted:
                        variant = int(rng.integers(VARIANTS))
                        seed = int(rng.integers(2**63 - 1, dtype=np.int64))
                        with torch.no_grad():
                            draw = (sample_postartifact_gauge_v1(context,
                                [int(base) * VARIANTS + variant], seed, SIDE) if index == 0 else
                                sample_fov_balance_v10(context,
                                [int(base) * VARIANTS + variant], seed, SIDE))
                        record = draw['provenance'][0]
                        section_id = record['physical_section_id']
                        assert section_id not in used_ids and record['split'] == 'train'
                        used_ids.add(section_id)
                        accepted = bool(draw['eligible'][0])
                        draws.write(json.dumps({'step': step, 'version': version, 'base': int(base),
                            'variant': variant, 'seed': seed, 'accepted': accepted,
                            'provenance': record}, allow_nan=False) + '\n')
                    valid = draw['valid_mask'][0]
                    pixels = torch.multinomial(valid.flatten().float(), 128,
                        replacement=True, generator=site_rng)
                    chart = torch.stack((pixels.remainder(SIDE),
                        pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                    inputs.append(draw['inputs'][0])
                    states.append(draw['state'][0])
                    reflections.append(draw['reflection'][0])
                    charts.append(chart)
                    truths.append(draw['centre'][0].reshape(-1, 3)[pixels])
                inputs, states, reflections, charts, truths = map(torch.stack,
                    (inputs, states, reflections, charts, truths))
                optimizer.param_groups[0]['lr'] = 3e-5 + 7e-5 * (1 + math.cos(math.pi * step / UPDATES)) / 2
                prediction = model.predict(inputs)
                proposed_state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflected = torch.arange(2, device='cuda')[None, None].expand(BATCH, model.modes, 2)
                proposed = rigid_points_090(proposed_state, reflected,
                    charts[:, None, None], (SIDE, SIDE))
                five = rigid_points_090(proposed_state, reflected, corners, (SIDE, SIDE))
                true_five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
                site_um = (proposed - truths[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
                normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                true_normal = full_frame_state_to_components(states)[1][..., :, 2]
                normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
                physical = (.75 * site_um + .25 * five_um + normal_um[..., None]) / 1000
                anchor_cost = physical[:, 16:].min(-1).values.reshape(BATCH, 64, slots)
                nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
                near = anchor_cost.min(-1).values.gather(1, nearest.indices)
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
                coverage_loss = (neighbourhood * near).sum(-1).mean()
                diversity_loss = coverage_loss.new_zeros(())
                if slots == 2:
                    pair = five[:, 16:, 0].reshape(BATCH, 64, 2, 5, 3)
                    separation = (pair[:, :, 0] - pair[:, :, 1]).norm(dim=-1).mean(-1) / 1000
                    selected_separation = separation.gather(1, nearest.indices)
                    diversity_loss = (.4 - selected_separation).clamp_min(0).square().mean()
                loss = coverage_loss + .2 * diversity_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                training.write(json.dumps({'step': step, 'presentations': BATCH * step,
                    'coverage_loss': float(coverage_loss.detach()),
                    'diversity_loss': float(diversity_loss.detach()),
                    'gradient_norm': float(gradient), 'lr': optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')

            if step in STEPS:
                rows = []
                with torch.inference_mode():
                    for version, panel in panels.items():
                        for case in panel:
                            inputs = case['inputs'][None].cuda()
                            chart = case['chart'].cuda()
                            truth = case['truth'].cuda()
                            prediction = model.predict(inputs)
                            states = prediction['state']
                            reflected = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
                            mapped = rigid_points_090(states[:, :, None].expand(-1, -1, 2, -1),
                                reflected, chart[None, None, None], (SIDE, SIDE))
                            errors = (mapped - truth[None, None, None]).norm(dim=-1).mean(-1).flatten() / 1000
                            mass = (prediction['log_mass'][..., None] + torch.stack((
                                F.logsigmoid(-prediction['reflection_logit']),
                                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                            first = int(mass.argmax())
                            top16 = mass.topk(16).indices
                            row = {**case['meta'], 'arm': arm, 'step': step,
                                'all_best_site_mm': float(errors.min()),
                                'top16_best_site_mm': float(errors[top16].min()),
                                'first_site_mm': float(errors[first]),
                                'first_mode': first // 2}
                            if slots == 2:
                                anchor_five = rigid_points_090(states[:, 16:, None].expand(-1, -1, 2, -1),
                                    reflected[:, 16:], corners, (SIDE, SIDE))[:, :, 0]
                                pairs = anchor_five.reshape(64, 2, 5, 3)
                                row['mean_slot_separation_mm'] = float((pairs[:, 0] - pairs[:, 1]
                                    ).norm(dim=-1).mean() / 1000)
                            rows.append(row)
                            readouts_file.write(json.dumps(row, allow_nan=False) + '\n')
                metrics = {}
                for version in panels:
                    metrics[version] = {}
                    for partition in ('natural', 'sparse', 'full'):
                        group = [row for row in rows if row['version'] == version
                            and row['partition'] == partition]
                        metrics[version][partition] = {
                            'cases': len(group),
                            'all_capture_le1p5mm': sum(row['all_best_site_mm'] <= 1.5 for row in group),
                            'top16_capture_le1p5mm': sum(row['top16_best_site_mm'] <= 1.5 for row in group),
                            'first_mean_site_mm': sum(row['first_site_mm'] for row in group) / len(group)}
                arm_readouts[str(step)] = metrics
                torch.save({'step': step, 'arm': arm, 'config': config, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, path / f'pose_anchor_{step:05d}.pt')
                readouts_file.flush()
                training.flush()
                draws.flush()
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                    'v10_natural_all_capture': metrics['v10']['natural']['all_capture_le1p5mm'],
                    'v9_natural_all_capture': metrics['v9']['natural']['all_capture_le1p5mm']}), flush=True)
    all_readouts[arm] = arm_readouts

control = all_readouts['control']['800']
intervention = all_readouts['intervention']['800']
initial = all_readouts['intervention']['0']
gate = {
    'v10_natural_gain_vs_control_ge6': intervention['v10']['natural']['all_capture_le1p5mm'] -
        control['v10']['natural']['all_capture_le1p5mm'] >= 6,
    'v10_natural_gain_vs_initial_ge4': intervention['v10']['natural']['all_capture_le1p5mm'] -
        initial['v10']['natural']['all_capture_le1p5mm'] >= 4,
    'v9_natural_loss_vs_control_le2': intervention['v9']['natural']['all_capture_le1p5mm'] >=
        control['v9']['natural']['all_capture_le1p5mm'] - 2,
    'v10_sparse_gain_ge1': intervention['v10']['sparse']['all_capture_le1p5mm'] >=
        control['v10']['sparse']['all_capture_le1p5mm'] + 1,
    'v10_full_gain_ge1': intervention['v10']['full']['all_capture_le1p5mm'] >=
        control['v10']['full']['all_capture_le1p5mm'] + 1,
    'v9_first_nonregression': intervention['v9']['natural']['first_mean_site_mm'] <=
        control['v9']['natural']['first_mean_site_mm'] + .2,
    'v10_first_nonregression': intervention['v10']['natural']['first_mean_site_mm'] <=
        control['v10']['natural']['first_mean_site_mm'] + .2}
summary = {'readouts': all_readouts, 'gate': gate, 'pass': all(gate.values())}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
output_hashes = {}
for file in RUN.rglob('*'):
    if file.is_file():
        with file.open('rb') as stream:
            output_hashes[file.relative_to(RUN).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
completion = {'schema': 'proposal-geometry-307-completion/v1', 'config_sha256': output_hashes['config.json'],
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'control_source_sha256': config['source_sha256'],
    'intervention_recovered_after_pretraining_device_error': RESUME,
    'protocol_sha256': config['protocol_sha256'],
    'output_sha256': output_hashes, 'summary': summary}
(RUN / 'completed.json').write_text(json.dumps(completion, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'gate': gate, 'pass': summary['pass']}), flush=True)
