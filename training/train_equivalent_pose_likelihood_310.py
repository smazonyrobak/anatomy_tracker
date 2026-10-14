"""Matched TRAIN-only test of a reflection-equivalent pose likelihood; no acquired gradients."""
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

from training.arbitrary_plane_fov_balance_v10 import FOV_SCALE, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import frame_to_physical_ouv, physical_ouv_to_frame
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261015310, 192, 1500
PARENT = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
RUN = ROOT / 'runs/equivalent_pose_likelihood_310_001'
assert Path(__file__).resolve().drive.upper() == PARENT.drive.upper() == RUN.drive.upper() == 'I:'
assert not RUN.exists() and FOV_SCALE == .72
parent_sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
assert parent_sha == '7e4be46fe0c8e6637e183865adce9b2ab554058b81b341bdf98bf0fd2ad0d797'
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
v10['provenance']['fov_balance_v10'] = {'code_sha256': hashlib.sha256(
    Path(sample_fov_balance_v10.__code__.co_filename).read_bytes()).hexdigest(),
    'fov_scale': FOV_SCALE}

RUN.mkdir(parents=True)
flags = torch.arange(2, device='cuda')[None, None]
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
panel_rng = np.random.default_rng(SEED + 1)
used_seeds, used_ids = set(), set()
panels = {}
with (RUN / 'panel_draws.jsonl').open('w', encoding='utf8') as draws, torch.inference_mode():
    for version, count in (('v10', 128), ('v9', 64), ('v10_sparse', 32), ('v10_steep', 32)):
        panels[version] = []
        for slot in range(count):
            base = 56 + slot % 8
            while True:
                variant = int(panel_rng.integers(VARIANTS))
                seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
                assert seed not in used_seeds
                used_seeds.add(seed)
                sample = (sample_postartifact_gauge_v1(v9, [base * VARIANTS + variant], seed, SIDE)
                          if version == 'v9' else
                          sample_fov_balance_v10(v10, [base * VARIANTS + variant], seed, SIDE))
                visible = float(((sample['visible'][0] > .05) &
                    ((sample['inputs'][0, 2] < .5) | sample['brush_mask'][0])).float().mean())
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                obliquity = math.degrees(math.acos(float(normal.abs().max().clamp(0, 1))))
                accepted = bool(sample['eligible'][0]) and (
                    version != 'v10_sparse' or visible < .15) and (
                    version != 'v10_steep' or obliquity >= 45)
                provenance = sample['provenance'][0]
                draws.write(json.dumps({'cohort': version, 'slot': slot, 'base': base,
                    'variant': variant, 'seed': seed, 'accepted': accepted,
                    'visible_fraction': visible, 'obliquity_deg': obliquity,
                    'provenance': provenance}, allow_nan=False) + '\n')
                if accepted:
                    break
            physical_id = provenance['physical_section_id']
            assert provenance['split'] == 'train' and physical_id not in used_ids
            used_ids.add(physical_id)
            sites = torch.multinomial(sample['valid_mask'][0].flatten().float(), 128,
                replacement=True, generator=torch.Generator(device='cuda').manual_seed(seed))
            chart = torch.stack((sites.remainder(SIDE),
                sites.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            panels[version].append({'inputs': sample['inputs'][0].cpu(), 'chart': chart.cpu(),
                'truth': sample['centre'][0].reshape(-1, 3)[sites].cpu(),
                'meta': {'physical_section_id': physical_id, 'base': base,
                         'variant': variant, 'seed': seed, 'visible_fraction': visible,
                         'obliquity_deg': obliquity}})
torch.save(panels, RUN / 'panels.pt')

plan_rng = np.random.default_rng(SEED + 2)
plan = []
with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, torch.inference_mode():
    for step in range(1, UPDATES + 1):
        batch_plan = []
        for slot, version in enumerate(('v9', 'v10', 'v10')):
            base = int(plan_rng.integers(56))
            while True:
                variant = int(plan_rng.integers(VARIANTS))
                seed = int(plan_rng.integers(2**63 - 1, dtype=np.int64))
                assert seed not in used_seeds
                used_seeds.add(seed)
                sample = (sample_postartifact_gauge_v1(v9, [base * VARIANTS + variant], seed, SIDE)
                          if version == 'v9' else
                          sample_fov_balance_v10(v10, [base * VARIANTS + variant], seed, SIDE))
                accepted = bool(sample['eligible'][0])
                provenance = sample['provenance'][0]
                draws.write(json.dumps({'step': step, 'slot': slot, 'version': version,
                    'base': base, 'variant': variant, 'seed': seed,
                    'accepted': accepted, 'provenance': provenance}, allow_nan=False) + '\n')
                if accepted:
                    break
            physical_id = provenance['physical_section_id']
            assert provenance['split'] == 'train' and physical_id not in used_ids
            used_ids.add(physical_id)
            batch_plan.append({'version': version, 'base': base, 'variant': variant,
                               'seed': seed, 'physical_section_id': physical_id})
        plan.append(batch_plan)
with (RUN / 'schedule.jsonl').open('w', encoding='utf8') as file:
    for step, batch_plan in enumerate(plan, 1):
        file.write(json.dumps({'step': step, 'slots': batch_plan}) + '\n')


def alias_target(state, reflection):
    center, frame, basis = full_frame_state_to_components(state)
    ouv = frame_to_physical_ouv(center, frame, basis).reshape(-1, 3, 3)
    end = (SIDE - 1) / SIDE
    alternate_ouv = torch.stack((ouv[:, 0] + end * ouv[:, 1], -ouv[:, 1], ouv[:, 2]), 1)
    alternate = full_frame_state_from_components(*physical_ouv_to_frame(alternate_ouv))
    return alternate, 1 - reflection


def losses(model, samples, seeds):
    inputs = torch.cat([sample['inputs'] for sample in samples])
    target_state = torch.cat([sample['state'] for sample in samples])
    reflection = torch.cat([sample['reflection'] for sample in samples])
    valid = torch.cat([sample['valid_mask'] for sample in samples])
    centres = torch.cat([sample['centre'] for sample in samples])
    prediction = model.predict(inputs)
    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
    reflected = flags.expand(len(states), model.modes, 2)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    sites = torch.stack([torch.multinomial(valid[i].flatten().float(), 128,
        replacement=True, generator=torch.Generator(device='cuda').manual_seed(seed))
        for i, seed in enumerate(seeds)])
    chart = torch.stack((sites.remainder(SIDE),
        sites.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
    truth = centres.flatten(1, 2).gather(1, sites[..., None].expand(-1, -1, 3))
    proposed = rigid_points_090(states, reflected, chart[:, None, None], (SIDE, SIDE))
    five = rigid_points_090(states, reflected, corners, (SIDE, SIDE))
    true_five = rigid_points_090(target_state, reflection, corners, (SIDE, SIDE))
    tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
    five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
    normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
    normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
    target_mass = F.softmax(-physical.detach() / .7, -1)
    nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
    near_mode = model.base_modes + nearest.indices
    near_cost = physical.reshape(len(inputs), model.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    physical_loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
        + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
        + .5 * physical[:, :2 * model.base_modes].min(-1).values
        + (neighbourhood * near_cost).sum(-1)).mean()
    return physical_loss, prediction, target_state, reflection, chart


def readout(model, arm, step):
    model.eval()
    rows = []
    with torch.inference_mode():
        for cohort, panel in panels.items():
            for start in range(0, len(panel), 4):
                group = panel[start:start + 4]
                inputs = torch.stack([case['inputs'] for case in group]).cuda()
                chart = torch.stack([case['chart'] for case in group]).cuda()
                truth = torch.stack([case['truth'] for case in group]).cuda()
                prediction = model.predict(inputs)
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflected = flags.expand(len(states), model.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                points = rigid_points_090(states, reflected, chart[:, None, None], (SIDE, SIDE))
                errors = (points - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                first = prior.argmax(-1)
                top16 = prior.topk(16, -1).indices
                for i, case in enumerate(group):
                    rows.append({'cohort': cohort, **case['meta'],
                        'first_gap_mm': float(errors[i, first[i]]),
                        'top16_oracle_gap_mm': float(errors[i, top16[i]].min())})
    with (RUN / f'{arm}_rows_{step:04d}.jsonl').open('w', encoding='utf8') as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + '\n')
    metrics = {}
    for cohort in panels:
        group = [row for row in rows if row['cohort'] == cohort]
        errors = np.array([row['first_gap_mm'] for row in group])
        metrics[cohort] = {'n': len(group), 'first_mean_mm': float(errors.mean()),
            'first_p90_mm': float(np.quantile(errors, .9)),
            'first_over5mm': int((errors > 5).sum()),
            'top16_within1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5 for row in group),
            'base_first_mean_mm': {str(base): float(np.mean([
                row['first_gap_mm'] for row in group if row['base'] == base]))
                for base in range(56, 64)}}
    model.train()
    print(json.dumps({'arm': arm, 'step': step, 'metrics': metrics}), flush=True)
    return metrics


model, _ = load_one_shot_checkpoint(PARENT, device='cuda')
assert model.modes == 80
model.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
for module in (*pose_modules, *shared_modules):
    module.requires_grad_(True)
parameters = [p for module in (*pose_modules, *shared_modules) for p in module.parameters()]
model.train()
first_samples = [(sample_postartifact_gauge_v1(v9, [slot['base'] * VARIANTS + slot['variant']],
    slot['seed'], SIDE) if slot['version'] == 'v9' else
    sample_fov_balance_v10(v10, [slot['base'] * VARIANTS + slot['variant']], slot['seed'], SIDE))
    for slot in plan[0]]
physical, prediction, target_state, reflection, chart = losses(
    model, first_samples, [slot['seed'] for slot in plan[0]])
alternate, alternate_reflection = alias_target(target_state, reflection)
alias_gap_um = (rigid_points_090(target_state, reflection, chart, (SIDE, SIDE)) -
    rigid_points_090(alternate, alternate_reflection, chart, (SIDE, SIDE))).abs().max()
assert float(alias_gap_um) < .25 and prediction['state'].dtype == torch.float32
log_likelihood = torch.logaddexp(
    model.component_log_prob(prediction, target_state, reflection).logsumexp(-1),
    model.component_log_prob(prediction, alternate, alternate_reflection).logsumexp(-1))
nll = -log_likelihood.mean()
physical_grad = torch.autograd.grad(physical, parameters, retain_graph=True, allow_unused=True)
nll_grad = torch.autograd.grad(nll, parameters, allow_unused=True)
assert bool(torch.isfinite(physical) & torch.isfinite(nll))
assert all(g is None or bool(torch.isfinite(g).all()) for g in (*physical_grad, *nll_grad))
physical_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in physical_grad if g is not None))
nll_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in nll_grad if g is not None))
assert physical_norm > 0 and nll_norm > 0
nll_weight = .25 * physical_norm / nll_norm
(RUN / 'preflight.json').write_text(json.dumps({'alias_max_abs_um': float(alias_gap_um),
    'physical_loss': float(physical), 'quotient_nll': float(nll),
    'physical_gradient_norm': physical_norm, 'nll_gradient_norm': nll_norm,
    'fixed_nll_weight': nll_weight, 'weighted_nll_gradient_fraction': .25,
    'preflight_section_ids': [slot['physical_section_id'] for slot in plan[0]],
    'parent_sha256': parent_sha}, indent=2), encoding='utf8')
del model, prediction, physical, nll, physical_grad, nll_grad, first_samples
del target_state, reflection, chart, alternate, alternate_reflection, log_likelihood, alias_gap_um
torch.cuda.empty_cache()

with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(PARENT, map_location='cpu', weights_only=True)
config = {'seed': SEED, 'parent_sha256': parent_sha, 'source_sha256': hashlib.sha256(
    Path(__file__).read_bytes()).hexdigest(), 'batches_per_arm': UPDATES,
    'batch_slots': ['v9', 'v10', 'v10'], 'synthetic_train_bases': list(range(56)),
    'synthetic_inner_bases': list(range(56, 64)), 'checkpoint_steps': [1500],
    'readout_steps': [0, 500, 1500], 'nll_weight': nll_weight,
    'equivalence': 'O,U,V -> O + (191/192)U,-U,V and reflection -> 1-reflection',
    'acquired_gradients': False, 'calibrated': False,
    'v9_input_sha256': v9['provenance']['input_sha256'],
    'model_source_sha256': hashlib.sha256(Path(__file__).with_name(
        'arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest(),
    'postartifact_source_sha256': hashlib.sha256(Path(sample_postartifact_gauge_v1.__code__
        .co_filename).read_bytes()).hexdigest(),
    'v10_source_sha256': hashlib.sha256(Path(sample_fov_balance_v10.__code__
        .co_filename).read_bytes()).hexdigest(),
    'panel_sha256': hashlib.sha256((RUN / 'panels.pt').read_bytes()).hexdigest(),
    'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'training_draws.jsonl').read_bytes()).hexdigest()}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

readouts, first_arm_hashes = {}, []
for arm in ('control', 'quotient_nll'):
    torch.manual_seed(SEED + 10)
    torch.cuda.manual_seed_all(SEED + 10)
    model, _ = load_one_shot_checkpoint(PARENT, device='cuda')
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    for module in (*pose_modules, *shared_modules):
        module.requires_grad_(True)
    groups = [
        {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 2e-5},
        {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 5e-6},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    for group in optimizer.param_groups:
        group['lr'] = .2 * group['base_lr']
    parameters = [p for group in groups for p in group['params']]
    readouts[arm] = {}
    model.train()
    readouts[arm][0] = readout(model, arm, 0)
    started = time.perf_counter()
    with (RUN / f'{arm}_training.jsonl').open('w', encoding='utf8') as file:
        for step, batch_plan in enumerate(plan, 1):
            samples = [(sample_postartifact_gauge_v1(v9,
                [slot['base'] * VARIANTS + slot['variant']], slot['seed'], SIDE)
                if slot['version'] == 'v9' else sample_fov_balance_v10(v10,
                [slot['base'] * VARIANTS + slot['variant']], slot['seed'], SIDE))
                for slot in batch_plan]
            assert all(bool(sample['eligible'][0]) and
                sample['provenance'][0]['physical_section_id'] == slot['physical_section_id']
                for sample, slot in zip(samples, batch_plan))
            image_hash = hashlib.sha256(torch.cat([sample['inputs'] for sample in samples]
                ).cpu().numpy().tobytes()).hexdigest()
            if arm == 'control':
                first_arm_hashes.append(image_hash)
            else:
                assert image_hash == first_arm_hashes[step - 1]
            physical, prediction, target_state, reflection, chart = losses(
                model, samples, [slot['seed'] for slot in batch_plan])
            if arm == 'quotient_nll':
                alternate, alternate_reflection = alias_target(target_state, reflection)
                nll = -torch.logaddexp(
                    model.component_log_prob(prediction, target_state, reflection).logsumexp(-1),
                    model.component_log_prob(prediction, alternate,
                                             alternate_reflection).logsumexp(-1)).mean()
                loss = physical + nll_weight * nll
            else:
                nll, loss = None, physical
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            file.write(json.dumps({'step': step, 'physical_loss': float(physical.detach()),
                'quotient_nll': None if nll is None else float(nll.detach()),
                'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'image_sha256': image_hash, 'elapsed_seconds': time.perf_counter() - started},
                allow_nan=False) + '\n')
            if step in (500, 1500):
                readouts[arm][step] = readout(model, arm, step)
                file.flush()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'step': parent['step'] + UPDATES, 'local_step': UPDATES,
        'arm': arm, 'config': config, 'calibrated': False}, RUN / f'{arm}_step_1500.pt')
    del model, optimizer
    torch.cuda.empty_cache()

control, treatment = readouts['control'][1500], readouts['quotient_nll'][1500]
a, b = control['v10'], treatment['v10']
gate = {'v10_mean_gain_ge0p25mm': a['first_mean_mm'] - b['first_mean_mm'] >= .25,
    'v10_over5mm_nonregression': b['first_over5mm'] <= a['first_over5mm'],
    'v10_p90_nonregression': b['first_p90_mm'] <= a['first_p90_mm'],
    'v10_top16_nonregression': b['top16_within1p5mm'] >= a['top16_within1p5mm'],
    'v10_nonworse_bases_ge6': sum(b['base_first_mean_mm'][str(i)] <=
        a['base_first_mean_mm'][str(i)] for i in range(56, 64)) >= 6}
summary = {'gate_pass': all(gate.values()), 'gate': gate, 'readouts': readouts,
    'interpretation': 'TRAIN-inner synthetic screen only; acquired donor guard remains required',
    'calibrated': False, 'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in RUN.iterdir() if path.is_file() and path.name != 'completed.json'}}
(RUN / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'gate_pass': summary['gate_pass'], 'gate': gate}), flush=True)
