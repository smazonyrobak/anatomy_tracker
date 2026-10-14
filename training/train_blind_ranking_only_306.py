"""Train only the image-conditioned probabilities of frozen 305 plane proposals."""
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

from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/BLIND_RANKING_ONLY_306_PROTOCOL_20261014.md'
parent_run = root / 'runs/joint_spatial_305_acquired_rehearsal_001'
checkpoint = parent_run / 'intervention/pose_local_00400.pt'
output = root / 'runs/blind_ranking_only_306_001'
assert source.drive.upper() == output.drive.upper() == 'I:' and not output.exists()
parent_receipt = json.loads((parent_run / 'intervention/completed.json').read_text())
with checkpoint.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_sha == parent_receipt['output_sha256'][checkpoint.name]

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
assert not model.shared_anchor_pose and not model.anchor_specific_features
model.eval().requires_grad_(False)
before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
layers = {'pose.2': (model.pose[-1], 21, (19, 20)),
          'anchor_pose.2': (model.anchor_pose[-1], 18, (16, 17)),
          'anchor_global.2': (model.anchor_global[-1], 18, (16, 17))}
allowed = {}
parameters = []
for name, (layer, stride, columns) in layers.items():
    rows = torch.tensor([i * stride + j for i in range(layer.out_features // stride)
                         for j in columns], device='cuda')
    mask = torch.zeros(layer.out_features, device='cuda')
    mask[rows] = 1
    layer.weight.requires_grad_(True)
    layer.bias.requires_grad_(True)
    layer.weight.register_hook(lambda gradient, mask=mask: gradient * mask[:, None])
    layer.bias.register_hook(lambda gradient, mask=mask: gradient * mask)
    parameters.extend((layer.weight, layer.bias))
    allowed[name + '.weight'] = rows.cpu()
    allowed[name + '.bias'] = rows.cpu()
optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0)
context = load_streaming_synthetic_v7_64('cuda')
seed = 20261014307
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
config = {'candidate': 306, 'seed': seed, 'optimizer_batches': 800,
    'synthetic_per_batch': 4, 'readouts': [0, 200, 400, 800],
    'train_bases': list(range(56)), 'inner_bases': list(range(56, 64)),
    'panels': ['inner', 'confirmation'], 'sections_per_panel_base': 8,
    'site_samples_per_section': 128, 'soft_target_temperature_mm': .35,
    'expected_error_coefficient': .5, 'optimizer_lr': 5e-5,
    'trainable_parameter_rows': {name: rows.tolist() for name, rows in allowed.items()},
    'checkpoint_sha256': checkpoint_sha,
    'parent_completion_sha256': hashlib.sha256((parent_run / 'completed.json').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'generator_provenance': context['provenance'],
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
output.mkdir(parents=True, exist_ok=False)
(output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

panels = {'inner': [], 'confirmation': []}
with (output / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for panel_name in panels:
        for base in range(56, 64):
            accepted = attempted = 0
            while accepted < 8:
                attempted += 1
                variant = int(rng.integers(VARIANTS))
                draw_seed = int(rng.integers(2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sample_postartifact_gauge_v1(
                        context, [base * VARIANTS + variant], draw_seed, 192)
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'kind': 'panel', 'panel': panel_name,
                    'base': base, 'attempt': attempted, 'accepted_slot': accepted if used else None,
                    'variant': variant, 'seed': draw_seed, 'accepted': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if not used:
                    continue
                accepted += 1
                sites = torch.multinomial(sample['valid_mask'][0].flatten().float(),
                    128, replacement=True, generator=torch.Generator(device='cuda').manual_seed(draw_seed))
                chart = torch.stack((sites.remainder(192),
                    sites.div(192, rounding_mode='floor')), -1).float() / 192
                panels[panel_name].append({'inputs': sample['inputs'][0].cpu(),
                    'chart': chart.cpu(),
                    'truth': sample['centre'][0].reshape(-1, 3)[sites].cpu(),
                    'base': base,
                    'physical_section_id': sample['provenance'][0]['physical_section_id'],
                    'obliquity_deg': sample['provenance'][0].get('obliquity_deg'),
                    'visible_fraction': float(sample['valid_mask'][0].float().mean())})
    assert all(len(panel) == 64 and len({row['physical_section_id'] for row in panel}) == 64
               for panel in panels.values())
    torch.save(panels, output / 'panels.pt')
    draws.flush()
    print(json.dumps({'event': 'panels_frozen', 'sections_each': 64}), flush=True)
with torch.inference_mode():
    initial_state = model.predict(panels['inner'][0]['inputs'][None].cuda())['state'].cpu()

readouts = {}
flags = torch.arange(2, device='cuda')[None, None]
with (output / 'training.jsonl').open('w', encoding='utf8') as training, \
        (output / 'readouts.jsonl').open('w', encoding='utf8') as readout_file, \
        (output / 'draws.jsonl').open('a', encoding='utf8') as draws:
    for step in range(801):
        if step:
            batch = []
            for slot in range(4):
                attempted = 0
                while True:
                    attempted += 1
                    base = int(rng.integers(56))
                    variant = int(rng.integers(VARIANTS))
                    draw_seed = int(rng.integers(2**63 - 1, dtype=np.int64))
                    with torch.no_grad():
                        sample = sample_postartifact_gauge_v1(
                            context, [base * VARIANTS + variant], draw_seed, 192)
                    used = bool(sample['eligible'][0])
                    draws.write(json.dumps({'kind': 'train', 'step': step, 'slot': slot,
                        'attempt': attempted, 'base': base, 'variant': variant,
                        'seed': draw_seed, 'accepted': used,
                        'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                    if used:
                        break
                sites = torch.multinomial(sample['valid_mask'][0].flatten().float(),
                    128, replacement=True)
                chart = torch.stack((sites.remainder(192),
                    sites.div(192, rounding_mode='floor')), -1).float() / 192
                batch.append((sample['inputs'][0], chart,
                    sample['centre'][0].reshape(-1, 3)[sites]))
            inputs = torch.stack([row[0] for row in batch])
            chart = torch.stack([row[1] for row in batch])
            truth = torch.stack([row[2] for row in batch])
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(4, model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            with torch.no_grad():
                error = (rigid_points_090(states, reflections, chart[:, None, None],
                    (192, 192)) - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                target = F.softmax(-error / .35, -1)
            loss = (F.kl_div(prior, target, reduction='batchmean')
                + .5 * (prior.exp() * error).sum(-1).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            training.write(json.dumps({'step': step, 'presentations': 4 * step,
                'loss': float(loss.detach()),
                'expected_error_mm': float((prior.exp() * error).sum(-1).mean().detach())},
                allow_nan=False) + '\n')
        if step in (0, 200, 400, 800):
            rows = []
            with torch.inference_mode():
                for start in range(0, 64, 4):
                    group = panels['inner'][start:start + 4]
                    inputs = torch.stack([row['inputs'] for row in group]).cuda()
                    chart = torch.stack([row['chart'] for row in group]).cuda()
                    truth = torch.stack([row['truth'] for row in group]).cuda()
                    prediction = model.predict(inputs)
                    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                    reflections = flags.expand(4, model.modes, 2)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    errors = (rigid_points_090(states, reflections, chart[:, None, None],
                        (192, 192)) - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                    first = prior.argmax(-1)
                    top = prior.topk(16, -1).indices
                    for j, item in enumerate(group):
                        rows.append({'physical_section_id': item['physical_section_id'],
                            'base': item['base'], 'obliquity_deg': item['obliquity_deg'],
                            'visible_fraction': item['visible_fraction'],
                            'first_error_mm': float(errors[j, first[j]]),
                            'top16_oracle_mm': float(errors[j, top[j]].min()),
                            'first_branch': int(first[j])})
            means = {str(base): float(np.mean([row['first_error_mm'] for row in rows
                if row['base'] == base])) for base in range(56, 64)}
            metrics = {'first_equal_base_mean_mm': float(np.mean(list(means.values()))),
                'first_by_base_mm': means,
                'top16_capture_1p5mm': sum(row['top16_oracle_mm'] <= 1.5 for row in rows)}
            readouts[str(step)] = metrics
            readout_file.write(json.dumps({'step': step, 'metrics': metrics, 'rows': rows},
                allow_nan=False) + '\n')
            readout_file.flush()
            training.flush()
            draws.flush()
            torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config,
                'calibrated': False}, output / f'rank_step_{step:05d}.pt')
            print(json.dumps({'event': 'readout', 'step': step, 'metrics': metrics}), flush=True)

for name, value in model.state_dict().items():
    original = before[name]
    current = value.detach().cpu()
    if name in allowed:
        unchanged = torch.ones(len(current), dtype=torch.bool)
        unchanged[allowed[name]] = False
        assert torch.equal(current[unchanged], original[unchanged])
    else:
        assert torch.equal(current, original), name
with torch.inference_mode():
    assert torch.equal(model.predict(panels['inner'][0]['inputs'][None].cuda())['state'].cpu(),
                       initial_state)
best = min(readouts, key=lambda step: (readouts[step]['first_equal_base_mean_mm'], int(step)))
inner_gain = readouts['0']['first_equal_base_mean_mm'] - readouts[best]['first_equal_base_mean_mm']
confirm_rows = {}
if inner_gain >= .25:
    for name, path in (('parent', checkpoint), ('selected', output / f'rank_step_{int(best):05d}.pt')):
        evaluator, _ = load_one_shot_checkpoint(path, device='cuda')
        rows = []
        with torch.inference_mode():
            for start in range(0, 64, 4):
                group = panels['confirmation'][start:start + 4]
                inputs = torch.stack([row['inputs'] for row in group]).cuda()
                chart = torch.stack([row['chart'] for row in group]).cuda()
                truth = torch.stack([row['truth'] for row in group]).cuda()
                prediction = evaluator.predict(inputs)
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(4, evaluator.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                errors = (rigid_points_090(states, reflections, chart[:, None, None],
                    (192, 192)) - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                first = prior.argmax(-1)
                top = prior.topk(16, -1).indices
                for j, item in enumerate(group):
                    rows.append({'physical_section_id': item['physical_section_id'],
                        'base': item['base'], 'first_error_mm': float(errors[j, first[j]]),
                        'top16_oracle_mm': float(errors[j, top[j]].min()),
                        'first_branch': int(first[j])})
        confirm_rows[name] = rows
        del evaluator
confirm = {}
gate = False
if confirm_rows:
    confirm = {name: {'mean_mm': float(np.mean([row['first_error_mm'] for row in rows])),
        'by_base_mm': {str(base): float(np.mean([row['first_error_mm'] for row in rows
            if row['base'] == base])) for base in range(56, 64)},
        'top16_capture_1p5mm': sum(row['top16_oracle_mm'] <= 1.5 for row in rows)}
        for name, rows in confirm_rows.items()}
    gate = (confirm['parent']['mean_mm'] - confirm['selected']['mean_mm'] >= .25
        and sum(confirm['selected']['by_base_mm'][str(base)] <
                confirm['parent']['by_base_mm'][str(base)] for base in range(56, 64)) >= 6
        and confirm['selected']['top16_capture_1p5mm'] >=
            confirm['parent']['top16_capture_1p5mm'] - 4)
summary = {'selected_step': int(best), 'inner_gain_mm': inner_gain,
    'inner_readouts': readouts, 'confirmation': confirm, 'confirmation_rows': confirm_rows,
    'synthetic_gate_pass': gate, 'candidate_geometry_exactly_frozen': True,
    'real_guard_required_if_synthetic_passes': gate,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(output / 'completed.json').write_text(json.dumps({'summary': summary,
    'output_sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in output.iterdir() if file.name != 'completed.json'}}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'selected_step': int(best),
    'inner_gain_mm': inner_gain, 'synthetic_gate_pass': gate}), flush=True)
