"""Train a scratch image/atlas pose score on distinct synthetic sections and hard wrong poses."""
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

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_plane)
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.arbitrary_plane_structural_matcher_071 import StructuralPoseMatcher

run = root / 'runs/structural_pose_matcher_071_phase1'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
seed, batches, synthetic, source_side, render_side, candidates = 2026100371, 3000, 3, 192, 64, 8
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hypotheses(state, reflection, generator, other_state=None, other_reflection=None):
    batch = len(state)
    jitter = torch.randn(batch, candidates, 9, device='cuda', generator=generator)
    update = torch.zeros_like(jitter)
    update[:, 1, 5] = jitter[:, 1, 5].sign() * (180 + 900 * jitter[:, 1, 5].abs().tanh())
    update[:, 2, 3:5] = 650 * jitter[:, 2, 3:5].tanh()
    update[:, 3, :2] = .35 * jitter[:, 3, :2].tanh()
    update[:, 3, 3:6] = 200 * jitter[:, 3, 3:6].tanh()
    update[:, 4, 2] = 2.5 * jitter[:, 4, 2].tanh()
    update[:, 4, 3:5] = 300 * jitter[:, 4, 3:5].tanh()
    update[:, 5, 3:5] = 200 * jitter[:, 5, 3:5].tanh()
    update[:, 5, 6:8] = .30 * jitter[:, 5, 6:8].tanh()
    update[:, 5, 8] = .25 * jitter[:, 5, 8].tanh()
    update[:, 6, :3] = .28 * jitter[:, 6, :3].tanh()
    update[:, 6, 3:6] = 550 * jitter[:, 6, 3:6].tanh()
    update[:, 6, 6:8] = .18 * jitter[:, 6, 6:8].tanh()
    update[:, 7, :3] = .75 * jitter[:, 7, :3].tanh()
    update[:, 7, 3:6] = 1250 * jitter[:, 7, 3:6].tanh()
    proposal = compose_full_frame_state(
        state[:, None].expand(-1, candidates, -1).reshape(-1, 12),
        update.reshape(-1, 9)).reshape(batch, candidates, 12)
    flags = reflection[:, None].expand(-1, candidates).clone()
    flags[:, 4] = 1 - flags[:, 4]
    if other_state is not None:
        proposal[:, 7] = other_state
        flags[:, 7] = other_reflection
    elif batch > 1:
        proposal[:, 7] = state.roll(1, 0)
        flags[:, 7] = reflection.roll(1, 0)
    else:
        flags[:, 7] = (torch.rand(batch, device='cuda', generator=generator) > .5).long()
    order = torch.rand(batch, candidates, device='cuda', generator=generator).argsort(-1)
    return (proposal.gather(1, order[..., None].expand(-1, -1, 12)),
            flags.gather(1, order))


def physical_error(proposal, flags, truth, reflection, side):
    chart = proposal.new_tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.5, .5]])

    def points(states, mirrored):
        centre, frame, basis = full_frame_state_to_components(states)
        xy = chart.expand(*states.shape[:-1], -1, -1).clone()
        xy[..., 0] = torch.where(mirrored[..., None].bool(), (side - 1) / side - xy[..., 0], xy[..., 0])
        return centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
            frame[..., :, :2] @ basis, xy - .5)

    actual = points(truth[:, None], reflection[:, None])
    return (points(proposal, flags) - actual).norm(dim=-1).mean(-1)


def render(proposal, flags, offsets, weights, atlas):
    batch, count = proposal.shape[:2]
    slab = render_finite_thickness_plane(
        atlas, proposal.flatten(0, 1), (render_side, render_side),
        (0., 0., 0.), (25., 25., 25.),
        offsets[:, None].expand(-1, count, -1).flatten(0, 1),
        weights[:, None].expand(-1, count, -1).flatten(0, 1))
    slab = torch.where(flags.flatten()[:, None, None, None].bool(), slab.flip(-1), slab)
    support = slab[:, 1:2].clamp(0, 1)
    return torch.cat((slab[:, :1] / support.clamp_min(1e-4) * support, support), 1).reshape(
        batch, count, 2, render_side, render_side)


context = load_streaming_synthetic_v7_64(device='cuda')
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
dev_records = [record for record in records if 16 <= record['section_index'] < 24 and record['eligible']]
assert len({record['synthetic_subject_plan_id'] for record in dev_records}) == 8
assert {record['animal_id'] for record in dev_records}.isdisjoint({
    base['lineage']['animal_id'] for base in context['bases']})
dev = []
for record in dev_records:
    path = panel / record['file']
    assert sha(path) == record['sha256']
    with np.load(path, allow_pickle=False) as arrays:
        dev.append((record, {key: torch.from_numpy(arrays[key].copy()).cuda() for key in
            ('inputs', 'target_state', 'reflection', 'offsets_um', 'weights')}))

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
    'source_side': source_side, 'render_side': render_side, 'candidates': candidates,
    'initialization': 'random; no pretrained weights, features or pseudolabels',
    'split': '64 independent synthetic TRAIN maps; frozen DEV plans distinct by synthetic subject',
    'dev_sections': [record['section_id'] for record in dev_records],
    'dev_records_sha256': sha(panel / 'records.jsonl'),
    'synthetic_provenance': context['provenance'],
    'loss': 'listwise physical-error soft targets; full-pose corner/centre error',
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
        ('train_structural_pose_matcher_071.py', 'arbitrary_plane_structural_matcher_071.py',
         'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model = StructuralPoseMatcher().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
draw_seed = seed * 10000000
subject_rng = torch.Generator().manual_seed(seed)
candidate_rng = torch.Generator(device='cuda').manual_seed(seed)
best = float('inf')


def evaluate():
    model.eval()
    errors, ranks, random_errors, modes = [], [], [], []
    with torch.inference_mode():
        for index, (record, arrays) in enumerate(dev):
            truth = arrays['target_state'][None]
            reflection = arrays['reflection'].reshape(1).long()
            local_rng = torch.Generator(device='cuda').manual_seed(
                seed + 100000 * record['section_index'] + int(record['animal_id'].split('-')[-1]))
            other = dev[(index + len(dev) // 2) % len(dev)][1]
            proposal, flags = hypotheses(truth, reflection, local_rng,
                other['target_state'][None], other['reflection'].reshape(1).long())
            expected = physical_error(proposal, flags, truth, reflection, 256)
            rendered = render(proposal, flags, arrays['offsets_um'][None],
                              arrays['weights'][None], context['atlas'])
            image = F.interpolate(arrays['inputs'][None], (render_side, render_side), mode='area')
            score = model(image, rendered)
            selected = score.argmax(-1)
            errors.append(float(expected.gather(1, selected[:, None])[0, 0]))
            random_errors.append(float(expected.mean()))
            ranks.append(int(selected == expected.argmin(-1)))
            modes.append(record['appearance_mode'])
    model.train()
    return {'sections': len(errors), 'selected_um': float(np.mean(errors)),
        'random_order_um': float(np.mean(random_errors)),
        'best_um': 0., 'correct_best': int(sum(ranks)),
        'by_appearance_um': {mode: float(np.mean([value for value, group in zip(errors, modes)
            if group == mode])) for mode in ('raw', 'exact_black', 'imperfect_brush')}}


started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=source_side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'offsets', 'weights')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        proposal, flags = hypotheses(batch['state'], batch['reflection'], candidate_rng)
        error = physical_error(proposal, flags, batch['state'], batch['reflection'], source_side)
        with torch.no_grad():
            rendered = render(proposal, flags, batch['offsets'], batch['weights'], context['atlas'])
        image = F.interpolate(batch['inputs'], (render_side, render_side), mode='area')
        score = model(image, rendered)
        target = F.softmax(-error / 550, -1)
        loss = -(target * F.log_softmax(score, -1)).sum(-1).mean()
        loss += .2 * (F.softmax(score, -1) * error).sum(-1).mean() / 1000
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            log.write(json.dumps({'batch': step, 'training_loss': float(loss.detach()),
                'training_selected_um': float(error.gather(1, score.detach().argmax(-1)[:, None]).mean()),
                'elapsed_s': time.perf_counter() - started}) + '\n')
            log.flush()
        if step % 500 == 0:
            result = evaluate()
            result.update({'batch': step, 'elapsed_s': time.perf_counter() - started})
            log.write(json.dumps(result) + '\n')
            log.flush()
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'batch': step, 'config': config, 'dev': result, 'calibrated': False},
                run / f'batch_{step:05d}.pt')
            if result['selected_um'] < best:
                best = result['selected_um']
                (run / 'best.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
print(json.dumps({'finished_batches': batches, 'best_dev_selected_um': best}), flush=True)
