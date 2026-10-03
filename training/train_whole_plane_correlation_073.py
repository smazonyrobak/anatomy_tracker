"""Blind whole-plane correlation pilot on fresh arbitrary-plane synthetic sections."""
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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.arbitrary_plane_whole_correlation_073 import WholePlaneCorrelationHead

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/whole_plane_correlation_073_pilot_002'
seed, batches, side = 2026100373, 2500, 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
parent_model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
parent_model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
parent_model.requires_grad_(False)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
head = WholePlaneCorrelationHead().cuda().train()
optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
normal_shifts = torch.arange(-1500, 1501, 500, device='cuda').float()
rolls = torch.arange(12, device='cuda').float() * (math.pi / 6)
scales = torch.tensor([.65, 1., 1.5], device='cuda')
shifts = torch.arange(-13, 14, device='cuda').float() / 16
chart_axis = (torch.arange(64, device='cuda').float() + .5) / 16 - 2
by, bx = torch.meshgrid(chart_axis, chart_axis, indexing='ij')
chart = torch.stack((bx, by), -1)
psf = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
psf_weights = torch.tensor([1., 2., 2., 2., 1.], device='cuda') / 8
points_xy = torch.tensor([[0., 0.], [255 / 256, 0.], [0., 255 / 256],
                          [255 / 256, 255 / 256], [127.5 / 256, 127.5 / 256]], device='cuda')
fit_angles = torch.arange(720, device='cuda').float() * (math.pi / 360)
fit_cos, fit_sin = fit_angles.cos(), fit_angles.sin()


def branch_states(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
    state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
    centre, frame, basis = full_frame_state_to_components(state)
    edge = frame[0, :, :, :2] @ basis[0]
    edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
    return choice, centre[0], edge, frame[0, :, :, 2]


def truth_label(sample, centre, edge, normal):
    true_centre, true_frame, true_basis = full_frame_state_to_components(sample['state'])
    true_edge = true_frame[0, :, :2] @ true_basis[0]
    true_xy = points_xy.clone()
    if bool(sample['reflection'][0]):
        true_xy[:, 0] = 255 / 256 - true_xy[:, 0]
    target = true_centre[0] + (true_xy - .5) @ true_edge.T
    x = points_xy - points_xy.mean(0)
    y = target - target.mean(0)
    best = None
    for branch in range(14):
        u, v = edge[branch, :, 0], edge[branch, :, 1]
        b0 = x[:, 0, None] * u + x[:, 1, None] * v
        b90 = -x[:, 1, None] * u + x[:, 0, None] * v
        a, b = (y * b0).sum(), (y * b90).sum()
        c, d, e = b0.square().sum(), b90.square().sum(), (b0 * b90).sum()
        dot = fit_cos * a + fit_sin * b
        norm = fit_cos.square() * c + fit_sin.square() * d + 2 * fit_cos * fit_sin * e
        angle = int((dot.clamp_min(0).square() / norm.clamp_min(1e-9)).argmax())
        scale = dot[angle].clamp_min(0) / norm[angle]
        delta = target.mean(0) - centre[branch]
        dz = delta @ normal[branch]
        inplane = torch.linalg.lstsq(edge[branch], delta - dz * normal[branch]).solution
        mean_q = points_xy.mean(0) - .5
        shift = inplane - scale * torch.stack((
            fit_cos[angle] * mean_q[0] - fit_sin[angle] * mean_q[1],
            fit_sin[angle] * mean_q[0] + fit_cos[angle] * mean_q[1]))
        zi = int((normal_shifts - dz).abs().argmin())
        ai = int((((rolls - fit_angles[angle] + math.pi) % (2 * math.pi))
                  - math.pi).abs().argmin())
        si = int((scales - scale).abs().argmin())
        txi = int((shifts - shift[0]).abs().argmin())
        tyi = int((shifts - shift[1]).abs().argmin())
        q = points_xy - .5
        co, sn = rolls[ai].cos(), rolls[ai].sin()
        mapped_chart = torch.stack((shifts[txi] + scales[si] * (co * q[:, 0] - sn * q[:, 1]),
                                    shifts[tyi] + scales[si] * (sn * q[:, 0] + co * q[:, 1])), -1)
        mapped = centre[branch] + normal[branch] * normal_shifts[zi] + mapped_chart @ edge[branch].T
        error = float((mapped - target).norm(dim=-1).mean())
        if best is None or error < best[0]:
            best = (error, branch, zi, ai, si, tyi, txi)
    return best


def atlas_planes(centre, edge, normal):
    base = centre + torch.einsum('ij,hwj->hwi', edge, chart)
    physical = base[None, None] + normal * (
        normal_shifts[:, None, None, None, None] + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    return rendered, rendered[:, 1:2].clamp(0, 1)


run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': 1,
    'parent_sha256': sha(parent), 'synthetic_provenance': context['provenance'],
    'normal_shifts_um': normal_shifts.tolist(), 'roll_degrees': 30,
    'relative_scales': scales.tolist(), 'translation_bins_edge_units': shifts.tolist(),
    'training': 'blind fixed-lattice complete-plane correlation; label nearest rounded truth fit, no truth candidate insertion',
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
        ('train_whole_plane_correlation_073.py', 'arbitrary_plane_whole_correlation_073.py',
         'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_one_shot_model.py')},
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'head_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        while True:
            virtual = [int(torch.randint(len(context['subjects']), (1,), generator=subject_rng))]
            sample = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            draws.write(json.dumps({**sample['provenance'][0], 'batch': step,
                'used': bool(sample['eligible'][0])}) + '\n')
            if bool(sample['eligible'][0]):
                break
        with torch.no_grad():
            prediction = parent_model.predict(sample['inputs'])
            choice, centre, edge, normal = branch_states(prediction)
            label = truth_label(sample, centre, edge, normal)
        query, mask_logits = head.query_features(prediction)
        scores = []
        for branch in range(14):
            with torch.no_grad():
                rendered, support = atlas_planes(centre[branch], edge[branch], normal[branch])
                atlas_feature = parent_model.atlas_encoder(rendered)
            scores.append(head.score(atlas_feature, support, query, mask_logits))
        logits = torch.cat(scores, 0).reshape(-1)
        _, branch, zi, ai, si, tyi, txi = label
        positive = (((((branch * 7 + zi) * 12 + ai) * 3 + si) * 27 + tyi) * 27 + txi)
        mask_target = F.adaptive_avg_pool2d(sample['valid_mask'][:, None].float(), mask_logits.shape[-2:])
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target,
            pos_weight=torch.tensor(4., device='cuda'))
        retrieval_loss = torch.logsumexp(logits, 0) - logits[positive]
        loss = retrieval_loss + .1 * mask_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            rank = int((logits.detach() > logits.detach()[positive]).sum()) + 1
            row = {'batch': step, 'loss': float(loss.detach()),
                'retrieval_loss': float(retrieval_loss.detach()),
                'mask_loss': float(mask_loss.detach()), 'positive_rank': rank,
                'label_five_point_um': label[0], 'gradient_norm': float(gradient),
                'elapsed_s': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            log.flush()
            draws.flush()
            if step == 1 or step % 250 == 0:
                print(json.dumps(row), flush=True)
        if step in (250, 750, 1500, batches):
            save(step)
print(json.dumps({'finished_batches': batches, 'elapsed_s': time.perf_counter() - started}), flush=True)
