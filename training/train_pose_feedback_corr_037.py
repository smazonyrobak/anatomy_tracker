"""Train off-plane atlas-correlation pose updates against a matched image-only control."""
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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.pose_feedback_corr_037 import PoseFeedbackCorrelation037

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/pose_feedback_corr_037'
seed, updates, sections, side, beam, fit_side = 2026103700, 4000, 2, 256, 8, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 18000 and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'])
del checkpoint
model.requires_grad_(False)
atlas_head = PoseFeedbackCorrelation037(use_atlas=True).cuda().train()
control_head = PoseFeedbackCorrelation037(use_atlas=False).cuda().train()
control_head.load_state_dict(atlas_head.state_dict())
optimizer = torch.optim.AdamW(list(atlas_head.parameters()) + list(control_head.parameters()),
                              lr=2e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': sections, 'side': side,
          'prior_beam': beam, 'fit_side': fit_side, 'parent': str(parent),
          'parent_sha256': sha(parent), 'frozen': '019 direct prediction, atlas encoder and local mapper',
          'candidate_set': 'actual prior top8 plus three independent perturbed known planes and exact known plane',
          'arms': {'atlas': '16x16 all-pair cross-modal spatial correlation and candidate fit',
                   'control': 'same-width image-to-image correlation, pose/image information, no atlas/fit'},
          'synthetic_provenance': context['provenance'], 'real_training_images': 0,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_pose_feedback_corr_037.py', 'pose_feedback_corr_037.py',
                             'arbitrary_plane_one_shot_model.py',
                             'arbitrary_plane_one_shot_stream.py',
                             'arbitrary_plane_full_frame_primitives.py',
                             'arbitrary_plane_geometry.py',
                             'arbitrary_plane_joint_model_v7.py',
                             'arbitrary_plane_recurrent_model.py',
                             'arbitrary_plane_ribbon_v6.py',
                             'arbitrary_plane_streaming_synthetic_v7_64.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
scales = torch.tensor([[.04, .04, .04, 350., 350., 350., .03, .03, .025],
                       [.12, .12, .12, 1000., 1000., 1000., .08, .08, .06],
                       [.25, .25, .25, 2000., 2000., 2000., .15, .15, .12]], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi', frame[..., :2] @ basis, chart - .5)


def residual_target(initial, reflection, true_state, true_reflection):
    true_centre, true_frame, true_basis = full_frame_state_to_components(true_state)
    true_edges = true_frame[..., :, :2] @ true_basis
    mismatch = reflection != true_reflection[:, None]
    edge_u = torch.where(mismatch[..., None], -true_edges[:, None, :, 0],
                         true_edges[:, None, :, 0])
    edge_v = true_edges[:, None, :, 1].expand_as(edge_u)
    goal_centre = true_centre[:, None] + torch.where(mismatch[..., None],
        -true_edges[:, None, :, 0] / side, torch.zeros_like(edge_u))
    origin = goal_centre - .5 * (edge_u + edge_v)
    fitted = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((origin, edge_u, edge_v), -2)))
    centre, frame, basis = full_frame_state_to_components(initial)
    goal_centre, goal_frame, goal_basis = full_frame_state_to_components(fitted)
    relative = frame.transpose(-1, -2) @ goal_frame
    skew = .5 * torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                             relative[..., 0, 2] - relative[..., 2, 0],
                             relative[..., 1, 0] - relative[..., 0, 1]), -1)
    sine = skew.norm(dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    rotation = skew * (angle / sine.clamp_min(1e-5))[..., None]
    translation = (frame.transpose(-1, -2) @ (goal_centre - centre)[..., None]).squeeze(-1)
    delta_basis = torch.linalg.solve(basis, goal_basis)
    log_scale = delta_basis.diagonal(dim1=-2, dim2=-1).log()
    shear = delta_basis[..., 0, 1] / delta_basis[..., 1, 1]
    update = torch.cat((rotation, translation, log_scale, shear[..., None]), -1)
    reliable = (angle < 2.5)[..., None].expand_as(rotation)
    mask = torch.cat((reliable, torch.ones_like(update[..., 3:], dtype=torch.bool)), -1)
    return update, mask


def arm_loss(head, prediction, first, atlas_feature, inputs, initial, reflection,
             target, chart, true_normal, original, prior, choice, desired, desired_mask):
    refined, score_delta, update = head(prediction, first, inputs, initial,
                                        reflection, atlas_feature)
    spatial = (points(refined, reflection, chart) - target[:, None]).norm(dim=-1).mean(-1)
    normal = full_frame_state_to_components(refined)[1][..., :, 2]
    normal_error = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    physical = spatial + normal_error
    importance = .5 + .5 * torch.exp(-original[:, :beam] / 4000)
    predicted_loss = (importance * F.smooth_l1_loss(physical[:, :beam] / 1000,
        torch.zeros_like(physical[:, :beam]), beta=1., reduction='none')).sum() / importance.sum()
    teacher_loss = F.smooth_l1_loss(physical[:, beam:] / 1000,
        torch.zeros_like(physical[:, beam:]), beta=1.)
    score = prior.gather(1, choice) + score_delta[:, :beam]
    rank_loss = F.kl_div(F.log_softmax(score, -1),
                         F.softmax(-physical[:, :beam].detach() / 750, -1),
                         reduction='none').sum(-1).mean()
    update_penalty = .002 * (update[..., :3] / 1.2).square().mean() \
                     + .002 * (update[..., 3:6] / 5000).square().mean() \
                     + .002 * (update[..., 6:9] / .35).square().mean()
    limit = update.new_tensor((1.2, 1.2, 1.2, 5000., 5000., 5000., .35, .35, .25))
    residual_error = F.smooth_l1_loss(update / limit,
        (desired / limit).clamp(-.98, .98), beta=.2, reduction='none')
    residual_loss = (residual_error * desired_mask).sum() / desired_mask.sum()
    loss = predicted_loss + teacher_loss + .25 * rank_loss + .5 * residual_loss + update_penalty
    selected = score.detach().argmax(-1)
    readout = {'loss': float(loss.detach()), 'spatial_top1_um': float(spatial[:, 0].detach().mean()),
               'spatial_best8_um': float(spatial[:, :beam].detach().min(-1).values.mean()),
               'spatial_selected_um': float(spatial[:, :beam].detach().gather(
                   1, selected[:, None]).mean()),
               'teacher_spatial_um': float(spatial[:, beam:].detach().mean()),
               'rank_loss': float(rank_loss.detach()), 'residual_loss': float(residual_loss.detach()),
               'translation_update_um': float(update[..., 3:6].detach().norm(dim=-1).mean())}
    return loss, readout


def save(step):
    torch.save({'atlas_head': atlas_head.state_dict(), 'control_head': control_head.state_dict(),
                'optimizer': optimizer.state_dict(), 'subjects_rng': subjects_rng.get_state(),
                'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step,
                'config': config, 'calibrated': False}, run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(sections))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                       'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(sections)])
                 for key in accepted[0]}
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
        target = batch['centre'].reshape(sections, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side), indices.div(side, rounding_mode='floor')),
                            -1).float() / side
        with torch.no_grad():
            prediction = model.predict(batch['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            chosen = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            perturb = torch.randn((sections, 3, 9), device='cuda') * scales[None]
            teacher = compose_full_frame_state(batch['state'][:, None].expand(-1, 4, -1),
                torch.cat((perturb, perturb.new_zeros(sections, 1, 9)), 1))
            initial = torch.cat((chosen, teacher), 1)
            reflection = torch.cat((choice % 2, batch['reflection'][:, None].expand(-1, 4)), 1)
            selected = {**prediction, 'state': initial}
            branch = torch.arange(beam + 4, device='cuda')[None].expand(sections, -1)
            first = model.map(selected, batch['offsets'], branch, reflection,
                              (fit_side, fit_side), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=fit_side,
                              source_shape=(side, side))
            atlas_feature = first['refinement_feature'][:, :, 64:128]
            original = (points(initial, reflection, chart) - target[:, None]).norm(dim=-1).mean(-1)
            true_normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
            desired, desired_mask = residual_target(
                initial, reflection, batch['state'], batch['reflection'])
        atlas_loss, atlas_row = arm_loss(atlas_head, prediction, first, atlas_feature,
            batch['inputs'], initial, reflection, target, chart, true_normal, original, prior, choice,
            desired, desired_mask)
        control_loss, control_row = arm_loss(control_head, prediction, first, atlas_feature,
            batch['inputs'], initial, reflection, target, chart, true_normal, original, prior, choice,
            desired, desired_mask)
        optimizer.zero_grad(set_to_none=True)
        (atlas_loss + control_loss).backward()
        atlas_gradient = torch.nn.utils.clip_grad_norm_(
            atlas_head.parameters(), 5., error_if_nonfinite=True)
        control_gradient = torch.nn.utils.clip_grad_norm_(
            control_head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_presentations': step * sections,
               'prior_top1_um': float(original[:, 0].mean()),
               'prior_best8_um': float(original[:, :beam].min(-1).values.mean()),
               'atlas': atlas_row, 'control': control_row,
               'atlas_gradient_norm': float(atlas_gradient),
               'control_gradient_norm': float(control_gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': updates * sections, 'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'), 'config_sha256': sha(run / 'config.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
