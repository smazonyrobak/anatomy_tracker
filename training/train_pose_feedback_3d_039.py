"""Short fresh-section pilot of image-to-3D-atlas matching and geometric pose fitting."""
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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.pose_feedback_3d_039 import PoseFeedback3D039

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/pose_feedback_3d_039_pilot_v2'
seed, updates, sections, side, beam = 2026103900, 500, 2, 256, 4
torch.set_num_threads(4)
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
head = PoseFeedback3D039().cuda().train()
optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': sections,
          'parent': str(parent), 'parent_sha256': sha(parent), 'prior_beam_training': beam,
          'candidate_set': 'actual prior top4 plus two independent true-plane perturbations and exact true plane',
          'atlas_context': 'finite-PSF 13 depths at -6 to +6 mm, 24x24 keys over 2x candidate field',
          'method': 'trainable image/3D-atlas descriptor, dense correspondence supervision, robust weighted geometric fit',
          'real_training_images': 0, 'synthetic_provenance': context['provenance'],
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_pose_feedback_3d_039.py', 'pose_feedback_3d_039.py',
                             'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
                             'arbitrary_plane_full_frame_primitives.py',
                             'arbitrary_plane_streaming_synthetic_v7_64.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
perturb_scale = torch.tensor((.16, .16, .16, 1400., 1400., 1400., .10, .10, .08), device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi',
        frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


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
        with torch.no_grad():
            prediction = model.predict(batch['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            chosen = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            perturb = torch.randn((sections, 2, 9), device='cuda') * perturb_scale
            teacher = compose_full_frame_state(batch['state'][:, None].expand(-1, 3, -1),
                torch.cat((perturb, perturb.new_zeros(sections, 1, 9)), 1))
            initial = torch.cat((chosen, teacher), 1)
            reflection = torch.cat((choice % 2, batch['reflection'][:, None].expand(-1, 3)), 1)
        result = head(prediction, batch['inputs'], initial, reflection,
                      context['atlas'], batch['offsets'], batch['weights'])
        truth_grid = batch['centre'][:, 8::16, 8::16].reshape(sections, 256, 3)
        valid_grid = batch['valid_mask'][:, 8::16, 8::16].reshape(sections, 256).bool()
        with torch.no_grad():
            distances = torch.cdist(truth_grid[:, None].expand(-1, beam + 3, -1, -1)
                .reshape(sections * (beam + 3), 256, 3),
                result['key_world'].reshape(sections * (beam + 3), -1, 3))
            nearest_distance, nearest = distances.min(-1)
            nearest = nearest.reshape(sections, beam + 3, 256)
            nearest_distance = nearest_distance.reshape(sections, beam + 3, 256)
            support = result['key_support'].gather(-1, nearest)
            in_range = (valid_grid[:, None] & (nearest_distance <= 1500) & (support > .1))
            ids = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            truth128 = batch['centre'].reshape(sections, -1, 3).gather(
                1, ids[..., None].expand(-1, -1, 3))
            chart128 = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                    -1).float() / side
        ce = F.cross_entropy(result['logits'].reshape(-1, result['logits'].shape[-1]),
                             nearest.flatten(), reduction='none').reshape_as(in_range)
        correspondence_loss = (ce * in_range).sum() / in_range.sum().clamp_min(1)
        expected_error = F.smooth_l1_loss(result['matched_world'] / 1000,
            truth_grid[:, None].expand(-1, beam + 3, -1, -1) / 1000,
            beta=.5, reduction='none').sum(-1)
        expected_loss = (expected_error * in_range).sum() / in_range.sum().clamp_min(1)
        spatial = (points(result['state'], reflection, chart128) - truth128[:, None]
                   ).norm(dim=-1).mean(-1)
        pose_loss = F.smooth_l1_loss(spatial / 1000, torch.zeros_like(spatial),
                                     beta=1., reduction='none')
        pose_loss = pose_loss[:, :beam].mean() + .5 * pose_loss[:, beam:].mean()
        visibility_loss = F.binary_cross_entropy_with_logits(result['visibility_logit'],
                                                              valid_grid.float())
        score = prior.gather(1, choice) + result['score_delta'][:, :beam]
        ranking_loss = F.kl_div(F.log_softmax(score, -1),
            F.softmax(-spatial[:, :beam].detach() / 750, -1), reduction='none').sum(-1).mean()
        loss = pose_loss + .15 * correspondence_loss + .25 * expected_loss \
               + .1 * visibility_loss + .25 * ranking_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        picked = score.detach().argmax(-1)
        row = {'step': step, 'synthetic_presentations': step * sections,
               'loss': float(loss.detach()), 'pose_loss': float(pose_loss.detach()),
               'correspondence_loss': float(correspondence_loss.detach()),
               'expected_loss': float(expected_loss.detach()),
               'visibility_loss': float(visibility_loss.detach()),
               'ranking_loss': float(ranking_loss.detach()),
               'in_range_fraction': float(in_range.float().mean()),
               'top1_um': float(spatial[:, 0].detach().mean()),
               'best4_um': float(spatial[:, :beam].detach().min(-1).values.mean()),
               'selected_um': float(spatial[:, :beam].detach().gather(1, picked[:, None]).mean()),
               'teacher_um': float(spatial[:, beam:].detach().mean()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
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
