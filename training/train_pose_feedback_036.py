"""Train every atlas-conditioned pose branch to reduce physical error."""
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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/pose_feedback_036'
seed, updates, sections, side, beam, first_side = 2026103600, 3000, 2, 256, 8, 64
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
model.pose_refiner.requires_grad_(True)
model.pose_refiner.train()
optimizer = torch.optim.AdamW(model.pose_refiner.parameters(), lr=1e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': sections,
          'side': side, 'prior_beam': beam, 'first_fit_side': first_side,
          'parent': str(parent), 'parent_sha256': sha(parent),
          'frozen_modules': 'all except existing shared pose_refiner',
          'candidate_set': 'actual 019 prior top8 plus three independently perturbed training-only true states plus exact training-only true state',
          'synthetic_provenance': context['provenance'], 'real_training_images': 0,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_pose_feedback_036.py', 'arbitrary_plane_one_shot_model.py',
                             'arbitrary_plane_one_shot_stream.py',
                             'arbitrary_plane_full_frame_primitives.py',
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


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
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
                              (first_side, first_side), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=first_side,
                              source_shape=(side, side))
            original = (points(initial, reflection, chart) - target[:, None]).norm(dim=-1).mean(-1)
        refined, score_delta, update = model.refine(first['refinement_feature'], initial)
        error = (points(refined, reflection, chart) - target[:, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(refined)[1][..., :, 2]
        true_normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        physical = error + normal_penalty
        importance = .2 + .8 * torch.exp(-original[:, :beam] / 4000)
        predicted_loss = (importance * torch.log1p(physical[:, :beam] / 500)).sum() / importance.sum()
        teacher_loss = torch.log1p(physical[:, beam:] / 500).mean()
        predicted_score = prior.gather(1, choice) + score_delta[:, :beam]
        rank_loss = F.kl_div(F.log_softmax(predicted_score, -1),
                             F.softmax(-physical[:, :beam].detach() / 750, -1),
                             reduction='none').sum(-1).mean()
        update_penalty = .003 * (update[..., :3] / .9).square().mean() \
                         + .003 * (update[..., 3:6] / 4000).square().mean() \
                         + .003 * (update[..., 6:9] / .25).square().mean()
        loss = predicted_loss + teacher_loss + .25 * rank_loss + update_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.pose_refiner.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        learned_choice = predicted_score.detach().argmax(-1)
        row = {'step': step, 'synthetic_presentations': step * sections,
               'predicted_initial_top1_um': float(original[:, 0].mean()),
               'predicted_refined_top1_um': float(error[:, 0].detach().mean()),
               'predicted_initial_oracle_um': float(original[:, :beam].min(-1).values.mean()),
               'predicted_refined_oracle_um': float(error[:, :beam].detach().min(-1).values.mean()),
               'predicted_refined_selected_um': float(error[:, :beam].detach().gather(
                   1, learned_choice[:, None]).mean()),
               'teacher_initial_um': float(original[:, beam:].mean()),
               'teacher_refined_um': float(error[:, beam:].detach().mean()),
               'pose_loss': float(predicted_loss.detach()), 'teacher_loss': float(teacher_loss.detach()),
               'rank_loss': float(rank_loss.detach()), 'update_penalty': float(update_penalty.detach()),
               'mean_translation_update_um': float(update[..., 3:6].detach().norm(dim=-1).mean()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
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
