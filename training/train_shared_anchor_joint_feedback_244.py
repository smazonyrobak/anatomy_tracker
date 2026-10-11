"""Fresh shared-anchor pose, atlas-feedback, and local-map training on I:."""
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
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/shared_anchor_joint_feedback_244_train_001'
PROTOCOL = Path('docs/publication/SHARED_ANCHOR_JOINT_FEEDBACK_244_PROTOCOL_20261012.md')
SEED, UPDATES, SIDE, SYNTHETIC, FEEDBACK_START = 2026101244, 6000, 192, 2, 1000
SAVE_STEPS = (0, 1000, 3000, 6000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
model = OneShotJointSliceModel(
    modes=16, normal_anchor_count=64, shared_anchor_pose=True,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 100000

rng = np.random.default_rng(SEED)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
real_schedule = []
while len(real_schedule) < UPDATES:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            real_schedule.append((int(donor), int(remaining[donor].pop())))
            if len(real_schedule) == UPDATES:
                break
assert len(set(real_schedule)) == UPDATES

RUN.mkdir(parents=True, exist_ok=False)
np.save(RUN / 'real_schedule.npy', np.asarray(real_schedule, np.int32))
source_names = (
    'train_shared_anchor_joint_feedback_244.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_reserved_real_stream_v8.py', 'arbitrary_plane_full_frame_primitives.py',
)
config = {
    'seed': SEED, 'batches': UPDATES, 'synthetic_per_batch': SYNTHETIC,
    'real_per_batch': 1, 'side': SIDE, 'feedback_after_batch': FEEDBACK_START,
    'save_steps': SAVE_STEPS, 'model_parameters': sum(p.numel() for p in model.parameters()),
    'initialization': 'all model weights random from seed; no parent weights/features/pseudolabels',
    'model': {'free_modes': 16, 'antipodal_shared_anchors': 64,
              'atlas_conditioned_map': True, 'shared_pose_updater': True,
              'fitted_candidate_scorer': True},
    'real_label_role': real['label_role'], 'real_schedule_sha256': hashlib.sha256(
        (RUN / 'real_schedule.npy').read_bytes()).hexdigest(),
    'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in source_names},
    'calibrated': False, 'public_benchmark_used': False,
}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

flags = torch.tensor((0, 1), device='cuda')
landmarks = torch.tensor(((0., 0.), (191., 0.), (0., 191.), (191., 191.),
                          (95.5, 95.5)), device='cuda') / SIDE


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (SIDE - 1) / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
                                               frame[..., :2] @ basis, chart - .5)


def mapped_sites(mapped, indices):
    batch, count, height, width = mapped['centre_surface_ccf_ap_dv_ml_um'].shape[:4]
    xy = torch.stack((indices.remainder(SIDE), indices.div(SIDE, rounding_mode='floor')), -1)
    grid = (xy.float() + .5) * (2 / SIDE) - 1
    grid = grid[:, None].expand(-1, count, -1, -1).reshape(batch * count, 1, -1, 2)
    surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
    return F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[:, :, 0].transpose(1, 2).reshape(batch, count, -1, 3)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               RUN / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, UPDATES + 1):
        accepted, pending = {}, list(range(SYNTHETIC))
        while pending:
            subjects = torch.randint(len(context['subjects']), (len(pending),),
                                     generator=subjects_rng).tolist()
            sampled = sample_one_shot_slide_artifacts_v7(context, subjects, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'batch': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                       'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(SYNTHETIC)])
                 for key in accepted[0]}
        donor, section = real_schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({**observation['identities'][0], 'batch': step,
                                'slot': SYNTHETIC, 'used': True,
                                'label_role': real['label_role']}) + '\n')

        image = torch.cat((batch['inputs'], observation['inputs']))
        truth = torch.cat((batch['state'], observation['state']))
        truth_reflection = torch.cat((batch['reflection'], observation['reflection']))
        prediction = model.predict(image)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        branch_reflection = flags[None, None].expand(len(image), model.modes, -1)
        five = (points(states, branch_reflection, landmarks)
                - points(truth, truth_reflection, landmarks)[:, None, None]).norm(dim=-1).mean(-1)
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 96, replacement=True)
        target = batch['centre'].reshape(SYNTHETIC, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(SIDE),
                             indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        dense = (points(states[:SYNTHETIC], branch_reflection[:SYNTHETIC],
                        chart[:, None, None]) - target[:, None, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][..., :, 2]
        normal_cost = 1000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        distance = torch.cat((.75 * dense + .25 * five[:SYNTHETIC]
                              + normal_cost[:SYNTHETIC, :, None],
                              five[SYNTHETIC:] + .25 * normal_cost[SYNTHETIC:, :, None]), 0).flatten(1)
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
        rank = -(F.softmax(-distance.detach() / 1000, -1) * prior).sum(-1)
        expected = (prior.exp() * distance).sum(-1) / 1000
        oracle = distance.min(-1).values / 1000
        nearest_normal = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(2, -1).indices
        anchor_rigid = distance.reshape(len(image), model.modes, 2).min(-1).values.gather(
            1, model.base_modes + nearest_normal).mean(-1) / 1000
        direct = (oracle[:SYNTHETIC] + .2 * expected[:SYNTHETIC]
                  + .05 * rank[:SYNTHETIC] + .2 * anchor_rigid[:SYNTHETIC]).mean()
        direct += .2 * (oracle[SYNTHETIC:] + .2 * expected[SYNTHETIC:]
                         + .05 * rank[SYNTHETIC:] + .2 * anchor_rigid[SYNTHETIC:]).mean()

        source = {key: value[:SYNTHETIC] if torch.is_tensor(value) else value
                  for key, value in prediction.items()}
        teacher = model.map({**source, 'state': batch['state'][:, None]},
                            batch['offsets'], torch.zeros((SYNTHETIC, 1), device='cuda', dtype=torch.long),
                            batch['reflection'][:, None], (64, 64), context['atlas'],
                            batch['weights'], feature_side=64, source_shape=(SIDE, SIDE))
        true_map_error = (mapped_sites(teacher, indices)[:, 0] - target).norm(dim=-1).mean() / 1000
        valid_small = F.interpolate(batch['valid_mask'][:, None].float(), (64, 64), mode='area')
        reliability = F.binary_cross_entropy_with_logits(
            teacher['correspondence_logit'][:, 0], valid_small)
        teacher_field = teacher['local_displacement_um'][:, 0]
        teacher_magnitude = teacher_field.square().mean().sqrt() / 1000
        adjacent_y = valid_small[:, :, 1:] * valid_small[:, :, :-1]
        adjacent_x = valid_small[:, :, :, 1:] * valid_small[:, :, :, :-1]
        smooth_y = ((teacher_field[:, :, 1:] - teacher_field[:, :, :-1]).abs()
                    * adjacent_y).sum() / adjacent_y.sum().clamp_min(1) / 1000
        smooth_x = ((teacher_field[:, :, :, 1:] - teacher_field[:, :, :, :-1]).abs()
                    * adjacent_x).sum() / adjacent_x.sum().clamp_min(1) / 1000
        loss = direct + true_map_error + .1 * reliability + .1 * teacher_magnitude \
            + .02 * (smooth_y + smooth_x)
        feedback_rank = direct.new_zeros(())
        feedback_rigid = direct.new_zeros(())
        feedback_mapped = direct.new_zeros(())

        if step > FEEDBACK_START:
            j = (step - 1) % SYNTHETIC
            chosen = prior[j:j + 1].detach().topk(3, -1).indices
            blind_state = prediction['state'][j:j + 1].gather(
                1, (chosen // 2)[..., None].expand(-1, -1, 12))
            delta = truth.new_zeros(1, 9)
            delta[:, :3] = torch.randn(1, 3, device='cuda') * .14
            delta[:, 3:6] = torch.randn(1, 3, device='cuda') * 900
            delta[:, 6:8] = torch.randn(1, 2, device='cuda') * .06
            delta[:, 8] = torch.randn(1, device='cuda') * .04
            guided = compose_full_frame_state(batch['state'][j:j + 1], delta)
            initial = torch.cat((blind_state, batch['state'][j:j + 1, None], guided[:, None]), 1)
            reflected = torch.cat((chosen % 2,
                                   batch['reflection'][j:j + 1, None].expand(-1, 2)), 1)
            selected = {key: value[j:j + 1] if torch.is_tensor(value) else value
                        for key, value in prediction.items()}
            selected['state'] = initial
            selected['log_mass'] = torch.cat((
                prediction['log_mass'][j:j + 1].gather(1, chosen // 2),
                prior.new_full((1, 2), -math.log(model.modes))), 1)
            selected['reflection_logit'] = torch.cat((
                prediction['reflection_logit'][j:j + 1].gather(1, chosen // 2),
                prior.new_zeros((1, 2))), 1)
            modes = torch.arange(5, device='cuda')[None]
            first = model.map(selected, batch['offsets'][j:j + 1], modes, reflected,
                              (48, 48), context['atlas'], batch['weights'][j:j + 1],
                              return_refinement_feature=True, feature_side=48,
                              source_shape=(SIDE, SIDE))
            refined_state, _, _ = model.refine(first['refinement_feature'], initial)
            refined = {**selected, 'state': refined_state}
            mapped = model.map(refined, batch['offsets'][j:j + 1], modes, reflected,
                               (64, 64), context['atlas'], batch['weights'][j:j + 1],
                               feature_side=64, source_shape=(SIDE, SIDE))
            target_j, index_j = target[j:j + 1], indices[j:j + 1]
            chart_j = chart[j:j + 1]
            before = (points(initial, reflected, chart_j[:, None])
                      - target_j[:, None]).norm(dim=-1).mean(-1)
            rigid = (points(refined_state, reflected, chart_j[:, None])
                     - target_j[:, None]).norm(dim=-1).mean(-1)
            mapped_error = (mapped_sites(mapped, index_j) - target_j[:, None]).norm(dim=-1).mean(-1)
            local_rms = mapped['local_displacement_um'].square().mean((2, 3, 4)).sqrt()
            quality = rigid + .25 * mapped_error + .25 * local_rms
            score = model.score_fitted_candidates(batch['inputs'][j:j + 1], refined,
                                                   mapped, context['atlas'], batch['weights'][j:j + 1])
            branch_prior = (selected['log_mass'] + torch.where(reflected.bool(),
                F.logsigmoid(selected['reflection_logit']),
                F.logsigmoid(-selected['reflection_logit'])))
            learned = score - branch_prior
            feedback_rank = F.kl_div(F.log_softmax(learned, -1),
                                     F.softmax(-quality.detach() / 750, -1), reduction='batchmean')
            best_blind = before[:, :3].detach().argmin(-1)
            feedback_rigid = (rigid[:, :3].gather(1, best_blind[:, None]).mean()
                              + rigid[:, 3:].mean()) / 2000
            feedback_mapped = mapped_error[:, 3:].mean() / 1000
            blind_expected = (F.softmax(score[:, :3], -1) * quality[:, :3]).sum(-1).mean() / 1000
            loss = loss + .3 * feedback_rank + .5 * feedback_rigid \
                + .25 * feedback_mapped + .25 * blind_expected

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'synthetic_presentations': step * SYNTHETIC,
               'real_presentations': step, 'direct': float(direct.detach()),
               'synthetic_oracle_um': float(oracle[:SYNTHETIC].detach().mean() * 1000),
               'synthetic_prior_selected_um': float(distance[:SYNTHETIC].detach().gather(
                   1, prior[:SYNTHETIC].detach().argmax(-1)[:, None]).mean()),
               'real_weak_prior_selected_um': float(distance[SYNTHETIC:].detach().gather(
                   1, prior[SYNTHETIC:].detach().argmax(-1)[:, None]).mean()),
               'exact_pose_map_um': float(true_map_error.detach() * 1000),
               'reliability': float(reliability.detach()),
               'feedback_rank': float(feedback_rank.detach()),
               'feedback_rigid_um': float(feedback_rigid.detach() * 1000),
               'feedback_guided_map_um': float(feedback_mapped.detach() * 1000),
               'loss': float(loss.detach()), 'gradient': float(gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step in SAVE_STEPS:
            save(step)

(RUN / 'completed.json').write_text(json.dumps({
    'batches': UPDATES, 'accepted_synthetic': UPDATES * SYNTHETIC,
    'distinct_real_train': len(set(real_schedule)),
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256((RUN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
                          for step in SAVE_STEPS},
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2))
print(json.dumps({'event': 'complete', 'batches': UPDATES,
                  'seconds': time.perf_counter() - started}), flush=True)
