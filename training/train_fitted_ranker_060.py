"""Fit the old/new eight-candidate scorer; all pose and tissue-map weights stay frozen."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/fitted_ranker_060'
seed, batches, synthetic, side = 2026100360, 6000, 2, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < batches:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == batches:
                break
assert len(set(schedule)) == batches

run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, np.int32))
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
          'real_per_batch': 1, 'legacy_modes': 16, 'normal_anchors': 64,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'real_labels': real['label_role'], 'real_bindings': real['bindings'],
          'synthetic_provenance': context['provenance'],
          'rank_target_temperature_um': {'synthetic': 500, 'real_weak_allen': 1000},
          'fitted_matcher_only_trainable': True, 'calibrated': False,
          'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_fitted_ranker_060.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_reserved_real_stream_v8.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_full_frame_primitives.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
                              fit_quality=True, vector_refinement=True,
                              candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'], strict=True)
model.requires_grad_(False)
model.fitted_matcher.requires_grad_(True)
optimizer = torch.optim.AdamW(model.fitted_matcher.parameters(), lr=3e-5, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                eligible = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': eligible}) + '\n')
                if eligible:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                       'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        observation['inputs'] = F.interpolate(observation['inputs'], (side, side),
                                              mode='bilinear', align_corners=False)
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
                                'slot': synthetic, 'used': True,
                                'label_role': real['label_role']}) + '\n')
        image = torch.cat((batch['inputs'], observation['inputs']))
        offsets_real = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
        weights_real = torch.ones_like(offsets_real)
        weights_real[:, [0, -1]] = .5
        weights_real /= weights_real.sum(-1, keepdim=True)
        offsets = torch.cat((batch['offsets'], offsets_real))
        weights = torch.cat((batch['weights'], weights_real))
        with torch.no_grad():
            prediction = model.predict(image)
            reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
            choice = torch.cat((prior[:, :32].topk(2, -1).indices,
                                prior[:, 32:].topk(6, -1).indices + 32), -1)
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            selected = {**prediction, 'state': state,
                        'log_mass': prediction['log_mass'].gather(1, choice // 2),
                        'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
            index = torch.arange(8, device='cuda')[None].expand(len(image), -1)
            first = model.map(selected, offsets, index, choice % 2, (64, 64),
                              context['atlas'], weights, return_refinement_feature=True,
                              feature_side=64, source_shape=(side, side))
            refined_state, delta, _ = model.refine(first['refinement_feature'], state)
            refined = {**selected, 'state': refined_state}
            mapped = model.map(refined, offsets, index, choice % 2, (96, 96),
                               context['atlas'], weights, feature_side=96,
                               source_shape=(side, side))
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(),
                                        96, replacement=True)
            target = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, indices[..., None].expand(-1, -1, 3))
            grid = (torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
            grid = grid[:, None].expand(-1, 8, -1, -1).reshape(synthetic * 8, 1, 96, 2)
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:synthetic].flatten(0, 1).permute(0, 3, 1, 2)
            fitted = F.grid_sample(surface, grid, padding_mode='border',
                                   align_corners=False).reshape(synthetic, 8, 3, 96).permute(0, 1, 3, 2)
            synthetic_error = (fitted - target[:, None]).norm(dim=-1).mean(-1)
            real_reference = points(observation['state'], observation['reflection'], corners)
            real_error = (points(refined_state[synthetic:], choice[synthetic:] % 2, corners)
                          - real_reference[:, None]).norm(dim=-1).mean(-1)
        score = model.score_fitted_candidates(image, refined, mapped,
                                              context['atlas'], weights) + delta
        synthetic_target = F.softmax(-synthetic_error / 500, -1)
        real_target = F.softmax(-real_error / 1000, -1)
        synthetic_loss = -(synthetic_target * F.log_softmax(score[:synthetic], -1)).sum(-1).mean()
        real_loss = -(real_target * F.log_softmax(score[synthetic:], -1)).sum(-1).mean()
        loss = synthetic_loss + real_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(model.fitted_matcher.parameters(),
                                                  10., error_if_nonfinite=True)
        assert step != 1 or clipped > 0
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / batches))
        for group in optimizer.param_groups:
            group['lr'] = 3e-5 * decay
        optimizer.step()
        row = {'batch': step, 'synthetic_presentations': step * synthetic,
               'real_train_presentations': step,
               'synthetic_selected_train_um': float(synthetic_error.gather(
                   1, score[:synthetic].detach().argmax(-1, keepdim=True)).mean()),
               'synthetic_oracle_train_um': float(synthetic_error.min(-1).values.mean()),
               'real_selected_train_um': float(real_error.gather(
                   1, score[synthetic:].detach().argmax(-1, keepdim=True)).mean()),
               'real_oracle_train_um': float(real_error.min(-1).values.mean()),
               'synthetic_loss': float(synthetic_loss.detach()),
               'real_loss': float(real_loss.detach()), 'gradient_norm': float(clipped),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 2000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches * synthetic,
    'unique_real_train_within_run': batches,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
