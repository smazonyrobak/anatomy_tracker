"""Matched pose continuation with versus without fit feedback from predicted planes."""
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

parent = root / 'runs/one_shot_candidate_energy_004/joint_step_08000.pt'
run = root / 'runs/one_shot_native_fit_feedback_005'
seed, updates, synthetic, side = 2026100805, 8000, 2, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()

used_real = {tuple(row) for path in (
    root / 'runs/one_shot_joint_mixed_real_003/real_schedule.npy',
    root / 'runs/one_shot_fit_feedback_matched_001/new_real_schedule.npy',
    root / 'runs/one_shot_joint_physical_pose_002/real_schedule.npy',
    root / 'runs/one_shot_pose_capture_003/real_schedule.npy',
    root / 'runs/one_shot_candidate_energy_004/real_schedule.npy')
    for row in np.load(path)}
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < updates:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == updates:
                break
assert len(set(schedule)) == updates and not set(schedule) & used_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': seed, 'updates_per_arm': updates, 'synthetic_per_batch': synthetic,
          'real_per_batch': 1, 'side': side, 'modes': 16, 'arms': ('control', 'fit'),
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'trainable': 'pose head only; encoder, atlas matcher, local mapper and uncertainty frozen identically',
          'shared_loss': 'best and ranked 32-branch tissue/full-frame geometry plus 4mm antipodal-normal anchor',
          'fit_only_loss': 'frozen learned atlas-match energy through actual image-prior and geometry-best predicted branches, synthetic and weak-real; coefficient fixed from first TRAIN batch at 20% of direct pose-head gradient',
          'fit_feedback_into_pose': True, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_native_fit_feedback_005.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


results = {}
for arm in ('control', 'fit'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
    model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.requires_grad_(False)
    model.pose.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.pose.parameters(), lr=3e-5, weight_decay=1e-4)
    subjects_rng = torch.Generator().manual_seed(seed)
    draw_seed = seed * 10000000
    fit_weight = None

    def save(step):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'step': step, 'arm': arm, 'fit_weight': fit_weight,
                    'config': config, 'calibrated': False},
                   directory / f'joint_step_{step:05d}.pt')

    save(0)
    started = time.perf_counter()
    with (directory / 'training.jsonl').open('w') as log, (directory / 'draws.jsonl').open('w') as draws:
        for step in range(1, updates + 1):
            accepted, pending = {}, list(range(synthetic))
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
            batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                     for key in accepted[0]}
            donor, section = schedule[step - 1]
            observation = sample_reserved_real_train(real, donor, [section], device='cuda')
            observation['inputs'] = F.interpolate(observation['inputs'], (side, side),
                                                  mode='bilinear', align_corners=False)
            draws.write(json.dumps({**observation['identities'][0], 'step': step, 'slot': synthetic,
                                    'used': True, 'label_role': real['label_role']}) + '\n')
            image = torch.cat((batch['inputs'], observation['inputs']))
            truth = torch.cat((batch['state'], observation['state']))
            truth_reflection = torch.cat((batch['reflection'], observation['reflection']))
            prediction = model.predict(image)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            branch_flags = flags[None, None].expand(len(image), model.modes, -1)
            reference = points(truth, truth_reflection, corners)
            five = (points(states, branch_flags, corners) - reference[:, None, None]).norm(dim=-1).mean(-1)
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            target = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, indices[..., None].expand(-1, -1, 3))
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            dense = (points(states[:synthetic], branch_flags[:synthetic], chart[:, None, None])
                     - target[:, None, None]).norm(dim=-1).mean(-1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            reference_normals = full_frame_state_to_components(truth)[1][:, :, 2]
            normal = 1 - (normals * reference_normals[:, None]).sum(-1).abs().clamp_max(1)
            distance = torch.cat((.75 * dense + .25 * five[:synthetic] + 4000 * normal[:synthetic, :, None],
                                  five[synthetic:]), 0)
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            flat_distance, flat_prior = distance.flatten(1), prior.flatten(1)
            direct = flat_distance.min(-1).values.mean() / 1000
            marginal = -torch.logsumexp(flat_prior - flat_distance / 1500, -1).mean()
            rank = F.kl_div(flat_prior, (-flat_distance.detach() / 1000).softmax(-1),
                            reduction='batchmean')
            direct_loss = direct + .75 * marginal + .75 * rank
            fit_term = None
            if arm == 'fit':
                real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
                real_weights = torch.ones_like(real_offsets)
                real_weights[:, [0, -1]] = .5
                real_weights /= real_weights.sum(-1, keepdim=True)
                offsets = torch.cat((batch['offsets'], real_offsets))
                weights = torch.cat((batch['weights'], real_weights))
                choice = torch.stack((flat_prior.detach().argmax(-1),
                                      flat_distance.detach().argmin(-1)), -1)
                mapped = model.map(prediction, offsets, choice // 2, choice % 2,
                                   (side, side), context['atlas'], weights)
                energy = mapped['fit_energy'].mean(-1)
                fit_term = (energy[:synthetic].sum() + .5 * energy[synthetic:].sum()) / (synthetic + .5)
                if fit_weight is None:
                    direct_gradient = torch.autograd.grad(direct_loss, model.pose[-1].weight,
                                                          retain_graph=True)[0].norm()
                    fit_gradient = torch.autograd.grad(fit_term, model.pose[-1].weight,
                                                       retain_graph=True)[0].norm()
                    assert float(fit_gradient) > 0
                    fit_weight = float((.2 * direct_gradient / fit_gradient).clamp(.01, .5))
                    (directory / 'fit_scale.json').write_text(json.dumps({
                        'fit_weight': fit_weight, 'first_train_batch_direct_gradient': float(direct_gradient),
                        'first_train_batch_fit_gradient': float(fit_gradient)}, indent=2))
            loss = direct_loss if fit_term is None else direct_loss + fit_weight * fit_term
            optimizer.param_groups[0]['lr'] = 3e-5 * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / updates)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.pose.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            selected = flat_prior.detach().argmax(-1)
            selected_tissue = dense.detach().flatten(1).gather(1, selected[:synthetic, None])
            selected_real = five.detach().flatten(1).gather(1, selected[synthetic:, None])
            row = {'arm': arm, 'step': step, 'presentations': step * len(image),
                   'synthetic_selected_tissue_um': float(selected_tissue.mean()),
                   'synthetic_oracle_tissue_um': float(dense.detach().flatten(1).min(-1).values.mean()),
                   'real_weak_selected_five_um': float(selected_real.mean()),
                   'direct_loss': float(direct_loss.detach()),
                   'fit_term': None if fit_term is None else float(fit_term.detach()),
                   'fit_weight': fit_weight, 'pose_gradient_norm': float(gradient),
                   'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
            if step % 4000 == 0:
                save(step)
    results[arm] = {'training_rows': updates,
                    'draws_sha256': hashlib.sha256((directory / 'draws.jsonl').read_bytes()).hexdigest(),
                    'fit_weight': fit_weight, 'seconds': time.perf_counter() - started}
assert results['control']['draws_sha256'] == results['fit']['draws_sha256']
(run / 'completed.json').write_text(json.dumps({'updates_per_arm': updates,
    'accepted_synthetic_per_arm': updates * synthetic, 'distinct_real_train_per_arm': updates,
    'matched_draws_sha256': results['control']['draws_sha256'], 'arms': results,
    'fit_feedback_into_pose': True, 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates_per_arm': updates,
                  'matched_draws_sha256': results['control']['draws_sha256']}), flush=True)
