"""Matched one-step physical pose correction: image-only versus atlas/warp evidence."""
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
run = root / 'runs/one_shot_vector_feedback_006'
seed, updates, synthetic, side, candidates = 2026100806, 8000, 2, 256, 4
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
    root / 'runs/one_shot_candidate_energy_004/real_schedule.npy',
    root / 'runs/one_shot_native_fit_feedback_005/real_schedule.npy')
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
          'real_per_batch': 1, 'side': side, 'candidate_branches': candidates,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'trainable': 'new shared one-step pose-correction and branch-ranking head only; all parent weights frozen',
          'comparison': 'same initial weights, data, branches and map compute; direct arm masks atlas/render/local-warp evidence, atlas arm sees it',
          'objective': 'bounded full-frame correction from actual top-four predicted branches; visible-tissue physical loss, antipodal normal, weak-real affine retention and post-correction branch ranking',
          'uncertainty_calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_vector_feedback_006.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


results = {}
for arm in ('direct', 'atlas'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
    model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                   vector_refinement=True).cuda().eval()
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    assert missing and all(key.startswith('pose_refiner.') for key in missing) and not unexpected
    del checkpoint
    model.requires_grad_(False)
    model.pose_refiner.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.pose_refiner.parameters(), lr=1e-4, weight_decay=1e-4)
    subjects_rng = torch.Generator().manual_seed(seed)
    draw_seed = seed * 10000000

    def save(step):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'step': step, 'arm': arm, 'config': config, 'calibrated': False},
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
            draws.write(json.dumps({**observation['identities'][0], 'step': step,
                                    'slot': synthetic, 'used': True,
                                    'label_role': real['label_role']}) + '\n')
            image = torch.cat((batch['inputs'], observation['inputs']))
            real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
            real_weights = torch.ones_like(real_offsets)
            real_weights[:, [0, -1]] = .5
            real_weights /= real_weights.sum(-1, keepdim=True)
            offsets = torch.cat((batch['offsets'], real_offsets))
            weights = torch.cat((batch['weights'], real_weights))
            with torch.no_grad():
                prediction = model.predict(image)
                prior = prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)
                chosen = prior.flatten(1).topk(candidates, -1).indices
                mode_index, reflection = chosen // 2, chosen % 2
                state = prediction['state'][torch.arange(len(image), device='cuda')[:, None], mode_index]
                mapped = model.map(prediction, offsets, mode_index, reflection,
                                   (side, side), context['atlas'], weights,
                                   return_refinement_feature=True)
                feature = mapped['refinement_feature']
                if arm == 'direct':
                    feature = feature.clone()
                    feature[:, :, 64:217] = 0
                    feature[:, :, 219:] = 0
                indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
                target = batch['centre'].reshape(synthetic, -1, 3).gather(
                    1, indices[..., None].expand(-1, -1, 3))
                chart = torch.stack((indices.remainder(side),
                                     indices.div(side, rounding_mode='floor')), -1).float() / side
                real_reference = points(observation['state'], observation['reflection'], corners)
                truth_normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
            refined, score, update = model.refine(feature, state)
            tissue = (points(refined[:synthetic], reflection[:synthetic], chart[:, None])
                      - target[:, None]).norm(dim=-1).mean(-1)
            real_five = (points(refined[synthetic:], reflection[synthetic:], corners)
                         - real_reference[:, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(refined[:synthetic])[1][..., :, 2]
            normal_penalty = 4000 * (1 - (normal * truth_normal[:, None]).sum(-1).abs().clamp_max(1))
            distance = torch.cat((tissue, real_five))
            pose_loss = ((tissue + normal_penalty).mean() + .25 * real_five.mean()) / 1250
            rank_loss = F.kl_div(F.log_softmax(score, -1),
                                 F.softmax(-distance.detach() / 1000, -1), reduction='batchmean')
            loss = pose_loss + rank_loss + .002 * (update[..., :3] / .9).square().mean()
            optimizer.param_groups[0]['lr'] = 1e-4 * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / updates)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.pose_refiner.parameters(), 5.,
                                                       error_if_nonfinite=True)
            optimizer.step()
            selected = score.detach().argmax(-1)
            row = {'arm': arm, 'step': step, 'presentations': step * len(image),
                   'synthetic_selected_tissue_um': float(tissue.detach().gather(
                       1, selected[:synthetic, None]).mean()),
                   'synthetic_oracle_tissue_um': float(tissue.detach().min(-1).values.mean()),
                   'real_weak_selected_five_um': float(real_five.detach().gather(
                       1, selected[synthetic:, None]).mean()),
                   'pose_loss': float(pose_loss.detach()), 'rank_loss': float(rank_loss.detach()),
                   'refiner_gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
            if step % 2000 == 0:
                save(step)
    results[arm] = {'training_rows': updates,
                    'draws_sha256': hashlib.sha256((directory / 'draws.jsonl').read_bytes()).hexdigest(),
                    'seconds': time.perf_counter() - started}
assert results['direct']['draws_sha256'] == results['atlas']['draws_sha256']
(run / 'completed.json').write_text(json.dumps({'updates_per_arm': updates,
    'accepted_synthetic_per_arm': updates * synthetic, 'distinct_real_train_per_arm': updates,
    'matched_draws_sha256': results['direct']['draws_sha256'], 'arms': results,
    'uncertainty_calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates_per_arm': updates,
                  'matched_draws_sha256': results['direct']['draws_sha256']}), flush=True)
