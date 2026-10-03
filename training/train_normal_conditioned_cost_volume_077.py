"""Train a frozen-059, two-pass 3D cost-volume pose updater on actual proposals."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.normal_conditioned_cost_volume_077 import NormalConditionedCostVolume077

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/normal_conditioned_cost_volume_077_pilot'
seed, batches, synthetic, side, chunk = 2026100477, 2000, 2, 256, 2
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
used_real = {tuple(row) for row in np.load(root / 'runs/one_shot_anchor_quality_059/real_schedule.npy')}
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < batches:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == batches:
                break
assert len(set(schedule)) == batches and not set(schedule) & used_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
    'real_weak_per_batch': 1, 'side': side, 'candidate_quota': {'old': 8, 'anchor': 6},
    'candidate_chunk': chunk, 'iterations_shared': 2, 'normal_context_um': [-6000, 6000],
    'atlas_context_grid': [13, 24, 24], 'correlation_grid': [13, 13, 13],
    'bounded_update_per_iteration': {'translation_um': 3000, 'normal_offset_um': 3000,
        'normal_tangent_tilt_deg_each': 15, 'roll_deg': 15,
        'isotropic_scale': [1 / 1.2, 1.2]},
    'parent': str(parent), 'parent_sha256': sha(parent),
    'protocol_sha256': sha(Path(__file__).parent.parent /
        'docs/publication/NORMAL_CONDITIONED_COST_VOLUME_077_PROTOCOL_20261004.md'),
    'real_schedule_sha256': sha(run / 'real_schedule.npy'),
    'excluded_059_real_schedule_sha256': sha(root / 'runs/one_shot_anchor_quality_059/real_schedule.npy'),
    'real_label_role': real['label_role'], 'real_bindings': real['bindings'],
    'synthetic_provenance': context['provenance'],
    'training': 'physical pose correction of all 14 actual 059 proposals; no truth-inserted plane or scalar reranker',
    'real_weak_loss_weight': .15, 'mapper_frozen': True, 'parent_frozen': True,
    'calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
        ('train_normal_conditioned_cost_volume_077.py', 'normal_conditioned_cost_volume_077.py',
         'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
         'arbitrary_plane_streaming_synthetic_v7_64.py',
         'arbitrary_plane_reserved_real_stream_v8.py',
         'arbitrary_plane_allen_atlas_binding_v6.py',
         'arbitrary_plane_full_frame_primitives.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
frozen = torch.load(parent, map_location='cpu', weights_only=True)
assert frozen['step'] == 50000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen
head = NormalConditionedCostVolume077().cuda().train()
optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def save(step):
    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
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
                                    generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'offsets', 'weights', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
            'slot': synthetic, 'used': True, 'label_role': real['label_role']}) + '\n')
        real_image = F.interpolate(observation['inputs'], (side, side),
                                   mode='bilinear', align_corners=False)
        image = torch.cat((batch['inputs'], real_image))
        real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
        real_weights = torch.ones_like(real_offsets)
        real_weights[:, [0, -1]] = .5
        real_weights /= real_weights.sum(-1, keepdim=True)
        offsets = torch.cat((batch['offsets'], real_offsets))
        weights = torch.cat((batch['weights'], real_weights))
        with torch.no_grad():
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            branch_weight = .5 / 14 + .5 * prior.gather(1, choice).softmax(-1)
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 96, replacement=True)
            target = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, indices[..., None].expand(-1, -1, 3))
            chart = torch.stack((indices.remainder(side),
                indices.div(side, rounding_mode='floor')), -1).float() / side
            real_reference = points(observation['state'], observation['reflection'], corners)
        optimizer.zero_grad(set_to_none=True)
        synthetic_errors = []
        real_errors = []
        total_loss = 0.
        for first in range(0, 14, chunk):
            last = first + chunk
            first_state, corrected = head(prediction, image, states[:, first:last],
                reflection[:, first:last], context['atlas'], offsets, weights)
            first_synthetic = (points(first_state[:synthetic], reflection[:synthetic, first:last],
                chart[:, None]) - target[:, None]).norm(dim=-1).mean(-1)
            after_synthetic = (points(corrected[:synthetic], reflection[:synthetic, first:last],
                chart[:, None]) - target[:, None]).norm(dim=-1).mean(-1)
            first_real = (points(first_state[synthetic:], reflection[synthetic:, first:last],
                corners) - real_reference[:, None]).norm(dim=-1).mean(-1)
            after_real = (points(corrected[synthetic:], reflection[synthetic:, first:last],
                corners) - real_reference[:, None]).norm(dim=-1).mean(-1)
            synthetic_loss = .25 * F.smooth_l1_loss(first_synthetic / 1000,
                torch.zeros_like(first_synthetic), beta=1., reduction='none') \
                + F.smooth_l1_loss(after_synthetic / 1000,
                    torch.zeros_like(after_synthetic), beta=1., reduction='none')
            real_loss = .25 * F.smooth_l1_loss(first_real / 1000,
                torch.zeros_like(first_real), beta=1., reduction='none') \
                + F.smooth_l1_loss(after_real / 1000,
                    torch.zeros_like(after_real), beta=1., reduction='none')
            loss = (synthetic_loss * branch_weight[:synthetic, first:last]).sum(-1).mean() \
                   + .15 * (real_loss * branch_weight[synthetic:, first:last]).sum(-1).mean()
            loss.backward()
            total_loss += float(loss.detach())
            synthetic_errors.append(after_synthetic.detach())
            real_errors.append(after_real.detach())
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            synthetic_error = torch.cat(synthetic_errors, -1)
            real_error = torch.cat(real_errors, -1)
            selected = prior.gather(1, choice).argmax(-1)
            row = {'batch': step, 'accepted_synthetic': step * synthetic,
                'distinct_real_weak': step, 'loss': total_loss,
                'synthetic_parent_selected_corrected_um': float(synthetic_error.gather(
                    1, selected[:synthetic, None]).mean()),
                'synthetic_best14_corrected_um': float(synthetic_error.min(-1).values.mean()),
                'real_parent_selected_corrected_um': float(real_error.gather(
                    1, selected[synthetic:, None]).mean()),
                'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in (1000, batches):
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({'batches': batches,
    'accepted_synthetic': batches * synthetic, 'distinct_real_weak': batches,
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in (0, 1000, batches)},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
