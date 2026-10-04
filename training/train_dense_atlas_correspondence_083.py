"""Supervise atlas cost-volume matches from fresh arbitrary-plane synthetic sections."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083, synthetic_match_targets

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/dense_atlas_correspondence_083_pilot'
seed, batches, side = 2026100483, 6000, 256
checkpoints = (0, 1000, 3000, 6000)
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
head = WholeSliceAtlasFeedback083().cuda().train()
head.requires_grad_(False)
head.image.requires_grad_(True)
head.atlas.requires_grad_(True)
head.log_temperature.requires_grad_(True)
optimizer = torch.optim.AdamW([
    *head.image.parameters(), *head.atlas.parameters(), head.log_temperature
], lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
run.mkdir(parents=True, exist_ok=False)
config = {
    'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'parent': str(parent), 'parent_sha256': sha(parent),
    'two_existing_beam_candidates': 'physical-best of 14 plus uniform random distinct slot',
    'candidate_beam': {'old': 8, 'anchor': 6},
    'trained': 'image and atlas embeddings plus correlation temperature only',
    'match_labels': 'nearest 9-depth/local-offset class at visible 32 and 16 query centres',
    'support_threshold': .5,
    'synthetic_provenance': context['provenance'],
    'development_gate': {'coarse_ce_relative_improvement_at_least': .2,
                         'coarse_top1_mm_relative_improvement_at_least': .2,
                         'fine_ce_relative_worsening_at_most': .05},
    'calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        'train_dense_atlas_correspondence_083.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_full_frame_primitives.py')},
}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'match_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            used = bool(sampled['eligible'][0])
            draws.write(json.dumps({**sampled['provenance'][0], 'step': step,
                                     'used': used}) + '\n')
            if used:
                accepted = {key: sampled[key] for key in
                    ('inputs', 'state', 'reflection', 'offsets', 'weights',
                     'centre', 'valid_mask')}
        with torch.no_grad():
            prediction = model.predict(accepted['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                                      96, replacement=True)
            target = accepted['centre'].reshape(1, -1, 3).gather(
                1, pixel[..., None].expand(-1, -1, 3))
            chart = torch.stack((pixel.remainder(side),
                                 pixel.div(side, rounding_mode='floor')), -1).float() / side
            state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflection = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
            distance = (points(state, reflection, chart) - target[:, None, None]).norm(dim=-1).mean(-1)
            true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            distance += 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))[..., None]
            best = distance.flatten(1).gather(1, beam).argmin(-1)
            other = torch.randint(13, (1,), device='cuda')
            other += (other >= best).long()
            choice = beam.gather(1, torch.stack((best, other), -1))
            chosen = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            labels = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
                                              chosen, choice % 2)
        output = head(prediction['feature'], chosen, choice % 2,
                      context['atlas'], accepted['offsets'], accepted['weights'],
                      match_only=True)
        losses, counts, possible = [], [], []
        for scale in ('fine', 'coarse'):
            logits = output[f'{scale}_match_logits']
            index = labels[f'{scale}_index']
            support = output[f'{scale}_match_support'].gather(2, index[:, :, None]).squeeze(2)
            mask = labels[f'{scale}_mask'] & (support >= .5)
            ce = F.cross_entropy(logits.flatten(0, 1), index.flatten(0, 1),
                                 reduction='none').reshape_as(mask)
            count = mask.flatten(2).sum(-1)
            loss = (ce * mask).flatten(2).sum(-1) / count.clamp_min(1)
            losses.append(loss.sum() / (count > 0).sum().clamp_min(1))
            counts.append(int(count.sum()))
            possible.append(int(labels[f'{scale}_mask'].sum()))
        loss = (losses[0] + losses[1]) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 5.)
        progress = step / batches
        optimizer.param_groups[0]['lr'] = 1e-4 * (.2 + .8 * .5 * (1 + math.cos(math.pi * progress)))
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'batch': step, 'fine_ce': float(losses[0].detach()),
                   'coarse_ce': float(losses[1].detach()),
                   'fine_supervised': counts[0], 'coarse_supervised': counts[1],
                   'fine_geometric': possible[0], 'coarse_geometric': possible[1],
                   'temperature': float(head.log_temperature.detach().exp()),
                   'gradient_norm': float(gradient),
                   'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in checkpoints:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'synthetic_presentations': batches,
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'match_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
