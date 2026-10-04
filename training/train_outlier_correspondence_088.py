"""Train an explicit no-match class on fresh blind-beam synthetic sections."""
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
from training.whole_slice_atlas_feedback_083 import synthetic_match_targets
from training.whole_slice_atlas_feedback_088 import WholeSliceAtlasFeedback088

parent_run = root / 'runs/allbeam_fitted_ranker_085_pilot'
parent = parent_run / 'ranker_step_01000.pt'
run = root / 'runs/outlier_correspondence_088_train'
seed, batches, side = 2026100488, 20000, 256
checkpoints = (0, 2000, 10000, 20000)
background_null_weight = .1
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


context = load_streaming_synthetic_v7_64(device='cuda')
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 1000 and not parent_receipt['calibrated']
assert not parent_receipt['public_benchmark_used']
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert sha(parent) == parent_receipt['checkpoint_sha256']['1000']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in parent_config['source_sha256'].items())
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in parent_config['synthetic_provenance']['source_sha256'].items())

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
frozen = torch.load(parent, map_location='cpu', weights_only=True)
assert frozen['step'] == 1000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
head = WholeSliceAtlasFeedback088().cuda().train()
missing, unexpected = head.load_state_dict(frozen['feedback'], strict=False)
assert missing == ['null_logit'] and not unexpected
del frozen
head.requires_grad_(False)
head.image.requires_grad_(True)
head.atlas.requires_grad_(True)
head.log_temperature.requires_grad_(True)
head.null_logit.requires_grad_(True)
trainable = [*head.image.parameters(), *head.atlas.parameters(),
             head.log_temperature, head.null_logit]
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'parent': str(parent), 'parent_sha256': sha(parent),
    'parent_085_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_085_config_sha256': sha(parent_run / 'config.json'),
    'parent_085_draws_sha256': sha(parent_run / 'draws.jsonl'),
    'parent_085_training_sha256': sha(parent_run / 'training.jsonl'),
    'candidate_beam': {'old': 8, 'anchor': 6, 'truth_inserted': False},
    'two_existing_beam_candidates': 'physical-best of 14 plus uniform random distinct slot',
    'physical_best_criterion': '96 random surviving pixels: mean rigid CCF distance in um plus 4000*(1-abs(predicted/true normal dot product))',
    'accepted_synthetic_per_batch': 1,
    'trainable': ['head.image', 'head.atlas', 'head.log_temperature', 'head.null_logit'],
    'frozen': ['085 model', 'head spatial/summary/pose/mapper/quality'],
    'classes': {'fine': {'real': 225, 'null': 225},
                'coarse': {'real': 729, 'null': 729}},
    'target': 'visible valid pixels: real class if synthetic target inside window and target-cell support >= 0.5; null otherwise',
    'loss': 'per scale, equal positive and null weighted-mean CE; then equal fine/coarse mean',
    'background_null_weight': background_null_weight,
    'learning_rate': '1e-4 cosine decay to 2e-5 over 20000 accepted draws',
    'synthetic_provenance': context['provenance'],
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        'train_outlier_correspondence_088.py', 'whole_slice_atlas_feedback_088.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'head': head.state_dict(),
        'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.get_state(),
        'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'config': config,
        'calibrated': False}, run / f'match_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
totals = {scale: {'positive': 0, 'null_visible': 0, 'null_background': 0}
          for scale in ('fine', 'coarse')}
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            used = bool(sampled['eligible'][0])
            draws.write(json.dumps({**sampled['provenance'][0], 'step': step,
                                    'draw_seed': draw_seed, 'used': used}) + '\n')
            draw_seed += 1
            if used:
                accepted = {key: sampled[key] for key in
                    ('inputs', 'state', 'offsets', 'weights', 'centre', 'valid_mask')}
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
            centre, frame, basis = full_frame_state_to_components(state)
            xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
            xy[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / side - xy[..., 0], xy[..., 0])
            rigid = centre[..., None, :] + torch.einsum(
                '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)
            distance = (rigid - target[:, None, None]).norm(dim=-1).mean(-1)
            true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            distance += 4000 * (1 - (normal * true_normal[:, None]).sum(-1)
                                  .abs().clamp_max(1))[..., None]
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
        losses, row = [], {'batch': step, 'synthetic_presentations': step,
                           'beam': beam[0].tolist(), 'choice': choice[0].tolist(),
                           'positive_beam_slot': int(best[0]),
                           'random_beam_slot': int(other[0])}
        for scale, width in (('fine', 32), ('coarse', 16)):
            logits = output[f'{scale}_match_logits']
            index = labels[f'{scale}_index']
            support = output[f'{scale}_match_support'].gather(2, index[:, :, None]).squeeze(2)
            valid = F.interpolate(accepted['valid_mask'][:, None].float(),
                                  (width, width), mode='bilinear',
                                  align_corners=False)[:, 0] == 1
            valid = valid[:, None].expand_as(index)
            positive = labels[f'{scale}_mask'] & (support >= .5)
            null_visible = valid & ~positive
            null_background = ~valid
            target_index = torch.where(positive, index, logits.shape[2] - 1)
            ce = F.cross_entropy(logits.flatten(0, 1), target_index.flatten(0, 1),
                                 reduction='none').reshape_as(index)
            null_weight = null_visible.float() + background_null_weight * null_background.float()
            positive_loss = (ce * positive).sum() / positive.sum().clamp_min(1)
            null_loss = (ce * null_weight).sum() / null_weight.sum().clamp_min(1)
            losses.append((positive_loss + null_loss) / 2)
            row[f'{scale}_positive_ce'] = float(positive_loss.detach())
            row[f'{scale}_null_ce'] = float(null_loss.detach())
            row[f'{scale}_positive'] = int(positive.sum())
            row[f'{scale}_null_visible'] = int(null_visible.sum())
            row[f'{scale}_null_background'] = int(null_background.sum())
            row[f'{scale}_geometric'] = int(labels[f'{scale}_mask'].sum())
            for name in ('positive', 'null_visible', 'null_background'):
                totals[scale][name] += row[f'{scale}_{name}']
        loss = (losses[0] + losses[1]) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        progress = step / batches
        optimizer.param_groups[0]['lr'] = 1e-4 * (.2 + .8 * .5 * (1 + math.cos(math.pi * progress)))
        optimizer.step()
        row.update({'loss': float(loss.detach()),
                    'temperature': float(head.log_temperature.detach().exp()),
                    'null_logit': float(head.null_logit.detach()),
                    'gradient_norm': float(gradient),
                    'seconds': time.perf_counter() - started})
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'synthetic_presentations': batches, 'counts': totals,
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'match_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
