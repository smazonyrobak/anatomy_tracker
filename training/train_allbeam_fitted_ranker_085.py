"""Fit the 14-branch atlas selector with frozen pose, map, and correspondence."""
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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
pilot = root / 'runs/dense_atlas_correspondence_083_pilot'
run = root / 'runs/allbeam_fitted_ranker_085_pilot'
seed, batches, side = 2026100485, 1000, 256
checkpoints = (0, 500, 1000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


context = load_streaming_synthetic_v7_64(device='cuda')
pilot_receipt = json.loads((pilot / 'completed.json').read_text())
pilot_config = json.loads((pilot / 'config.json').read_text())
assert pilot_receipt['batches'] == 6000 and not pilot_receipt['calibrated']
assert not pilot_receipt['public_benchmark_used']
assert sha(pilot / 'config.json') == pilot_receipt['config_sha256']
assert sha(pilot / 'draws.jsonl') == pilot_receipt['draws_sha256']
assert sha(pilot / 'training.jsonl') == pilot_receipt['training_sha256']
assert sha(pilot / 'match_step_06000.pt') == pilot_receipt['checkpoint_sha256']['6000']
assert sha(parent) == pilot_config['parent_sha256']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in pilot_config['source_sha256'].items())
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in pilot_config['synthetic_provenance']['source_sha256'].items())

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
frozen = torch.load(parent, map_location='cpu', weights_only=True)
assert frozen['step'] == 50000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
model.fitted_matcher.requires_grad_(True)
del frozen
frozen = torch.load(pilot / 'match_step_06000.pt', map_location='cpu', weights_only=True)
assert frozen['step'] == 6000 and not frozen['calibrated']
head = WholeSliceAtlasFeedback083().cuda().eval()
head.load_state_dict(frozen['head'], strict=True)
head.requires_grad_(False)
for layer in (head.spatial, head.summary, head.quality):
    layer.requires_grad_(True)
del frozen
trainable = [*head.spatial.parameters(), *head.summary.parameters(),
             *head.quality.parameters(), *model.fitted_matcher.parameters()]
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'parent': str(parent), 'parent_sha256': sha(parent),
    'pilot_083_completed_sha256': sha(pilot / 'completed.json'),
    'pilot_083_config_sha256': sha(pilot / 'config.json'),
    'pilot_083_checkpoint_6000_sha256': sha(pilot / 'match_step_06000.pt'),
    'candidate_beam': {'old': 8, 'anchor': 6, 'truth_inserted': False},
    'candidate_chunk': 2, 'accepted_synthetic_per_batch': 1,
    'trainable': ['head.spatial', 'head.summary', 'head.quality', 'model.fitted_matcher'],
    'frozen': ['model pose/encoder/map/atlas_encoder', 'head image/atlas embeddings',
               'head pose/mapper_feature', 'head temperature'],
    'score_target': '-(mapped96 physical CCF error in mm + 0.03 * mean((local_displacement_um/1000)^2))',
    'loss': 'smooth_l1(total fitted candidate score, score_target, beta=1), mean over all 14',
    'synthetic_error_points': '1024 fixed surviving-tissue pixels, identical across the 14 branches',
    'synthetic_provenance': context['provenance'],
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        'train_allbeam_fitted_ranker_085.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_full_frame_primitives.py')},
    'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'feedback': head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'ranker_step_{step:05d}.pt')


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
                    ('inputs', 'offsets', 'weights', 'centre', 'valid_mask')}
        with torch.no_grad():
            prediction = model.predict(accepted['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            state = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
            valid = accepted['valid_mask'].flatten().nonzero().flatten()
            indices = valid[torch.linspace(0, len(valid) - 1, 1024,
                                           device='cuda').round().long()]
            target = accepted['centre'].reshape(-1, 3)[indices]
            grid = (torch.stack((indices.remainder(side),
                indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        optimizer.zero_grad(set_to_none=True)
        losses, errors, scores = [], [], []
        for start in range(0, 14, 2):
            choice = beam[:, start:start + 2]
            chosen_state = state[:, start:start + 2]
            reflection = choice % 2
            local_index = torch.arange(2, device='cuda')[None]
            with torch.no_grad():
                first = head(prediction['feature'], chosen_state, reflection,
                             context['atlas'], accepted['offsets'], accepted['weights'])
                selected = {**prediction, 'state': first['state'],
                    'log_mass': prediction['log_mass'].gather(1, choice // 2),
                    'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
                mapped64 = model.map(selected, accepted['offsets'], local_index,
                    reflection, (64, 64), context['atlas'], accepted['weights'],
                    return_refinement_feature=True, feature_side=64,
                    source_shape=(side, side), spatial_evidence=first['spatial_evidence'])
                difficulty = fit_summary(mapped64)
            second = head(prediction['feature'], first['state'], reflection,
                          context['atlas'], accepted['offsets'], accepted['weights'],
                          fit_summary=difficulty)
            selected['state'] = second['state'].detach()
            with torch.no_grad():
                mapped96 = model.map(selected, accepted['offsets'], local_index,
                    reflection, (96, 96), context['atlas'], accepted['weights'],
                    feature_side=96, source_shape=(side, side),
                    spatial_evidence=second['spatial_evidence'])
                surface = mapped96['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
                query = grid[None].expand(2, -1, -1).reshape(2, 1, 1024, 2)
                fitted = F.grid_sample(surface, query, padding_mode='border',
                                       align_corners=False).reshape(2, 3, 1024).permute(0, 2, 1)
                physical_mm = (fitted - target[None]).norm(dim=-1).mean(-1) / 1000
                warp_mm2 = (mapped96['local_displacement_um'] / 1000).square().mean((2, 3, 4))[0]
                target_score = -(physical_mm + .03 * warp_mm2)
            score = (model.score_fitted_candidates(accepted['inputs'], selected, mapped96,
                     context['atlas'], accepted['weights']) + second['quality_logit'])[0]
            loss = F.smooth_l1_loss(score, target_score, beta=1., reduction='sum') / 14
            loss.backward()
            losses.append(loss.detach())
            errors.append(physical_mm.detach())
            scores.append(score.detach())
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            errors = torch.cat(errors)
            scores = torch.cat(scores)
            selected = int(scores.argmax())
            row = {'batch': step, 'synthetic_presentations': step,
                   'loss': float(torch.stack(losses).sum()),
                   'selected_mapped96_mm': float(errors[selected]),
                   'best14_mapped96_mm': float(errors.min()),
                   'selected_minus_best14_mm': float(errors[selected] - errors.min()),
                   'gradient_norm': float(gradient),
                   'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step % 500 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'synthetic_presentations': batches,
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'ranker_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
