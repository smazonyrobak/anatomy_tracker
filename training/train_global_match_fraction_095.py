"""TRAIN-only whole-field readout of frozen 094 correspondence evidence."""

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
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.global_match_fraction_095 import GlobalMatchFraction095
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_MATCH_FRACTION_095_PROTOCOL_20261004.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent = parent_run / 'joint_step_20000.pt'
run = root / 'runs/global_match_fraction_095_pilot'
seed, batches, side, pixels = 2026100495, 12000, 256, 1024
checkpoints = (0, 4000, 12000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '701f768a156f95618ca5543e8865e0e26f8628c93c258d50e2bbc9da2ffe0a6c'
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent) == parent_receipt['checkpoint_sha256']['20000']
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 20000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
fitter = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval()
fitter.load_state_dict(checkpoint['spatial_fit'], strict=True)
del checkpoint
model.requires_grad_(False)
fitter.requires_grad_(False)
head = GlobalMatchFraction095().cuda().train()
optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
attempts = 0
flags = torch.tensor([0, 1], device='cuda')

config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 4,
    'candidate_rule': 'physical-best, support-nearest >=0.5mm worse, highest-prior >=1mm worse, remaining random; documented nonduplicate fallbacks',
    'score': 'direct log prior + log(sigmoid(predicted true fraction))',
    'loss': 'unweighted soft-target fraction BCE + 0.1 pairwise physical-error rank for >=0.5mm gaps',
    'fraction_target': 'visible=bilinear(valid_mask,32)==1; positive=visible & any supported fine bin & matched CCF within 1500um; positive/visible',
    'source_shape': (side, side), 'parent_094': str(parent),
    'parent_094_sha256': sha(parent),
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_global_match_fraction_095.py', 'global_match_fraction_095.py',
        'spatial_joint_fit_094.py', 'arbitrary_plane_one_shot_model.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_geometry.py', 'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'synthetic_provenance': context['provenance'], 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'fraction_head': head.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'fraction_step_{step:05d}.pt')


save(0)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            record = sampled['provenance'][0]
            used = bool(sampled['eligible'][0])
            attempts += 1
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                                    'used': used}) + '\n')
            draw_seed += 1
            if used:
                accepted = {key: sampled[key] for key in (
                    'inputs', 'state', 'reflection', 'offsets', 'weights',
                    'centre', 'valid_mask')}

        with torch.no_grad():
            prediction = model.predict(accepted['inputs'])
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags[None, None].expand(1, model.modes, 2)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                                      pixels, replacement=True)
            target = accepted['centre'].reshape(1, -1, 3).gather(
                1, pixel[..., None].expand(-1, -1, 3))
            chart = torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                                -1).float() / side
            rigid_mm = (rigid_points_090(states, reflections, chart) - target[:, None, None]
                        ).norm(dim=-1).mean(-1).flatten(1) / 1000
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            initial = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
            atlas_pair = render_atlas_planes_090(context['atlas'], initial, beam % 2,
                accepted['offsets'], accepted['weights'], candidate_chunk=2)
            support = atlas_pair[:, :, 1].mean((-2, -1))[0]
            beam_error = rigid_mm.gather(1, beam)[0]
            beam_prior = prior.gather(1, beam)[0]
            best = int(beam_error.argmin())
            gap = beam_error - beam_error[best]
            remaining = [index for index in range(14) if index != best]
            close = [index for index in remaining if gap[index] >= .5]
            support_fallback = not close
            if not close:
                close = remaining
            support_choice = min(close, key=lambda index: float(abs(support[index] - support[best])))
            remaining.remove(support_choice)
            wrong = [index for index in remaining if gap[index] >= 1.]
            prior_fallback = not wrong
            if not wrong:
                wrong = remaining
            prior_choice = max(wrong, key=lambda index: float(beam_prior[index]))
            remaining.remove(prior_choice)
            fourth = remaining[int(torch.randint(len(remaining), (1,), generator=subject_rng))]
            slots = torch.tensor([best, support_choice, prior_choice, fourth], device='cuda')
            choice = beam[:, slots]
            chosen_rigid_mm = beam_error[slots]
            chosen_support = support[slots]
            reflected = choice % 2
            mode = choice // 2
            chosen_state = prediction['state'].gather(1, mode[..., None].expand(-1, -1, 12))
            fitted = fitter(prediction['feature'], chosen_state, reflected,
                context['atlas'], accepted['offsets'], accepted['weights'],
                source_shape=(side, side))
            truth_ccf = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                (32, 32), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            visible = F.interpolate(accepted['valid_mask'][:, None].float(),
                (32, 32), mode='bilinear', align_corners=False)[:, 0] == 1
            available = (fitted['fine_match_support'] >= .5).any(2)
            correct = (visible[:, None] & available &
                ((fitted['fine_match_ccf_um'] - truth_ccf[:, None]).norm(dim=-1) <= 1500))
            fraction = correct.float().sum((-2, -1)) / visible.sum((-2, -1)).clamp_min(1)[:, None]

        optimizer.zero_grad(set_to_none=True)
        quality_logit = head(fitted)['quality_logit']
        fraction_bce = F.binary_cross_entropy_with_logits(quality_logit, fraction)
        score = prior.gather(1, choice) + F.logsigmoid(quality_logit)
        pair_terms = []
        for left in range(4):
            for right in range(left + 1, 4):
                delta = chosen_rigid_mm[right] - chosen_rigid_mm[left]
                if abs(float(delta)) >= .5:
                    pair_terms.append(F.softplus(.3 - delta.sign() *
                        (score[0, left] - score[0, right])))
        rank_loss = torch.stack(pair_terms).mean() if pair_terms else score.sum() * 0
        loss = fraction_bce + .1 * rank_loss
        factor = .1 + .9 * .5 * (1 + math.cos(math.pi * step / batches))
        optimizer.param_groups[0]['lr'] = 3e-4 * factor
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': chosen_rigid_mm.tolist(), 'support': chosen_support.tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'actual_fraction': fraction[0].tolist(),
            'predicted_fraction': quality_logit[0].sigmoid().detach().tolist(),
            'score': score[0].detach().tolist(),
            'fraction_bce': float(fraction_bce.detach()),
            'pair_rank_loss': float(rank_loss.detach()), 'total_loss': float(loss.detach()),
            'gradient_norm': float(gradient), 'learning_rate': optimizer.param_groups[0]['lr'],
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'actual_fraction',
                'predicted_fraction', 'fraction_bce', 'pair_rank_loss',
                'peak_gpu_mb', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'draw_attempts': attempts,
    'protocol_sha256': sha(protocol), 'parent_094_checkpoint_sha256': sha(parent),
    'source_sha256': config['source_sha256'], 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'fraction_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
