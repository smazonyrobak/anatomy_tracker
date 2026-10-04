"""TRAIN-only blind-candidate anatomical ranking from the frozen 090 scorer."""
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
os.environ['TRITON_CACHE_DIR'] = str(root / 'cache/triton')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import (
    GlobalAtlasContrast090, render_atlas_planes_090, rigid_points_090,
)
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_ATLAS_BLIND_RANK_091_PROTOCOL_20261004.md'
parent_089_run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
parent_089 = parent_089_run / 'joint_step_06000.pt'
parent_090_run = root / 'runs/global_atlas_contrast_090_pilot'
parent_090 = parent_090_run / 'contrast_step_15000.pt'
run = root / 'runs/global_atlas_blind_rank_091_pilot'
seed, batches, side, pixels = 2026100491, 30000, 256, 1024
checkpoints = (0, 5000, 15000, 30000)
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


assert sha(protocol) == '9657708cfba91b683072639230712ffbac3565234314d70fb1d948fccae3cf99'
assert sha(parent_089_run / 'completed.json') == '7b39c3c89ee2c590b598e3596a50afea6051b664bfca243cf47220b6cc27c0da'
receipt_089 = json.loads((parent_089_run / 'completed.json').read_text())
config_089 = json.loads((parent_089_run / 'config.json').read_text())
assert receipt_089['batches'] == receipt_089['accepted_synthetic'] == config_089['batches'] == 6000
assert config_089['synthetic_provenance']['all_train_bases'] == 64
assert config_089['candidate_beam'] == {'old': 8, 'anchor': 6}
assert not any(receipt_089[key] for key in ('calibrated', 'public_benchmark_used',
                                            'real_labels_used', 'external_pretrained_weights_used'))
assert sha(parent_089_run / 'config.json') == receipt_089['config_sha256']
assert sha(parent_089_run / 'draws.jsonl') == receipt_089['draws_sha256']
assert sha(parent_089_run / 'training.jsonl') == receipt_089['training_sha256']
assert sha(parent_089) == receipt_089['checkpoint_sha256']['6000'] \
    == 'dd49c68416f406d9d9a75f54b235f9c0984cadf0aea0b462febb87e6fbd11f0a'
assert all(sha(source / name) == digest for name, digest in config_089['source_sha256'].items())

assert sha(parent_090_run / 'completed.json') == 'd547c347871f8f111e9405aca38d8b75dcc20ec1fb48219a6519ab3ee93262b2'
receipt_090 = json.loads((parent_090_run / 'completed.json').read_text())
config_090 = json.loads((parent_090_run / 'config.json').read_text())
assert receipt_090['batches'] == receipt_090['accepted_synthetic'] == config_090['batches'] == 15000
assert tuple(config_090['checkpoints']) == (0, 3000, 8000, 15000)
assert config_090['parent_089_checkpoint_sha256'] == sha(parent_089)
assert not any(receipt_090[key] or config_090[key] for key in (
    'calibrated', 'public_benchmark_used', 'real_labels_used', 'external_pretrained_weights_used'))
assert sha(parent_090_run / 'config.json') == receipt_090['config_sha256'] \
    == 'bf281953f9b168444c6e984235ac966b9acfa855afafa96685a6247bbf35f9e7'
assert sha(parent_090_run / 'draws.jsonl') == receipt_090['draws_sha256'] \
    == 'f31a2e328e430903d108da086a48f42da47c89f9ebd97b0fb47ddb1df0815fc7'
assert sha(parent_090_run / 'training.jsonl') == receipt_090['training_sha256'] \
    == '542f86524287619985db31c497ddf277edd984123d00c0da3049b2febbe1a17a'
assert sha(parent_090) == receipt_090['checkpoint_sha256']['15000'] \
    == 'dc18a41478a628d5815a5da4ddca290c3700e47ec2547b84e50c063fc157e9dc'
assert all(sha(source / name) == digest for name, digest in config_090['source_sha256'].items())

context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance']['all_train_bases'] == 64
assert context['provenance']['source_sha256'] == config_090['synthetic_provenance']['source_sha256']
checkpoint_089 = torch.load(parent_089, map_location='cpu', weights_only=True)
assert checkpoint_089['step'] == 6000 and not checkpoint_089['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint_089['model'], strict=True)
head = WholeSliceAtlasFeedback083().cuda().eval()
head.load_state_dict(checkpoint_089['feedback'], strict=True)
del checkpoint_089
model.requires_grad_(False)
head.requires_grad_(False)

checkpoint_090 = torch.load(parent_090, map_location='cpu', weights_only=True)
assert checkpoint_090['step'] == 15000 and not checkpoint_090['calibrated']
assert checkpoint_090['config'] == config_090
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
scorer = GlobalAtlasContrast090().cuda().train()
scorer.load_state_dict(checkpoint_090['scorer'], strict=True)
del checkpoint_090
optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
attempts = 0

config = {'seed': seed, 'batches': batches, 'batch_size': 1,
    'accepted_synthetic_per_batch': 1, 'checkpoints': list(checkpoints),
    'candidate_beam': {'old': 8, 'anchor': 6}, 'candidate_count_train': 14,
    'candidate_count_inference': 14,
    'candidate_source': 'blind frozen-089 pose-prior 8-old/6-anchor IDs; two frozen 083 feedback passes with frozen 64-grid map fit summary; no physical reference',
    'atlas_planes': 'rigid corrected full-frame pose/reflection; 96 grid at 256-source pixel centres, exact finite PSF offsets/weights, raw intensity divided by support plus support',
    'physical_target': 'mean rigid 3-D CCF error on up to 1024 evenly sampled surviving observed pixels',
    'objective': 'all blind better-worse pairs with error gap >=0.5 mm; softplus(1-(score_better-score_worse)); weight 4 if support gap <=0.05 else 1; divide by sum of weights per section',
    'zero_pair_section': 'accepted and logged; no optimizer update',
    'sampling': 'fresh independent TRAIN plane and one appearance per attempt; no deliberate physical-plane deduplication, reuse or paired appearances',
    'optimizer': 'new AdamW lr 1e-4 weight_decay 1e-4, cosine decay to zero over 30000 sections, gradient clip 5; train all 090 scorer parameters',
    'candidate_render_chunk': 2, 'candidate_encoder_chunk': 3,
    'parent_089_checkpoint': str(parent_089),
    'parent_089_checkpoint_sha256': sha(parent_089),
    'parent_089_completed_sha256': sha(parent_089_run / 'completed.json'),
    'parent_089_config_sha256': sha(parent_089_run / 'config.json'),
    'parent_090_checkpoint': str(parent_090),
    'parent_090_checkpoint_sha256': sha(parent_090),
    'parent_090_completed_sha256': sha(parent_090_run / 'completed.json'),
    'parent_090_config_sha256': sha(parent_090_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_global_atlas_blind_rank_091.py', 'train_global_atlas_contrast_090.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_one_shot_model.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py',
        'arbitrary_plane_full_frame_primitives.py')},
    'synthetic_provenance': context['provenance'], 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))

run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'scorer': scorer.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'rank_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            attempts += 1
            record = sampled['provenance'][0]
            used = bool(sampled['eligible'][0])
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
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
            initial = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
            corrected = []
            for start in range(0, 14, 2):
                branch = beam[:, start:start + 2]
                reflected = branch % 2
                first = head(prediction['feature'], initial[:, start:start + 2],
                             reflected, context['atlas'], accepted['offsets'], accepted['weights'])
                selected = {**prediction, 'state': first['state'],
                    'log_mass': prediction['log_mass'].gather(1, branch // 2),
                    'reflection_logit': prediction['reflection_logit'].gather(1, branch // 2)}
                mapped64 = model.map(selected, accepted['offsets'],
                    torch.arange(2, device='cuda')[None], reflected, (64, 64),
                    context['atlas'], accepted['weights'], return_refinement_feature=True,
                    feature_side=64, source_shape=(side, side),
                    spatial_evidence=first['spatial_evidence'])
                second = head(prediction['feature'], first['state'], reflected,
                    context['atlas'], accepted['offsets'], accepted['weights'],
                    fit_summary=fit_summary(mapped64))
                corrected.append(second['state'])
            states = torch.cat(corrected, 1)
            reflection = beam % 2
            atlas_pair = render_atlas_planes_090(context['atlas'], states, reflection,
                accepted['offsets'], accepted['weights'], candidate_chunk=2)
            valid = accepted['valid_mask'][0].flatten().nonzero().flatten()
            chosen = valid[torch.linspace(0, len(valid) - 1, min(pixels, len(valid)),
                                        device='cuda').round().long()]
            chart = torch.stack((chosen.remainder(side),
                chosen.div(side, rounding_mode='floor')), -1).float() / side
            target_ccf = accepted['centre'][0].reshape(-1, 3)[chosen]
            physical_mm = (rigid_points_090(states[0], reflection[0], chart)
                           - target_ccf[None]).norm(dim=-1).mean(-1) / 1000
            support = atlas_pair[0, :, 1].mean((-2, -1))
            better, worse = torch.where(
                physical_mm[None, :] - physical_mm[:, None] >= .5)
            matched = (support[better] - support[worse]).abs() <= .05
            weights = 1 + 3 * matched.to(torch.float32)
            sample_sha = hashlib.sha256(chosen.to(torch.int32).cpu().numpy().tobytes()).hexdigest()

        optimizer.zero_grad(set_to_none=True)
        scores = scorer(accepted['inputs'], atlas_pair, candidate_chunk=3)[0]
        loss = ((weights * F.softplus(1 - (scores[better] - scores[worse]))).sum()
                / weights.sum()) if len(better) else scores.sum() * 0
        temperature = float(scorer.log_temperature.detach().clamp(
            math.log(2.), math.log(20.)).exp())
        optimizer.param_groups[0]['lr'] = 1e-4 * .5 * (1 + math.cos(math.pi * step / batches))
        gradient = 0.
        if len(better):
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                scorer.parameters(), 5., error_if_nonfinite=True))
            optimizer.step()
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'],
            'candidate_ids': beam[0].tolist(),
            'candidate_states': states[0].tolist(),
            'candidate_reflection': reflection[0].tolist(),
            'candidate_support_mean': support.tolist(),
            'physical_error_mm': physical_mm.tolist(),
            'eligible_pair_count': len(better), 'matched_pair_count': int(matched.sum()),
            'pair_weight_sum': float(weights.sum()),
            'sample_indices_sha256': sample_sha, 'sample_pixels': len(chosen),
            'scores': scores.detach().tolist(), 'loss': float(loss.detach()),
            'gradient_norm': gradient, 'temperature': temperature,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in (
                'batch', 'draw_attempt', 'eligible_pair_count', 'matched_pair_count',
                'loss', 'gradient_norm', 'temperature', 'learning_rate', 'seconds')}),
                flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'draw_attempts': attempts,
    'protocol_sha256': sha(protocol),
    'parent_089_checkpoint_sha256': sha(parent_089),
    'parent_089_completed_sha256': sha(parent_089_run / 'completed.json'),
    'parent_090_checkpoint_sha256': sha(parent_090),
    'parent_090_completed_sha256': sha(parent_090_run / 'completed.json'),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'rank_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
