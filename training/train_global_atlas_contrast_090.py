"""TRAIN-only 15,000-section pilot of new whole-plane image/atlas contrast."""
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
protocol = source.parent / 'docs/publication/GLOBAL_ATLAS_CONTRAST_090_PROTOCOL_20261004.md'
parent_run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
parent = parent_run / 'joint_step_06000.pt'
run = root / 'runs/global_atlas_contrast_090_pilot'
seed, batches, side, pixels = 2026100490, 15000, 256, 1024
checkpoints = (0, 3000, 8000, 15000)
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


assert sha(protocol) == '1955336305c9a65d58aad89a0eccdbf4b90d331521624688483131c4d8d03b01'
assert sha(parent_run / 'completed.json') == '7b39c3c89ee2c590b598e3596a50afea6051b664bfca243cf47220b6cc27c0da'
receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert receipt['batches'] == receipt['accepted_synthetic'] == parent_config['batches'] == 6000
assert parent_config['synthetic_provenance']['all_train_bases'] == 64
assert parent_config['candidate_beam'] == {'old': 8, 'anchor': 6}
assert not any(receipt[key] for key in ('calibrated', 'public_benchmark_used',
                                        'real_labels_used', 'external_pretrained_weights_used'))
assert sha(parent_run / 'config.json') == receipt['config_sha256']
assert sha(parent_run / 'draws.jsonl') == receipt['draws_sha256']
assert sha(parent_run / 'training.jsonl') == receipt['training_sha256']
assert sha(parent) == receipt['checkpoint_sha256']['6000'] \
    == 'dd49c68416f406d9d9a75f54b235f9c0984cadf0aea0b462febb87e6fbd11f0a'
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())

context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance']['all_train_bases'] == 64
assert context['provenance']['source_sha256'] == parent_config['synthetic_provenance']['source_sha256']
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
head = WholeSliceAtlasFeedback083().cuda().eval()
head.load_state_dict(checkpoint['feedback'], strict=True)
del checkpoint
model.requires_grad_(False)
head.requires_grad_(False)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
scorer = GlobalAtlasContrast090().cuda().train()
optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
attempts = 0

config = {'seed': seed, 'batches': batches, 'batch_size': 1,
    'accepted_synthetic_per_batch': 1, 'checkpoints': list(checkpoints),
    'candidate_beam': {'old': 8, 'anchor': 6}, 'candidate_count_train': 15,
    'candidate_count_inference': 14,
    'candidate_source': 'blind frozen-089 pose-prior 8-old/6-anchor IDs; two frozen 083 feedback passes with frozen 64-grid map fit summary',
    'physical_positive': 'one-shot observed affine physical state/reflection, TRAIN slot 14 only; never in inference beam',
    'atlas_planes': 'rigid corrected full-frame pose/reflection; 96 grid at 256-source pixel centres, exact finite PSF offsets/weights, raw intensity divided by support plus support',
    'physical_target': 'mean rigid 3-D CCF error on up to 1024 evenly sampled surviving observed pixels; softmax(-error_mm/0.5)',
    'objective': 'listwise soft-target cross-entropy; only new dual CNNs, projections and bounded log temperature trainable',
    'optimizer': 'AdamW lr 3e-4 weight_decay 1e-4, cosine decay to zero, gradient clip 5',
    'candidate_render_chunk': 2, 'candidate_encoder_chunk': 3,
    'parent_089_checkpoint': str(parent),
    'parent_089_checkpoint_sha256': sha(parent),
    'parent_089_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_089_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_global_atlas_contrast_090.py', 'global_atlas_contrast_090.py',
        'arbitrary_plane_one_shot_model.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py', 'arbitrary_plane_one_shot_stream.py',
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
        run / f'contrast_step_{step:05d}.pt')


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
            draws.write(json.dumps({**record, 'step': step, 'used': used}) + '\n')
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
            states = torch.cat((*corrected, accepted['state'][:, None]), 1)
            reflection = torch.cat((beam % 2, accepted['reflection'][:, None]), 1)
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
            target = F.softmax(-physical_mm / .5, dim=0)
            support = atlas_pair[0, :, 1].mean((-2, -1))
            sample_sha = hashlib.sha256(chosen.to(torch.int32).cpu().numpy().tobytes()).hexdigest()

        optimizer.zero_grad(set_to_none=True)
        scores = scorer(accepted['inputs'], atlas_pair, candidate_chunk=3)[0]
        loss = -(target * F.log_softmax(scores, dim=0)).sum()
        temperature = float(scorer.log_temperature.detach().clamp(
            math.log(2.), math.log(20.)).exp())
        optimizer.param_groups[0]['lr'] = 3e-4 * .5 * (1 + math.cos(math.pi * step / batches))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        positive_rank = int((scores.detach() > scores.detach()[-1]).sum()) + 1
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'],
            'candidate_ids': [*beam[0].tolist(), 'physical_positive'],
            'candidate_states': states[0].tolist(),
            'candidate_reflection': reflection[0].tolist(),
            'candidate_support_mean': support.tolist(),
            'physical_error_mm': physical_mm.tolist(), 'target_probability': target.tolist(),
            'sample_indices_sha256': sample_sha, 'sample_pixels': len(chosen),
            'scores': scores.detach().tolist(), 'positive_rank': positive_rank,
            'loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'temperature': temperature,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in (
                'batch', 'draw_attempt', 'positive_rank', 'loss', 'gradient_norm',
                'temperature', 'learning_rate', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'draw_attempts': attempts,
    'protocol_sha256': sha(protocol),
    'parent_089_checkpoint_sha256': sha(parent),
    'parent_089_completed_sha256': sha(parent_run / 'completed.json'),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'contrast_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
