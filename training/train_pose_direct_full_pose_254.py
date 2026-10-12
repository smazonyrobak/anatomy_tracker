"""Fresh direct-anchor pose supervision on the matched v7 TRAIN draw stream."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/pose_direct_full_pose_254_train_001'
DOC = Path('docs/publication/POSE_DIRECT_FULL_POSE_254_PROTOCOL_20261012.md')
SEED, BATCHES, BATCH, SIDE = 2026101249, 3000, 4, 192
SAVES = (0, 500, 1500, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=False,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 100000
flags = torch.tensor((0, 1), device='cuda')

RUN.mkdir(parents=True, exist_ok=False)
source_names = (
    Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_stream.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_full_frame_primitives.py',
)
config = {
    'seed': SEED, 'batches': BATCHES, 'synthetic_per_batch': BATCH, 'side': SIDE,
    'saves': SAVES, 'full_model_parameters': sum(p.numel() for p in model.parameters()),
    'initialization': 'fresh random; no prior weights, features or pseudolabels',
    'training_stage': 'direct-anchor pose and reflection only; map/refiner/scorer remain random',
    'loss': '15 degree antipodal anchor CE + 0.25 nearest-three-anchor rigid mm + 0.1 soft-target reflection BCE',
    'matched_249_250_251': 'same synthetic draw seed and subject RNG; direct head with full-pose gradients',
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in source_names},
    'real_labels_used': False, 'calibrated': False, 'public_benchmark_used': False,
}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(BATCHES + 1):
        if step in SAVES:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'step': step, 'config': config, 'calibrated': False},
                       RUN / f'joint_step_{step:05d}.pt')
        if step == BATCHES:
            break
        batch_number = step + 1
        accepted, pending = {}, list(range(BATCH))
        while pending:
            subjects = torch.randint(len(context['subjects']), (len(pending),),
                                     generator=subjects_rng).tolist()
            sampled = sample_one_shot_slide_artifacts_v7(context, subjects, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'batch': batch_number, 'slot': slot,
                                        'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)])
                 for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 256, replacement=True)
        target = batch['centre'].reshape(BATCH, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(SIDE),
                             indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        centres, frames, bases = full_frame_state_to_components(states)
        chart_reflected = chart[:, None, None].expand(BATCH, model.modes, 2, -1, -1).clone()
        chart_reflected[..., 0] = torch.where(flags[None, None, :, None].bool(),
            (SIDE - 1) / SIDE - chart_reflected[..., 0], chart_reflected[..., 0])
        rigid = (centres[..., None, :] + torch.einsum(
            '...ij,...pj->...pi', frames[..., :2] @ bases, chart_reflected - .5)
            - target[:, None, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
        cosine = (normal @ model.normal_anchor_frames[:, :, 2].T).abs().clamp_max(1)
        angle = torch.acos(cosine.clamp_max(1 - 1e-7))
        normal_target = F.softmax(-.5 * (angle / math.radians(15)) ** 2, -1)
        normal_ce = -(normal_target * prediction['log_mass']).sum(-1)
        nearest = cosine.topk(3, -1).indices
        near_weights = normal_target.gather(1, nearest)
        near_weights = near_weights / near_weights.sum(-1, keepdim=True)
        near_mm = (rigid.min(-1).values.gather(1, nearest) * near_weights).sum(-1) / 1000
        reflection_bce = (F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'],
            batch['reflection'].float()[:, None].expand_as(prediction['reflection_logit']),
            reduction='none') * normal_target).sum(-1)
        loss = (normal_ce + .25 * near_mm + .1 * reflection_bce).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        selected = (prediction['log_mass'][..., None] + reflection_log).flatten(1).argmax(-1)
        selected_mm = rigid.flatten(1).gather(1, selected[:, None]).mean() / 1000
        selected_axis = angle.gather(1, selected.div(2, rounding_mode='floor')[:, None]).mean()
        record = {'batch': batch_number, 'synthetic_presentations': batch_number * BATCH,
                  'normal_ce': float(normal_ce.detach().mean()),
                  'normal_target_entropy': float(-(normal_target * normal_target.log()).sum(-1).mean()),
                  'near_mm': float(near_mm.detach().mean()),
                  'reflection_bce': float(reflection_bce.detach().mean()),
                  'selected_rigid_mm': float(selected_mm.detach()),
                  'selected_normal_axis_deg': float(selected_axis.detach() * 180 / math.pi),
                  'loss': float(loss.detach()), 'gradient': float(gradient),
                  'clipped': bool(gradient > 5), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(record) + '\n')
        if batch_number == 1 or batch_number % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(record), flush=True)

(RUN / 'completed.json').write_text(json.dumps({
    'batches': BATCHES, 'synthetic_presentations': BATCHES * BATCH,
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256((RUN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
                          for step in SAVES},
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'batches': BATCHES,
                  'seconds': time.perf_counter() - started}), flush=True)
