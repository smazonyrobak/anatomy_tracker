"""Train direct dense tissue-to-CCF coordinates and their fitted global plane."""
import hashlib
import json
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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/dense_joint_geometry_258_train_001'
DOC = Path('docs/publication/DENSE_JOINT_GEOMETRY_258_PROTOCOL_20261012.md')
SEED, BATCHES, BATCH, SIDE = 2026101249, 5000, 4, 192
SAVES = (0, 1000, 3000, 5000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=0, dense_coordinate_pyramid=True,
    atlas_conditioning=False, vector_refinement=False, fitted_ranking=False,
).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 100000
axis = (torch.arange(SIDE // 2, device='cuda') + .5) / (SIDE // 2) - .5 / SIDE
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
design = torch.cat((torch.ones_like(chart[..., :1]), chart), -1).reshape(-1, 3)

RUN.mkdir(parents=True, exist_ok=False)
sources = (
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
    'arbitrary_plane_geometry.py',
)
config = {
    'seed': SEED, 'batches': BATCHES, 'batch_size': BATCH, 'side': SIDE,
    'saves': SAVES, 'parameters': sum(p.numel() for p in model.parameters()),
    'initialization': 'fresh random; no prior weights, features or pseudolabels',
    'geometry': 'dense pixel-to-CCF mean plus validity; weighted differentiable full-plane fit',
    'loss': 'valid-pixel Charbonnier CCF mm + 0.2 class-balanced validity BCE; after batch 1000 add 0.5 fitted-plane rigid mm',
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in sources},
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
                                      ('inputs', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)])
                 for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        field = prediction['dense_coordinate']
        valid = (F.avg_pool2d(batch['valid_mask'][:, None].float(), 2)[:, 0] > .999)
        target = F.avg_pool2d(batch['centre'].permute(0, 3, 1, 2), 2)
        coordinate = model.center_origin[None, :, None, None] + \
            field[:, :3] * model.center_scale[None, :, None, None]
        pixel_mm = (coordinate - target).norm(dim=1) / 1000
        dense_mm = ((pixel_mm.square() + .05**2).sqrt() - .05).mul(valid).sum((1, 2)) \
            / valid.sum((1, 2)).clamp_min(1)
        logits = field[:, 3]
        validity = .5 * (F.softplus(-logits) * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1) \
            + .5 * (F.softplus(logits) * ~valid).sum((1, 2)) / (~valid).sum((1, 2)).clamp_min(1)
        if batch_number > 1000:
            weight = logits.sigmoid().flatten(1).clamp_min(1e-6)
            gram = torch.einsum('ni,bn,nj->bij', design, weight, design)
            right = torch.einsum('ni,bn,bnj->bij', design, weight,
                                 coordinate.flatten(2).transpose(1, 2))
            coefficients = torch.linalg.solve(
                gram + 1e-6 * torch.eye(3, device='cuda')[None], right)
            rigid = torch.einsum('ni,bij->bnj', design, coefficients).reshape(BATCH, SIDE // 2,
                                                                             SIDE // 2, 3)
            plane_mm = ((rigid - target.permute(0, 2, 3, 1)).norm(dim=-1) / 1000 * valid).sum((1, 2)) \
                / valid.sum((1, 2)).clamp_min(1)
        else:
            plane_mm = dense_mm.detach() * 0
        loss = (dense_mm + .2 * validity + .5 * plane_mm).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        positive = (logits > 0) & valid
        dice = 2 * positive.sum((1, 2)) / ((logits > 0).sum((1, 2)) + valid.sum((1, 2))).clamp_min(1)
        record = {'batch': batch_number, 'synthetic_presentations': batch_number * BATCH,
                  'dense_charbonnier_mm': float(dense_mm.detach().mean()),
                  'fitted_plane_mm': float(plane_mm.detach().mean()),
                  'validity_balanced_bce': float(validity.detach().mean()),
                  'validity_dice': float(dice.detach().float().mean()),
                  'loss': float(loss.detach()), 'gradient': float(gradient),
                  'clipped': bool(gradient > 5), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(record) + '\n')
        if batch_number == 1 or batch_number % 500 == 0:
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
