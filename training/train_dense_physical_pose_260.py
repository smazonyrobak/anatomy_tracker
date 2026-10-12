"""Continue the 258 dense model with explicit fitted centre, edge and span supervision."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT = ROOT / 'runs/dense_joint_geometry_258_train_001'
CHECKPOINT = PARENT / 'joint_step_05000.pt'
RUN = ROOT / 'runs/dense_physical_pose_260_train_001'
DOC = Path('docs/publication/DENSE_PHYSICAL_POSE_260_PROTOCOL_20261012.md')
BATCHES, BATCH, SIDE = 3000, 4, 192
SAVES = (0, 1000, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_done = json.loads((PARENT / 'completed.json').read_text())
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == parent_done['checkpoint_sha256']['5000']
assert hashlib.sha256((PARENT / 'config.json').read_bytes()).hexdigest() == parent_done['config_sha256']
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
                              dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
assert saved['step'] == 5000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
optimizer.load_state_dict(saved['optimizer'])
subjects_rng = torch.Generator()
subjects_rng.set_state(saved['subjects_rng'])
draw_seed = saved['draw_seed']
torch.set_rng_state(saved['torch_rng'])
torch.cuda.set_rng_state_all(saved['cuda_rng'])
del saved
axis = (torch.arange(SIDE // 2, device='cuda') + .5) / (SIDE // 2) - .5 / SIDE
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
design = torch.cat((torch.ones_like(chart[..., :1]), chart), -1).reshape(-1, 3)

RUN.mkdir(parents=True, exist_ok=False)
sources = (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
           'arbitrary_plane_joint_model_v7.py',
           'arbitrary_plane_recurrent_model.py',
           'arbitrary_plane_ribbon_v6.py',
           'arbitrary_plane_one_shot_slide_artifacts_v7.py',
           'arbitrary_plane_one_shot_slide_artifacts_v6.py',
           'arbitrary_plane_one_shot_slide_artifacts_v4.py',
           'arbitrary_plane_one_shot_slide_artifacts_v3.py',
           'arbitrary_plane_one_shot_stream.py',
           'arbitrary_plane_streaming_synthetic_v7_64.py',
           'arbitrary_plane_streaming_synthetic_v7.py',
           'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
           'arbitrary_plane_subject_torch_v6.py',
           'arbitrary_plane_subject_sampling_v6.py',
           'subject_deformed_slab_multiresolution_bundle_v2.py',
           'arbitrary_plane_allen_atlas_binding_v6.py',
           'arbitrary_plane_full_frame_primitives.py',
           'arbitrary_plane_geometry.py')
config = {
    'parent_checkpoint_sha256': parent_done['checkpoint_sha256']['5000'],
    'parent_completed_sha256': hashlib.sha256((PARENT / 'completed.json').read_bytes()).hexdigest(),
    'batches': BATCHES, 'batch_size': BATCH, 'side': SIDE, 'saves': SAVES,
    'prior_presentations': parent_done['synthetic_presentations'],
    'initialization': 'continuation of project-only fresh-random 258; no external weights, features or pseudolabels',
    'loss': '258 dense Charbonnier + 0.2 balanced validity + 0.5 fitted rigid mm + 0.25 centre mm + 0.4 mean edge-vector mm + 0.5 mean edge-span log ratio + 0.5 unsigned normal cosine',
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in sources},
    'synthetic_provenance': context['provenance'],
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
                                      ('inputs', 'centre', 'valid_mask', 'state', 'reflection')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)])
                 for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        field = prediction['dense_coordinate']
        valid = F.avg_pool2d(batch['valid_mask'][:, None].float(), 2)[:, 0] > .999
        target = F.avg_pool2d(batch['centre'].permute(0, 3, 1, 2), 2)
        coordinate = model.center_origin[None, :, None, None] + (
            field[:, :3] * model.center_scale[None, :, None, None])
        pixel_mm = (coordinate - target).norm(dim=1) / 1000
        dense_mm = ((pixel_mm.square() + .05**2).sqrt() - .05).mul(valid).sum((1, 2)) \
            / valid.sum((1, 2)).clamp_min(1)
        logits = field[:, 3]
        validity = .5 * (F.softplus(-logits) * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1) \
            + .5 * (F.softplus(logits) * ~valid).sum((1, 2)) / (~valid).sum((1, 2)).clamp_min(1)
        weight = logits.sigmoid().flatten(1).clamp_min(1e-6)
        gram = torch.einsum('ni,bn,nj->bij', design, weight, design)
        right = torch.einsum('ni,bn,bnj->bij', design, weight,
                             coordinate.flatten(2).transpose(1, 2))
        coefficients = torch.linalg.solve(gram + 1e-6 * torch.eye(3, device='cuda')[None], right)
        rigid = torch.einsum('ni,bij->bnj', design, coefficients).reshape(BATCH, SIDE // 2,
                                                                         SIDE // 2, 3)
        plane_mm = ((rigid - target.permute(0, 2, 3, 1)).norm(dim=-1) / 1000 * valid).sum((1, 2)) \
            / valid.sum((1, 2)).clamp_min(1)
        predicted_u, predicted_v = coefficients[:, 1], coefficients[:, 2]
        predicted_centre = coefficients[:, 0] + .5 * (predicted_u + predicted_v)
        true_centre, true_frame, true_basis = full_frame_state_to_components(batch['state'])
        true_edges = true_frame[:, :, :2] @ true_basis
        true_u, true_v = true_edges[:, :, 0], true_edges[:, :, 1]
        reflected = batch['reflection'].bool()
        target_u = torch.where(reflected[:, None], -true_u, true_u)
        target_centre = true_centre - reflected[:, None] * true_u / SIDE
        centre_mm = (predicted_centre - target_centre).norm(dim=-1) / 1000
        edge_mm = .5 * ((predicted_u - target_u).norm(dim=-1)
                        + (predicted_v - true_v).norm(dim=-1)) / 1000
        span_log = .5 * ((predicted_u.norm(dim=-1).clamp_min(1).log()
                          - target_u.norm(dim=-1).log()).abs()
                         + (predicted_v.norm(dim=-1).clamp_min(1).log()
                            - true_v.norm(dim=-1).log()).abs())
        predicted_normal = F.normalize(torch.cross(predicted_u, predicted_v, dim=-1),
                                       dim=-1, eps=1e6)
        normal_cosine = (predicted_normal * true_frame[:, :, 2]).sum(-1).abs().clamp_max(1)
        axis_cost = 1 - normal_cosine
        loss = (dense_mm + .2 * validity + .5 * plane_mm + .25 * centre_mm
                + .4 * edge_mm + .5 * span_log + .5 * axis_cost).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        dice = 2 * ((logits > 0) & valid).sum((1, 2)) / (
            (logits > 0).sum((1, 2)) + valid.sum((1, 2))).clamp_min(1)
        record = {'batch': batch_number, 'new_synthetic_presentations': batch_number * BATCH,
                  'total_synthetic_presentations': config['prior_presentations'] + batch_number * BATCH,
                  'dense_charbonnier_mm': float(dense_mm.detach().mean()),
                  'fitted_plane_mm': float(plane_mm.detach().mean()),
                  'validity_balanced_bce': float(validity.detach().mean()),
                  'validity_dice': float(dice.detach().float().mean()),
                  'centre_mm': float(centre_mm.detach().mean()),
                  'edge_vector_mm': float(edge_mm.detach().mean()),
                  'span_log_ratio': float(span_log.detach().mean()),
                  'normal_axis_cost': float(axis_cost.detach().mean()),
                  'predicted_minor_raster_span_mm': float(torch.minimum(
                      predicted_u.norm(dim=-1), predicted_v.norm(dim=-1)).detach().mean() / 1000),
                  'loss': float(loss.detach()), 'gradient': float(gradient),
                  'clipped': bool(gradient > 5), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(record) + '\n')
        if batch_number == 1 or batch_number % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(record), flush=True)

(RUN / 'completed.json').write_text(json.dumps({
    'batches': BATCHES, 'new_synthetic_presentations': BATCHES * BATCH,
    'total_synthetic_presentations': config['prior_presentations'] + BATCHES * BATCH,
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256((RUN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
                          for step in SAVES},
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'batches': BATCHES,
                  'seconds': time.perf_counter() - started}), flush=True)
