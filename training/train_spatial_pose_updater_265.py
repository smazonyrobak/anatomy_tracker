"""Bounded, matched full-atlas/support-only spatial pose pilot."""
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

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.spatial_pose_candidate_265 import (
    SpatialPoseUpdater, candidate_bank, five_point_error, render_coarse, render_exact,
)

PARENT = ROOT / 'runs/dense_physical_pose_260_train_001'
FRESH = ROOT / 'data/spatial_pose_fresh_dev_265_001'
RUN = ROOT / 'runs/spatial_pose_updater_265_train_001'
DOC = Path('docs/publication/SPATIAL_POSE_UPDATER_265_PROTOCOL_20261012.md')
SEED, BATCHES, SIDE = 2026101265, 3000, 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_done = json.loads((PARENT / 'completed.json').read_text())
fresh_done = json.loads((FRESH / 'completed.json').read_text())
assert hashlib.sha256((PARENT / 'joint_step_03000.pt').read_bytes()).hexdigest() == parent_done['checkpoint_sha256']['3000']
assert hashlib.sha256((FRESH / 'records.jsonl').read_bytes()).hexdigest() == fresh_done['records_sha256']
assert fresh_done['prior_physical_ids_disjoint'] and fresh_done['synthetic_plans'] == 8
context = load_streaming_synthetic_v7_64(device='cuda')
parent = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
    dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
parent.load_state_dict(torch.load(PARENT / 'joint_step_03000.pt',
                                   map_location='cpu', weights_only=True)['model'], strict=True)
parent.requires_grad_(False)
anchor_model = OneShotJointSliceModel(modes=0, normal_anchor_count=64)
anchors = anchor_model.normal_anchor_frames.cuda()
del anchor_model
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
full = SpatialPoseUpdater().cuda().train()
support = SpatialPoseUpdater().cuda().train()
support.load_state_dict(full.state_dict())
models = {'full': full, 'support': support}
optimizers = {name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
              for name, model in models.items()}
subject_rng = torch.Generator().manual_seed(SEED)
candidate_rng = torch.Generator(device='cuda').manual_seed(SEED)
draw_seed = SEED * 10000000

sources = ('train_spatial_pose_updater_265.py', 'spatial_pose_candidate_265.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'arbitrary_plane_full_frame_primitives.py')
config = {'seed': SEED, 'batches': BATCHES, 'batch_size': 1, 'side': SIDE,
    'candidate_bank': 6144, 'candidate_training_draw': 16,
    'coarse_atlas_square_mm': 32, 'coarse_raster': 48, 'fine_raster': 96,
    'fine_top_k': 32, 'psf_nominal_um': 25,
    'initialization': 'new full/support pair from identical fresh random weights; frozen project-only 260 centre scaffold; no external weights, features or pseudolabels',
    'parent_checkpoint_sha256': parent_done['checkpoint_sha256']['3000'],
    'fresh_dev_completed_sha256': hashlib.sha256((FRESH / 'completed.json').read_bytes()).hexdigest(),
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in sources},
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
started = time.perf_counter()

with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for batch in range(1, BATCHES + 1):
        while True:
            subject = int(torch.randint(len(context['subjects']), (1,), generator=subject_rng))
            sample = sample_one_shot_slide_artifacts_v7(context, [subject], draw_seed, side=SIDE)
            draw_seed += 1
            identity = sample['provenance'][0]
            used = bool(sample['eligible'][0])
            draws.write(json.dumps({**identity, 'batch': batch, 'used': used}) + '\n')
            if used:
                break
        image = sample['inputs']
        truth = sample['state']
        true_flag = sample['reflection']
        with torch.no_grad():
            predicted = parent.dense_coordinate_plane(parent.predict(image), source_side=SIDE)
            centre = full_frame_state_to_components(predicted)[0][0]
            bank, flags, _ = candidate_bank(centre, anchors)
            initial_error = five_point_error(bank, flags, truth, true_flag)
            order = initial_error.argsort()
            indices = torch.cat((order[:4],
                order[4:64][torch.randperm(60, generator=candidate_rng, device='cuda')[:4]],
                order[64:512][torch.randperm(448, generator=candidate_rng, device='cuda')[:4]],
                order[512:][torch.randperm(6144 - 512, generator=candidate_rng,
                                          device='cuda')[:4]]))
            candidate = bank[indices]
            reflected = flags[indices]
            e0 = initial_error[indices]
            full_coarse = render_coarse(context['atlas'], centre, anchors, indices)
            support_coarse = torch.cat((torch.zeros_like(full_coarse[:, :1]),
                                        full_coarse[:, 1:2]), 1)
        results = {}
        for name, model in models.items():
            coarse = full_coarse if name == 'full' else support_coarse
            query = model.encode_image(image)
            score0, update0, _ = model(query, coarse)
            state1 = compose_full_frame_state(candidate, update0)
            slab1 = render_exact(context['atlas'], state1, reflected,
                                 support_only=name == 'support')
            score1, update1, _ = model(query, slab1)
            state2 = compose_full_frame_state(state1, update1)
            slab2 = render_exact(context['atlas'], state2, reflected,
                                 support_only=name == 'support')
            score2, _, _ = model(query, slab2)
            e1 = five_point_error(state1, reflected, truth, true_flag)
            e2 = five_point_error(state2, reflected, truth, true_flag)
            target0 = F.softmax(-e0 / .8, 0).detach()
            target2 = F.softmax(-e2.detach() / .8, 0)
            near = F.softmax(-e0 / 1.2, 0).detach()
            loss = (-(target0 * F.log_softmax(score0, 0)).sum()
                    -(target2 * F.log_softmax(score2, 0)).sum()
                    + .5 * (near * e1).sum() + (near * e2).sum()
                    + .3 * (F.softmax(score2, 0) * e2).sum())
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizers[name].step()
            results[name] = {'loss': float(loss.detach()), 'initial_best_five_mm': float(e0.min()),
                'corrected_best_five_mm': float(e2.detach().min()),
                'selected_five_mm': float(e2.detach()[score2.detach().argmax()]),
                'gradient': float(gradient)}
        if batch == 1 or batch % 100 == 0:
            row = {'batch': batch, 'unique_eligible_physical_sections': batch,
                   'full': results['full'], 'support': results['support'],
                   'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            log.flush()
            draws.flush()
            if batch == 1 or batch % 500 == 0:
                print(json.dumps(row), flush=True)
        if batch in (1000, 2000, 3000):
            torch.save({'batch': batch, 'full': full.state_dict(),
                'support': support.state_dict(),
                'full_optimizer': optimizers['full'].state_dict(),
                'support_optimizer': optimizers['support'].state_dict(),
                'subject_rng': subject_rng.get_state(),
                'candidate_rng': candidate_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'config': config, 'calibrated': False},
                RUN / f'joint_step_{batch:05d}.pt')

(RUN / 'completed.json').write_text(json.dumps({
    'batches': BATCHES, 'unique_eligible_physical_sections': BATCHES,
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256((RUN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
                          for step in (1000, 2000, 3000)},
    'calibrated': False, 'biological_validation': False,
    'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'batches': BATCHES,
                  'seconds': time.perf_counter() - started}), flush=True)
