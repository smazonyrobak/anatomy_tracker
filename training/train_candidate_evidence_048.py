"""Learn to choose the anatomically correct fitted branch from inference-only evidence."""
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
from torch import nn

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.candidate_evidence_048 import candidate_evidence
from training.joint_pose_fit_047 import joint_pose_fit
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
fine_path = root / 'runs/pose_fine_subgrid_046_pilot/joint_step_02000.pt'
run = root / 'runs/candidate_evidence_048_pilot'
seed, batches, beam = 2026104800, 1000, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
context = load_streaming_synthetic_v7_64(device='cuda')
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                             vector_refinement=True, candidate_ranking=True,
                             fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
pose.requires_grad_(False)
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(coarse_path, map_location='cpu', weights_only=True)['head'])
coarse.requires_grad_(False)
fine = PoseFineMatch046().cuda().eval()
fine.load_state_dict(torch.load(fine_path, map_location='cpu', weights_only=True)['model'])
fine.requires_grad_(False)
ranker = nn.Sequential(nn.Linear(11, 32), nn.GELU(), nn.Linear(32, 1)).cuda().train()
nn.init.zeros_(ranker[-1].weight)
nn.init.zeros_(ranker[-1].bias)
optimizer = torch.optim.AdamW(ranker.parameters(), lr=1e-3, weight_decay=1e-3)

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'sections_per_batch': 1,
          'beam': beam, 'candidate_features': 11,
          'training': 'fresh unique physical synthetic TRAIN sections, one randomized appearance per plane',
          'trainable': 'new randomly initialized 11-32-1 candidate evidence ranker only',
          'frozen_lineage': 'scratch 019 pose/warp, 041 coarse matcher, 046 fine matcher',
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'coarse_sha256': hashlib.sha256(coarse_path.read_bytes()).hexdigest(),
          'fine_sha256': hashlib.sha256(fine_path.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_candidate_evidence_048.py',
                                         'candidate_evidence_048.py', 'joint_pose_fit_047.py')},
          'synthetic_provenance': context['provenance'],
          'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.save({'model': ranker.state_dict(), 'optimizer': optimizer.state_dict(),
            'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'step': 0, 'config': config, 'calibrated': False},
           run / 'joint_step_00000.pt')
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        while True:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=256)
            draw_seed += 1
            used = bool(sampled['eligible'][0] and sampled['valid_mask'][0].any())
            draws.write(json.dumps({**sampled['provenance'][0], 'batch': step,
                                    'used': used}) + '\n')
            if used:
                break
        with torch.inference_mode():
            prediction = pose.predict(sampled['inputs'])
            fitted = joint_pose_fit(pose, coarse, fine, prediction, sampled['inputs'],
                context['atlas'], sampled['offsets'], sampled['weights'], beam=beam)
            evidence = candidate_evidence(prediction, fitted)
            index = torch.multinomial(sampled['valid_mask'].flatten(1).float(), 128,
                                      replacement=True)
            truth = sampled['centre'].reshape(1, -1, 3).gather(
                1, index[..., None].expand(-1, -1, 3))
            grid = (torch.stack((index.remainder(256),
                                 index.div(256, rounding_mode='floor')), -1).float() + .5
                    ) * (2 / 256) - 1
            grid = grid[:, None].expand(-1, beam, -1, -1).reshape(beam, 1, 128, 2)
            surface = fitted['mapped']['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1)
            points = F.grid_sample(surface.permute(0, 3, 1, 2), grid,
                mode='bilinear', padding_mode='border', align_corners=False)
            points = points.reshape(1, beam, 3, 128).permute(0, 1, 3, 2)
            error = (points - truth[:, None]).norm(dim=-1).mean(-1)
        evidence = evidence.clone()
        prior = fitted['prior'].clone()
        error = error.clone()
        score = prior + ranker(evidence).squeeze(-1)
        hard = F.cross_entropy(score, error.argmin(-1))
        soft = F.kl_div(F.log_softmax(score, -1),
                        (-error / 500).softmax(-1), reduction='batchmean')
        loss = .5 * hard + .5 * soft
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        selected = score.detach().argmax(-1)
        row = {'batch': step, 'synthetic_presentations': step,
               'loss': float(loss.detach()), 'hard_loss': float(hard.detach()),
               'soft_loss': float(soft.detach()),
               'selected_mapped_um': float(error.gather(1, selected[:, None]).mean()),
               'best_eight_mapped_um': float(error.min(-1).values.mean()),
               'gradient_norm': float(gradient),
               'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 100 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 500 == 0:
            torch.save({'model': ranker.state_dict(), 'optimizer': optimizer.state_dict(),
                        'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'step': step, 'config': config, 'calibrated': False},
                       run / f'joint_step_{step:05d}.pt')
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'synthetic_presentations': batches,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
