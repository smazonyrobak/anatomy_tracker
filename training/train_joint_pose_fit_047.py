"""Continue the scratch pose model through correspondence, fitting, and mapping feedback."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.joint_pose_fit_047 import joint_pose_fit
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
fine_path = root / 'runs/pose_fine_subgrid_046_pilot/joint_step_02000.pt'
run = root / 'runs/joint_pose_fit_047_pilot'
seed, updates = 2026104700, 500
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
pose.pose.requires_grad_(True)
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(coarse_path, map_location='cpu', weights_only=True)['head'])
coarse.requires_grad_(False)
fine = PoseFineMatch046().cuda().eval()
fine.load_state_dict(torch.load(fine_path, map_location='cpu', weights_only=True)['model'])
fine.requires_grad_(False)
optimizer = torch.optim.AdamW(pose.pose.parameters(), lr=1e-5, weight_decay=1e-4)

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': updates, 'sections_per_batch': 1,
          'beam': 4, 'queries_per_candidate': 8, 'shortlist_keys': 16,
          'atlas_render_and_map_side': 64,
          'trainable': '019 pose head only; 019 mapper, 041, and 046 frozen with differentiable forwards',
          'training': 'fresh arbitrary-plane synthetic TRAIN sections; no appearance pairing',
          'checkpoint_lineage': 'randomly initialized 019/041/025/045/046; no external model weights',
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'coarse_sha256': hashlib.sha256(coarse_path.read_bytes()).hexdigest(),
          'fine_sha256': hashlib.sha256(fine_path.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_joint_pose_fit_047.py', 'joint_pose_fit_047.py')},
          'synthetic_provenance': context['provenance'], 'calibrated': False,
          'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.save({'model': pose.state_dict(), 'optimizer': optimizer.state_dict(),
            'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'step': 0, 'config': config, 'calibrated': False},
           run / 'joint_step_00000.pt')
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        while True:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=256)
            draw_seed += 1
            used = bool(sampled['eligible'][0] and sampled['valid_mask'][0].any())
            draws.write(json.dumps({**sampled['provenance'][0], 'batch': step,
                                    'used': used}) + '\n')
            if used:
                break
        prediction = pose.predict(sampled['inputs'])
        target = sampled['centre']
        index = torch.multinomial(sampled['valid_mask'].flatten(1).float(), 128,
                                  replacement=True)
        truth = target.reshape(1, -1, 3).gather(1, index[..., None].expand(-1, -1, 3))
        xy = torch.stack((index.remainder(256), index.div(256, rounding_mode='floor')),
                         -1).float() / 256
        state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflected = torch.arange(2, device='cuda')[None, None].expand(1, 16, -1)
        centre, frame, basis = full_frame_state_to_components(state)
        chart = xy[:, None, None].expand(-1, 16, 2, -1, -1).clone()
        chart[..., 0] = torch.where(reflected[..., None].bool(),
                                    255 / 256 - chart[..., 0], chart[..., 0])
        rigid = centre[..., None, :] + torch.einsum('bkrij,bkrqj->bkrqi',
            frame[..., :, :2] @ basis, chart - .5)
        distance = (rigid - truth[:, None, None]).norm(dim=-1).mean(-1)
        true_normal = full_frame_state_to_components(sampled['state'])[1][..., 2]
        normal_penalty = 4000 * (1 - (frame[..., :, 2] * true_normal[:, None, None])
                                  .sum(-1).abs().clamp_max(1))
        distance = (distance + normal_penalty).flatten(1)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        target_mass = (-distance.detach() / 1000).softmax(-1)
        direct = (distance.min(-1).values / 1000
                  - .75 * torch.logsumexp(prior - distance / 1500, -1)
                  + .75 * F.kl_div(prior, target_mass, reduction='none').sum(-1)).mean()
        fitted = joint_pose_fit(pose, coarse, fine, prediction, sampled['inputs'],
            context['atlas'], sampled['offsets'], sampled['weights'])
        grid = (torch.stack((index.remainder(256),
                             index.div(256, rounding_mode='floor')), -1).float() + .5) * (2 / 256) - 1
        grid = grid[:, None].expand(-1, 4, -1, -1).reshape(4, 1, 128, 2)
        surface = fitted['mapped']['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1)
        mapped_points = F.grid_sample(surface.permute(0, 3, 1, 2), grid,
            mode='bilinear', padding_mode='border', align_corners=False)
        mapped_points = mapped_points.reshape(1, 4, 3, 128).permute(0, 1, 3, 2)
        mapped_error = (mapped_points - truth[:, None]).norm(dim=-1).mean(-1)
        posterior = fitted['score'].softmax(-1)
        rank = F.kl_div(F.log_softmax(fitted['score'], -1),
                        (-mapped_error.detach() / 800).softmax(-1),
                        reduction='none').sum(-1).mean()
        expected = (posterior * mapped_error).sum(-1).mean() / 1000
        local = fitted['mapped']['local_displacement_um']
        warp_penalty = local.square().mean().sqrt() / 1000
        loss = (direct + .7 * expected + .2 * mapped_error.min(-1).values.mean() / 1000
                + .35 * rank + .05 * warp_penalty)
        if step == 1:
            gradient = torch.autograd.grad((fitted['score'] - fitted['prior']).sum(),
                pose.pose[-1].weight, retain_graph=True)[0].reshape(16, 21, -1)
            fit_gradient = gradient[:, :12].norm()
            assert torch.isfinite(fit_gradient) and fit_gradient > 0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_gradient = torch.nn.utils.clip_grad_norm_(pose.pose.parameters(), 5.,
                                                         error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'synthetic_presentations': step,
               'direct_loss': float(direct.detach()), 'rank_loss': float(rank.detach()),
               'expected_mapped_um': float(expected.detach() * 1000),
               'best_four_mapped_um': float(mapped_error.min(-1).values.detach().mean()),
               'match_residual_um': float(fitted['match_residual_um'].detach().mean()),
               'gradient_norm': float(total_gradient),
               'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        if step == 1:
            row['fit_evidence_to_pose_gradient_norm'] = float(fit_gradient)
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 50 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 250 == 0:
            torch.save({'model': pose.state_dict(), 'optimizer': optimizer.state_dict(),
                        'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'step': step, 'config': config, 'calibrated': False},
                       run / f'joint_step_{step:05d}.pt')
(run / 'completed.json').write_text(json.dumps({
    'batches': updates, 'synthetic_presentations': updates,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
