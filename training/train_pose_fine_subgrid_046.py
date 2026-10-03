"""Train spatial patch ranking and continuous sub-grid 3D correction."""
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
from training.atlas_oriented_patch_025 import extract_patches
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046
from training.pose_fine_patch_045 import atlas_key_patches

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
fine_parent = root / 'runs/pose_fine_patch_045_pilot/patch_step_00500.pt'
run = root / 'runs/pose_fine_subgrid_046_pilot'
seed, updates, sections, beam, topk = 2026104600, 2000, 2, 8, 32
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
fine = PoseFineMatch046().cuda().train()
fine.patch.load_state_dict(torch.load(fine_parent, map_location='cpu', weights_only=True)['model'])
optimizer = torch.optim.AdamW([{'params': fine.patch.parameters(), 'lr': 3e-5},
                               {'params': fine.pair.parameters(), 'lr': 1e-4}],
                              weight_decay=1e-4)

run.mkdir(parents=True, exist_ok=False)
sources = ('train_pose_fine_subgrid_046.py', 'pose_fine_match_046.py',
           'pose_fine_patch_045.py', 'atlas_oriented_patch_025.py',
           'arbitrary_plane_one_shot_stream.py')
config = {'seed': seed, 'updates': updates, 'sections_per_batch': sections,
          'query_lists_per_section': 3, 'coarse_candidates': beam, 'shortlist_keys': topk,
          'patch_side': 64, 'offset_bound_um': 1500,
          'positive_distance_scale_um': 500, 'positive_list_limit_um': 1500,
          'candidate_training': 'prior top-one, physically best of eight, random of eight; truth only in TRAIN',
          'train_sections': 'fresh independent arbitrary-plane synthetic TRAIN sections',
          'scratch_lineage': '019, 041, 025, 045; new pair head random; no external weights',
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'coarse_sha256': hashlib.sha256(coarse_path.read_bytes()).hexdigest(),
          'fine_parent_sha256': hashlib.sha256(fine_parent.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in sources},
          'synthetic_provenance': context['provenance'],
          'real_training_images': 0, 'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.save({'model': fine.state_dict(), 'optimizer': optimizer.state_dict(),
            'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'step': 0, 'config': config, 'calibrated': False},
           run / 'joint_step_00000.pt')
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(sections))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=256)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row] and
                    sampled['valid_mask'][row, 8::16, 8::16].any())
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot,
                                        'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'centre', 'valid_mask', 'offsets', 'weights')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(sections)])
                 for key in accepted[0]}
        with torch.no_grad():
            prediction = pose.predict(batch['inputs'])
            prior_score = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior_score.topk(beam, -1).indices
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            coarse_result = coarse(prediction, batch['inputs'], state, reflection,
                                   context['atlas'], batch['offsets'], batch['weights'])
            centre, frame, basis = full_frame_state_to_components(state)
            edges = frame[..., :, :2] @ basis
            yx = torch.stack(torch.meshgrid(torch.arange(8, 256, 16, device='cuda'),
                                            torch.arange(8, 256, 16, device='cuda'),
                                            indexing='ij'), -1).reshape(256, 2)
            valid_grid = batch['valid_mask'][:, 8::16, 8::16].reshape(sections, 256).bool()
            truth_grid = batch['centre'][:, 8::16, 8::16].reshape(sections, 256, 3)
            query_rows, query_yx, patches, distances, supports = [], [], [], [], []
            key_world, frames, relative, coarse_scores, local_targets = [], [], [], [], []
            for row in range(sections):
                ids = batch['valid_mask'][row].flatten().nonzero().flatten()
                chart = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')),
                                     -1).float() / 256
                chart = chart[None].expand(beam, -1, -1).clone()
                chart[..., 0] = torch.where(reflection[row, :, None].bool(),
                                             255 / 256 - chart[..., 0], chart[..., 0])
                plane = centre[row, :, None] + torch.einsum('kij,kqj->kqi',
                    edges[row], chart - .5)
                physical = (plane - batch['centre'][row].reshape(-1, 3)[ids][None]
                            ).norm(dim=-1).mean(-1)
                candidates = (0, int(physical.argmin()),
                              int(torch.randint(beam, (1,), device='cuda')))
                options = valid_grid[row].nonzero().flatten()
                selected = options[torch.randint(len(options), (3,), device='cuda')]
                for candidate, query in zip(candidates, selected):
                    top = coarse_result['logits'][row, candidate, query].topk(topk)
                    world = coarse_result['key_world'][row, candidate, top.indices]
                    support = coarse_result['key_support'][row, candidate, top.indices]
                    patch = atlas_key_patches(context['atlas'], world,
                        state[row, candidate][None].expand(topk, -1),
                        reflection[row, candidate][None].expand(topk),
                        batch['offsets'][row][None].expand(topk, -1),
                        batch['weights'][row][None].expand(topk, -1))
                    rotation = frame[row, candidate]
                    patches.append(patch)
                    distances.append((world - truth_grid[row, query]).norm(dim=-1))
                    supports.append(support)
                    key_world.append(world)
                    frames.append(rotation)
                    relative.append((world - coarse_result['query_base'][row, candidate, query]) @ rotation)
                    coarse_scores.append(top.values)
                    local_targets.append((truth_grid[row, query] - world) @ rotation)
                    query_rows.append(row)
                    query_yx.append(yx[query])
            query_patch = extract_patches(batch['inputs'],
                torch.as_tensor(query_rows, device='cuda'), torch.stack(query_yx).float())
            patches = torch.cat(patches)
            distances = torch.stack(distances)
            supports = torch.stack(supports) > .1
            world = torch.stack(key_world)
            rotations = torch.stack(frames)
            relative = torch.stack(relative)
            coarse_scores = torch.stack(coarse_scores)
            local_targets = torch.stack(local_targets)
        scores, offset = fine(query_patch, patches, relative, coarse_scores)
        masked_scores = scores.masked_fill(~supports, -1e4)
        proximity = torch.exp(-.5 * (distances / 500).square()) * supports
        positive = distances.masked_fill(~supports, float('inf')).amin(-1) <= 1500
        if positive.any():
            target = proximity[positive] / proximity[positive].sum(-1, keepdim=True)
            rank_loss = -(target * F.log_softmax(masked_scores[positive], -1)).sum(-1).mean()
            nearby = (distances <= 1500) & supports
            offset_error = F.smooth_l1_loss(offset[positive] / 1000,
                local_targets[positive] / 1000, beta=.25, reduction='none').sum(-1)
            offset_weight = proximity[positive] * nearby[positive]
            offset_loss = ((offset_error * offset_weight).sum(-1) /
                           offset_weight.sum(-1).clamp_min(1e-9)).mean()
        else:
            rank_loss = scores.sum() * 0
            offset_loss = scores.sum() * 0
        if (~positive).any() and supports[~positive].any():
            negative_loss = F.softplus(scores[~positive][supports[~positive]] - 3).mean()
        else:
            negative_loss = scores.sum() * 0
        loss = rank_loss + .5 * offset_loss + .25 * negative_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(fine.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        index = masked_scores.detach().argmax(-1)
        chosen_world = world.gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
        chosen_offset = offset.detach().gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
        corrected = chosen_world + torch.einsum('lij,lj->li', rotations, chosen_offset)
        chosen_target = (world + torch.einsum('lij,lkj->lki', rotations, local_targets))[:, 0]
        corrected_error = (corrected - chosen_target).norm(dim=-1)
        chosen_support = supports.gather(1, index[:, None])[:, 0]
        corrected_error = torch.where(chosen_support, corrected_error,
                                      corrected_error.new_full(corrected_error.shape, 1e9))
        row = {'step': step, 'synthetic_presentations': step * sections,
               'query_lists': len(query_rows), 'positive_lists': int(positive.sum()),
               'loss': float(loss.detach()), 'rank_loss': float(rank_loss.detach()),
               'offset_loss': float(offset_loss.detach()),
               'negative_loss': float(negative_loss.detach()),
               'corrected_recall_500um': float((corrected_error[positive] <= 500).float().mean())
                    if positive.any() else 0.,
               'corrected_recall_1500um': float((corrected_error[positive] <= 1500).float().mean())
                    if positive.any() else 0.,
               'gradient_norm': float(gradient),
               'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 500 == 0:
            torch.save({'model': fine.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
                run / f'joint_step_{step:05d}.pt')
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'synthetic_presentations': updates * sections,
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
