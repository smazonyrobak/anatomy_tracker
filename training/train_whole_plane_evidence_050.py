"""Train whole-slice atlas evidence against actual wrong pose proposals."""
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

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.whole_plane_evidence_050 import WholePlaneEvidence050

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
patch_parent = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
run = root / 'runs/whole_plane_evidence_050_pilot'
seed, batches, beam = 2026105000, 5000, 8
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
model = WholePlaneEvidence050().cuda().train()
model.patch.load_state_dict(torch.load(patch_parent, map_location='cpu', weights_only=True)['model'])
optimizer = torch.optim.AdamW(({'params': model.patch.parameters(), 'lr': 1e-5},
                               {'params': [p for n, p in model.named_parameters()
                                           if not n.startswith('patch.')], 'lr': 2e-4}), weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'sections_per_batch': 1, 'inference_beam': beam,
          'training_candidates': 'actual 019 top-eight, exact synthetic teacher, one perturbed teacher, one distant teacher',
          'objective': 'shared whole-plane synthetic physical ranking, teacher contrast, support-matched wrong-plane margin',
          'lineage': 'scratch 025 atlas patch continuation; scratch 019 frozen pose proposal; new spatial whole-plane head',
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'patch_parent_sha256': hashlib.sha256(patch_parent.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_whole_plane_evidence_050.py', 'whole_plane_evidence_050.py')},
          'synthetic_provenance': context['provenance'],
          'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        while True:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=256)
            draw_seed += 1
            used = bool(sampled['eligible'][0] and sampled['valid_mask'][0].any())
            draws.write(json.dumps({**sampled['provenance'][0], 'batch': step, 'used': used}) + '\n')
            if used:
                break
        with torch.no_grad():
            prediction = pose.predict(sampled['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            proposed = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            near_scale = sampled['state'].new_tensor((.07, .07, .07, 450, 450, 450, .03, .03, .04))
            far_scale = sampled['state'].new_tensor((.25, .25, .25, 2500, 2500, 2500, .12, .12, .10))
            shifted = compose_full_frame_state(sampled['state'][:, None].expand(-1, 2, -1),
                torch.randn(1, 2, 9, device='cuda') * torch.stack((near_scale, far_scale)))
            candidates = torch.cat((proposed, sampled['state'][:, None], shifted), 1)
            reflection = torch.cat((choice % 2, sampled['reflection'][:, None].expand(-1, 3)), 1)
            index = torch.multinomial(sampled['valid_mask'].flatten(1).float(), 128, replacement=True)
            target = sampled['centre'].reshape(1, -1, 3).gather(
                1, index[..., None].expand(-1, -1, 3))
            chart = torch.stack(((index.remainder(256).float() + .5) / 256,
                                 (index.div(256, rounding_mode='floor').float() + .5) / 256), -1)
            centre, frame, basis = full_frame_state_to_components(candidates)
            chart = chart[:, None].expand(-1, candidates.shape[1], -1, -1).clone()
            chart[..., 0] = torch.where(reflection[..., None].bool(), 1 - chart[..., 0], chart[..., 0])
            points = centre[..., None, :] + torch.einsum(
                'bkij,bkqj->bkqi', frame[..., :, :2] @ basis, chart - .5)
            error = (points - target[:, None]).norm(dim=-1).mean(-1)
        score, support = model(sampled['inputs'], candidates, reflection, context['atlas'],
                               sampled['offsets'], sampled['weights'])
        selection = prior.gather(1, choice) + score[:, :beam]
        rank = F.kl_div(F.log_softmax(selection, -1),
                        (-error[:, :beam] / 500).softmax(-1), reduction='batchmean')
        teacher = F.cross_entropy(score, torch.full((1,), beam, device='cuda', dtype=torch.long))
        hard = (error[:, :beam] > 2000) & ((support[:, :beam] - support[:, beam:beam + 1]).abs() < .10)
        hard = hard.float()
        margin = (F.softplus(1 + score[:, :beam] - score[:, beam:beam + 1]) * hard).sum() \
                 / hard.sum().clamp_min(1)
        loss = rank + .25 * teacher + .25 * margin
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        selected = selection.detach().argmax(-1)
        row = {'batch': step, 'loss': float(loss.detach()), 'rank': float(rank.detach()),
               'teacher': float(teacher.detach()), 'hard_margin': float(margin.detach()),
               'support_matched_hard': int(hard.sum()), 'selected_rigid_um': float(error[:, :beam].gather(1, selected[:, None]).mean()),
               'best_eight_rigid_um': float(error[:, :beam].min(-1).values.mean()),
               'gradient_norm': float(gradient), 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 1000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
