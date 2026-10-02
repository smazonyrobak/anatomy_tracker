"""Isolate candidate ranking while retaining the frozen 019 plane and tissue model."""
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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/one_shot_ranker_020'
seed, updates, synthetic, side, beam = 2026102000, 6000, 2, 256, 8
draw_seed = 20261020000000000
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 18000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               vector_refinement=True, candidate_ranking=True,
                               fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
model.fitted_matcher.requires_grad_(True)
optimizer = torch.optim.AdamW(model.fitted_matcher.parameters(), lr=1e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic,
          'beam': beam, 'side': side, 'draw_seed_start': draw_seed,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'],
          'trainable': ['fitted_matcher'], 'real_training_images': 0,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_ranker_020.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'offsets', 'weights', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        image = batch['inputs']
        with torch.no_grad():
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            selected = {**prediction,
                        'state': prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12)),
                        'log_mass': prediction['log_mass'].gather(1, choice // 2),
                        'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
            index = torch.arange(beam, device='cuda')[None].expand(synthetic, -1)
            first = model.map(selected, batch['offsets'], index, choice % 2,
                              (64, 64), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=64,
                              source_shape=(side, side))
            refined_state, delta, _ = model.refine(first['refinement_feature'], selected['state'])
            refined = {**selected, 'state': refined_state}
            mapped = model.map(refined, batch['offsets'], index, choice % 2,
                               (96, 96), context['atlas'], batch['weights'],
                               feature_side=96, source_shape=(side, side))
            target = F.interpolate(batch['centre'].permute(0, 3, 1, 2), (96, 96),
                                   mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            valid = F.interpolate(batch['valid_mask'][:, None].float(), (96, 96), mode='area')[:, 0]
            distance = (mapped['centre_surface_ccf_ap_dv_ml_um'] - target[:, None]).norm(dim=-1)
            physical = (distance * valid[:, None]).sum((-1, -2)) / valid.sum((-1, -2))[:, None]
            target_rank = F.softmax(-physical / 500, -1)
        score = model.score_fitted_candidates(image, refined, mapped,
                                               context['atlas'], batch['weights']) + delta
        loss = F.kl_div(F.log_softmax(score, -1), target_rank, reduction='batchmean') \
               + .25 * (F.softmax(score, -1) * physical).sum(-1).mean() / 1000
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.fitted_matcher.parameters(),
                                                  5., error_if_nonfinite=True)
        optimizer.step()
        selected_error = physical.gather(1, score.detach().argmax(-1)[:, None])
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'selected_mapped_um': float(selected_error.mean()),
               'oracle_mapped_um': float(physical.min(-1).values.mean()),
               'top1_oracle_fraction': float((score.detach().argmax(-1) == physical.argmin(-1)).float().mean()),
               'loss': float(loss.detach()), 'gradient_norm': float(gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 2000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates * synthetic,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
