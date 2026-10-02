"""Train only new spatial correspondence evidence on actual 019 top-eight planes."""
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

from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.one_shot_candidate_correspondence_021 import OneShotCorrespondenceModel021

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/one_shot_correspondence_021'
seed, updates, synthetic, side, beam = 2026102100, 2000, 2, 256, 8
draw_seed = 20261021000000000
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
model = OneShotCorrespondenceModel021().cuda().eval()
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
assert not unexpected and set(missing) == {name for name in model.state_dict() if '_021.' in name}
del checkpoint
model.requires_grad_(False)
for module in (model.image_descriptor_021, model.atlas_descriptor_021, model.match_field_021):
    module.requires_grad_(True)
optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                              lr=2e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic,
          'beam': beam, 'side': side, 'draw_seed_start': draw_seed,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'],
          'training_positive': 'true observed-plane branch; never supplied at inference',
          'candidate_negatives': 'actual parent top-eight, pixel negatives only where CCF error exceeds 700 um',
          'trainable': ['image_descriptor_021', 'atlas_descriptor_021', 'match_field_021'],
          'real_training_images': 0, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_correspondence_021.py',
                                         'one_shot_candidate_correspondence_021.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py')}}
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
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                       'centre', 'valid_mask')}
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
                        'state': torch.cat((prediction['state'].gather(
                            1, (choice // 2)[..., None].expand(-1, -1, 12)),
                            batch['state'][:, None]), 1),
                        'log_mass': torch.cat((prediction['log_mass'].gather(1, choice // 2),
                                               prior.new_full((synthetic, 1), -2.772588722)), 1),
                        'reflection_logit': torch.cat((prediction['reflection_logit'].gather(
                            1, choice // 2), prior.new_zeros(synthetic, 1)), 1)}
            reflected = torch.cat((choice % 2, batch['reflection'][:, None]), 1)
            index = torch.arange(beam + 1, device='cuda')[None].expand(synthetic, -1)
            first = model.map(selected, batch['offsets'], index, reflected,
                              (64, 64), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=64,
                              source_shape=(side, side))
            refined_state, delta, _ = model.refine(first['refinement_feature'], selected['state'])
            refined = {**selected, 'state': refined_state}
            mapped = model.map(refined, batch['offsets'], index, reflected,
                               (96, 96), context['atlas'], batch['weights'],
                               feature_side=96, source_shape=(side, side))
            old_score = model.score_fitted_candidates(image, refined, mapped,
                                                       context['atlas'], batch['weights']) + delta
            target = F.interpolate(batch['centre'].permute(0, 3, 1, 2), (96, 96),
                                   mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            valid = F.interpolate(batch['valid_mask'][:, None].float(), (96, 96), mode='area')[:, 0]
            distance = (mapped['centre_surface_ccf_ap_dv_ml_um'] - target[:, None]).norm(dim=-1)
            physical = (distance * valid[:, None]).sum((-1, -2)) / valid.sum((-1, -2))[:, None]
        score, pixel_logit, peak = model.correspondence_score_021(
            image, refined, mapped, context['atlas'], batch['weights'], old_score,
            return_fields=True)
        target_rank = F.softmax(-physical / 500, -1)
        rank = F.kl_div(F.log_softmax(score, -1), target_rank, reduction='batchmean')
        pred_rank = F.kl_div(F.log_softmax(score[:, :beam], -1),
                             F.softmax(-physical[:, :beam] / 500, -1), reduction='batchmean')
        expected = (F.softmax(score[:, :beam], -1) * physical[:, :beam]).sum(-1).mean() / 1000
        pixel_target = (-distance / 500).exp()
        pixel = (F.binary_cross_entropy_with_logits(pixel_logit, pixel_target,
                                                    reduction='none') * valid[:, None]).sum() \
                / (valid.sum() * (beam + 1))
        sampled = torch.multinomial(valid.flatten(1) + 1e-6, 64, replacement=False)
        match = peak.flatten(2).gather(2, sampled[:, None].expand(-1, beam + 1, -1))
        far = (distance[:, :beam].flatten(2).gather(2, sampled[:, None].expand(-1, beam, -1))
               > 700).float()
        hard_negative = (F.softplus(8 * (match[:, :beam] - match[:, beam:] + .1)) * far).sum() \
                        / far.sum().clamp_min(1)
        loss = .5 * rank + 2 * pred_rank + expected + .5 * pixel + .5 * hard_negative
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'selected_mapped_um': float(physical.gather(1, score.detach().argmax(-1)[:, None]).mean()),
               'selected_predicted_mapped_um': float(physical[:, :beam].gather(
                   1, score[:, :beam].detach().argmax(-1)[:, None]).mean()),
               'oracle_predicted_mapped_um': float(physical[:, :beam].min(-1).values.mean()),
               'rank_loss': float(rank.detach()), 'predicted_rank_loss': float(pred_rank.detach()),
               'pixel_loss': float(pixel.detach()), 'hard_negative_loss': float(hard_negative.detach()),
               'loss': float(loss.detach()), 'gradient_norm': float(gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
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
