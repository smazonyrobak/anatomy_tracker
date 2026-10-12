"""Bounded TRAIN-inner whole-volume correspondence pilot (no parent weights)."""
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

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_context_match_266 import GlobalContextMatcher, atlas_input_and_masks

RUN = ROOT / 'runs/global_context_match_266_train_inner_001'
DOC = Path('docs/publication/GLOBAL_CONTEXT_MATCH_266_PROTOCOL_20261012.md')
SEED, BATCHES, BATCH_SIZE, SIDE = 2026101266, 1500, 2, 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas_input, fine_mask, coarse_mask = atlas_input_and_masks(context['atlas'])
support_input = atlas_input.clone()
support_input[0] = 0
d, h, w = atlas_input.shape[-3:]
dc, hc, wc = (d + 1) // 2, (h + 1) // 2, (w + 1) // 2
fine_indices = fine_mask.nonzero()[:, 0]
coarse_indices = coarse_mask.nonzero()[:, 0]
fine_lookup = torch.full((d * h * w,), -1, device='cuda', dtype=torch.long)
coarse_lookup = torch.full((dc * hc * wc,), -1, device='cuda', dtype=torch.long)
fine_lookup[fine_indices] = torch.arange(len(fine_indices), device='cuda')
coarse_lookup[coarse_indices] = torch.arange(len(coarse_indices), device='cuda')
token_pixels = torch.arange(16, device='cuda') * 12 + 6
token_y, token_x = torch.meshgrid(torch.arange(16, device='cuda'),
                                  torch.arange(16, device='cuda'), indexing='ij')
token_position = torch.stack((token_x.flatten(), token_y.flatten()), -1).float()


def targets(coordinates, valid):
    fine = torch.div(coordinates, 200., rounding_mode='floor').long()
    coarse = torch.div(coordinates, 400., rounding_mode='floor').long()
    inside = ((fine >= 0) & (fine < fine.new_tensor((d, h, w)))).all(-1)
    inside &= ((coarse >= 0) & (coarse < coarse.new_tensor((dc, hc, wc)))).all(-1)
    fine = fine.clamp_min(0)
    coarse = coarse.clamp_min(0)
    fine[:, :, 0].clamp_(max=d - 1)
    fine[:, :, 1].clamp_(max=h - 1)
    fine[:, :, 2].clamp_(max=w - 1)
    coarse[:, :, 0].clamp_(max=dc - 1)
    coarse[:, :, 1].clamp_(max=hc - 1)
    coarse[:, :, 2].clamp_(max=wc - 1)
    fine_label = fine_lookup[(fine[..., 0] * h + fine[..., 1]) * w + fine[..., 2]]
    coarse_label = coarse_lookup[(coarse[..., 0] * hc + coarse[..., 1]) * wc + coarse[..., 2]]
    valid = valid & inside & (fine_label >= 0) & (coarse_label >= 0)
    return valid, fine_label, coarse_label


def evaluate(model, source, panel):
    model.eval()
    with torch.inference_mode():
        atlas_fine, atlas_coarse = model.encode_atlas(source)
        atlas_fine, atlas_coarse = atlas_fine[fine_mask], atlas_coarse[coarse_mask]
        rows = []
        for start in range(0, len(panel), 4):
            block = panel[start:start + 4]
            image = torch.stack([case['image'] for case in block]).cuda()
            coords = torch.stack([case['coordinates'] for case in block]).cuda()
            visible = torch.stack([case['valid'] for case in block]).cuda()
            valid, fine_label, coarse_label = targets(coords, visible)
            query, match_logit = model.encode_image(image)
            logits = model.similarity(query, atlas_fine)
            coarse_logits = model.similarity(query, atlas_coarse)
            top = logits.topk(8, dim=-1).indices
            flat = fine_indices[top]
            matched = torch.stack((flat // (h * w), flat // w % h, flat % w), -1)
            distance = ((matched.float() + .5) * 200 - coords[:, :, None]).norm(dim=-1) / 1000
            selected = match_logit.sigmoid() > .5
            for b, case in enumerate(block):
                truth = valid[b]
                predicted = selected[b]
                within = distance[b] < .5
                good = (truth & predicted & within.any(-1)).nonzero()[:, 0]
                position = token_position[good]
                separation = (position[:, None] - position[None]).norm(dim=-1) >= 3
                triple = any(bool((separation[j] & separation[k]).any())
                             for j in range(len(good)) for k in range(j + 1, len(good))
                             if bool(separation[j, k]))
                rows.append({'base': case['base'], 'mode': case['mode'],
                    'angle_deg': case['angle_deg'], 'valid_sites': int(truth.sum()),
                    'selected_sites': int(predicted.sum()),
                    'matchable_tp': int((truth & predicted).sum()),
                    'top1_hits': int((truth & within[:, 0]).sum()),
                    'top8_hits': int((truth & within.any(-1)).sum()),
                    'auto_top8_hits': len(good), 'three_separated': bool(triple),
                    'fine_ce': float(F.cross_entropy(logits[b, truth], fine_label[b, truth]))
                               if bool(truth.any()) else None,
                    'coarse_ce': float(F.cross_entropy(coarse_logits[b, truth],
                                                       coarse_label[b, truth]))
                                 if bool(truth.any()) else None})
        return rows


RUN.mkdir(parents=True, exist_ok=False)
sources = ('global_context_match_266.py', 'train_global_context_match_266.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py')
config = {'seed': SEED, 'batches': BATCHES, 'batch_size': BATCH_SIZE,
    'unique_train_presentations': BATCHES * BATCH_SIZE,
    'train_bases': list(range(56)), 'inner_bases': list(range(56, 64)),
    'inner_sections_per_base': 8, 'atlas_grid_um': 200,
    'atlas_codebook_valid_fine': len(fine_indices),
    'atlas_codebook_valid_coarse': len(coarse_indices),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in sources},
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'synthetic_provenance': context['provenance'],
    'prior_weights_or_features': False, 'calibrated': False,
    'biological_validation': False, 'public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
panel = []
inner_records = []
inner_rng = torch.Generator().manual_seed(SEED + 1)
inner_seed = SEED * 10000000 + 9000000
for base in range(56, 64):
    accepted = 0
    while accepted < 8:
        subject = base * VARIANTS + int(torch.randint(VARIANTS, (1,), generator=inner_rng))
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v7(context, [subject], inner_seed, side=SIDE)
        inner_seed += 1
        if not bool(sample['eligible'][0]):
            continue
        coords = sample['centre'][0, token_pixels[:, None], token_pixels[None, :]].reshape(256, 3)
        valid = sample['valid_mask'][0, token_pixels[:, None], token_pixels[None, :]].reshape(256)
        true_frame = full_frame_state_to_components(sample['state'])[1]
        angle = float(torch.rad2deg(torch.acos(true_frame[0, :, 2].abs().max().clamp_max(1))))
        provenance = sample['provenance'][0]
        filename = f'inner_base_{base:02d}_section_{accepted:02d}.npz'
        np.savez_compressed(RUN / filename, image=sample['inputs'][0].cpu().numpy(),
                            coordinates=coords.cpu().numpy(), valid=valid.cpu().numpy())
        panel.append({'base': base, 'mode': provenance['mode'], 'angle_deg': angle,
            'image': sample['inputs'][0].cpu(), 'coordinates': coords.cpu(), 'valid': valid.cpu()})
        inner_records.append({'file': filename,
            'sha256': hashlib.sha256((RUN / filename).read_bytes()).hexdigest(),
            'base': base, 'subject': subject, 'angle_deg': angle,
            'physical_section_id': provenance['physical_section_id'],
            'animal_id': provenance['base_lineage']['animal_id'],
            'specimen_id': provenance['base_lineage']['specimen_id'],
            'experiment_id': provenance['base_lineage']['experiment_id'],
            'mode': provenance['mode'], 'seed_prefix': provenance['seed_prefix']})
        accepted += 1
(RUN / 'inner_records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in inner_records),
                                          encoding='utf8')
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
full = GlobalContextMatcher().cuda()
support = GlobalContextMatcher().cuda()
support.load_state_dict(full.state_dict())
models = {'full': full, 'support': support}
optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
              for arm, model in models.items()}
subject_rng = torch.Generator().manual_seed(SEED + 2)
train_seed = SEED * 10000000
started = time.perf_counter()

with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
     (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'inner_readout.jsonl').open('w', encoding='utf8') as readout:
    for arm, model in models.items():
        for row in evaluate(model, atlas_input if arm == 'full' else support_input, panel):
            readout.write(json.dumps({'batch': 0, 'arm': arm, **row}) + '\n')
    readout.flush()
    for batch in range(1, BATCHES + 1):
        samples = []
        while len(samples) < BATCH_SIZE:
            subjects = [int(torch.randint(56 * VARIANTS, (1,), generator=subject_rng))
                        for _ in range(BATCH_SIZE - len(samples))]
            with torch.no_grad():
                attempt = sample_one_shot_slide_artifacts_v7(context, subjects, train_seed, side=SIDE)
            train_seed += 1
            for row, subject in enumerate(subjects):
                provenance = attempt['provenance'][row]
                eligible = bool(attempt['eligible'][row])
                draws.write(json.dumps({'batch': batch, 'subject': subject, 'used': eligible,
                    'physical_section_id': provenance['physical_section_id'],
                    'base_lineage': provenance['base_lineage'],
                    'seed_prefix': provenance['seed_prefix'], 'mode': provenance['mode']}) + '\n')
                if eligible and len(samples) < BATCH_SIZE:
                    samples.append((attempt['inputs'][row], attempt['centre'][row],
                                    attempt['valid_mask'][row]))
        image = torch.stack([sample[0] for sample in samples])
        coordinates = torch.stack([sample[1][token_pixels[:, None],
                                                   token_pixels[None, :]].reshape(256, 3)
                                   for sample in samples])
        visible = torch.stack([sample[2][token_pixels[:, None],
                                               token_pixels[None, :]].reshape(256)
                               for sample in samples])
        valid, fine_label, coarse_label = targets(coordinates, visible)
        losses = {}
        for arm, model in models.items():
            model.train()
            source = atlas_input if arm == 'full' else support_input
            query, match_logit = model.encode_image(image)
            atlas_fine, atlas_coarse = model.encode_atlas(source)
            fine_logits = model.similarity(query[valid], atlas_fine[fine_mask])
            coarse_logits = model.similarity(query[valid], atlas_coarse[coarse_mask])
            fine_ce = (F.cross_entropy(fine_logits, fine_label[valid])
                       if bool(valid.any()) else match_logit.new_zeros(()))
            coarse_ce = (F.cross_entropy(coarse_logits, coarse_label[valid])
                         if bool(valid.any()) else match_logit.new_zeros(()))
            mask_bce = F.binary_cross_entropy_with_logits(match_logit, valid.float(),
                pos_weight=match_logit.new_tensor(3.))
            loss = fine_ce + .5 * coarse_ce + .2 * mask_bce
            optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.,
                                                       error_if_nonfinite=True)
            optimizers[arm].step()
            losses[arm] = {'fine_ce': float(fine_ce.detach()),
                'coarse_ce': float(coarse_ce.detach()), 'mask_bce': float(mask_bce.detach()),
                'gradient': float(gradient)}
        if batch == 1 or batch % 100 == 0:
            row = {'batch': batch, 'presentations': batch * BATCH_SIZE,
                   'valid_token_sites': int(valid.sum()), 'full': losses['full'],
                   'support': losses['support'], 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            log.flush()
            draws.flush()
            if batch == 1 or batch % 500 == 0:
                print(json.dumps(row), flush=True)
        if batch in (500, 1500):
            for arm, model in models.items():
                for row in evaluate(model, atlas_input if arm == 'full' else support_input, panel):
                    readout.write(json.dumps({'batch': batch, 'arm': arm, **row}) + '\n')
            readout.flush()
            torch.save({'batch': batch, 'full': full.state_dict(),
                'support': support.state_dict(),
                'full_optimizer': optimizers['full'].state_dict(),
                'support_optimizer': optimizers['support'].state_dict(),
                'subject_rng': subject_rng.get_state(), 'train_seed': train_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'calibrated': False}, RUN / f'match_step_{batch:05d}.pt')

(RUN / 'completed.json').write_text(json.dumps({
    'batches': BATCHES, 'presentations': BATCHES * BATCH_SIZE,
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'inner_records_sha256': hashlib.sha256((RUN / 'inner_records.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'inner_readout_sha256': hashlib.sha256((RUN / 'inner_readout.jsonl').read_bytes()).hexdigest(),
    'checkpoints_sha256': {str(step): hashlib.sha256((RUN / f'match_step_{step:05d}.pt').read_bytes()).hexdigest()
                           for step in (500, 1500)},
    'calibrated': False, 'biological_validation': False,
    'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'batches': BATCHES,
                  'seconds': time.perf_counter() - started}), flush=True)
