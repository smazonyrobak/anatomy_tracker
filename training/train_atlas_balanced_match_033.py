"""Continue the same selector with conditional matching and differentiable pose loss."""
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

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches
from training.atlas_learned_match_032 import AtlasLearnedMatch032

bank_dir = root / 'runs/atlas_global_patch_030'
descriptor_checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
parent_path = root / 'runs/atlas_learned_match_032/match_step_03000.pt'
run = root / 'runs/atlas_balanced_match_033'
start, extra, sections, queries, side = 3000, 6000, 2, 32, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert json.loads((parent_path.parent / 'completed.json').read_text())['updates'] == start
with (bank_dir / 'completed.json').open() as stream:
    bank_completion = json.load(stream)
for name, expected in bank_completion['files_sha256'].items():
    with (bank_dir / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
axis_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
axis_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
context = load_streaming_synthetic_v7_64(device='cuda')
saved_descriptor = torch.load(descriptor_checkpoint, map_location='cpu', weights_only=True)
descriptor = AtlasOrientedPatch025().cuda().eval()
descriptor.load_state_dict(saved_descriptor['model'], strict=True)
del saved_descriptor
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == start and not parent['calibrated']
model = AtlasLearnedMatch032().cuda().train()
model.load_state_dict(parent['model'], strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
optimizer.load_state_dict(parent['optimizer'])
for group in optimizer.param_groups:
    group['lr'] = 1e-4
subjects_rng = torch.Generator()
subjects_rng.set_state(parent['subjects_rng'])
draw_seed = parent['draw_seed']
torch.set_rng_state(parent['torch_rng'])
torch.cuda.set_rng_state_all(parent['cuda_rng'])

run.mkdir(parents=True, exist_ok=False)
config = {'parent_checkpoint_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
          'atlas_bank_completed_sha256': hashlib.sha256((bank_dir / 'completed.json').read_bytes()).hexdigest(),
          'descriptor_checkpoint_sha256': hashlib.sha256(descriptor_checkpoint.read_bytes()).hexdigest(),
          'start_batch': start, 'additional_batches': extra, 'sections_per_batch': sections,
          'queries_per_section': queries, 'candidate_matches_per_query': 16,
          'draw_seed_start': draw_seed, 'positive_distance_um': 500,
          'loss': 'conditional 16-way CE + 5x-positive availability BCE + physical Huber plane loss',
          'optimizer': 'continued AdamW state, lr changed to 1e-4', 'gradient_clip': 5.,
          'synthetic_provenance': context['provenance'],
          'training_queries': 'synthetic GT-valid tissue pixel selection, curriculum stage only',
          'training_sections': 'fresh independent random synthetic TRAIN sections, one appearance each',
          'external_pretrained_weights_features_pseudolabels': False,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_atlas_balanced_match_033.py', 'atlas_learned_match_032.py',
                                         'atlas_oriented_patch_025.py', 'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'match_step_{step:05d}.pt')


save(start)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(start + 1, start + extra + 1):
        accepted, pending = {}, list(range(sections))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1]
                                      for key in ('inputs', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(sections)])
                 for key in accepted[0]}
        selected = []
        for row in range(sections):
            available = batch['valid_mask'][row].flatten().nonzero().flatten()
            chosen = available[torch.randperm(len(available), device='cuda')[:queries]]
            selected.append(torch.stack((chosen // side, chosen % side), -1))
        pixel = torch.stack(selected)
        row = torch.arange(sections, device='cuda')[:, None].expand(-1, queries)
        with torch.no_grad():
            patch = extract_patches(batch['inputs'], row.flatten(), pixel.reshape(-1, 2).float())
            patch = torch.stack((patch, patch.flip(-1)), 1).flatten(0, 1)
            query_feature = F.normalize(descriptor.shared(descriptor.image_stem(patch)), dim=-1)
            similarity = query_feature.half() @ bank.T
            score, flat = similarity.reshape(sections, queries, 2 * len(bank)).topk(16, -1)
            parity, index = flat // len(bank), flat % len(bank)
            query_feature = query_feature.reshape(sections, queries, 2, 128)
            image_feature = torch.gather(query_feature, 2, parity[..., None].expand(-1, -1, -1, 128))
            atlas_feature = bank[index].float()
            location = positions[index // 512]
            sign = 1 - 2 * parity
            u = axis_u[index % 512] * sign[..., None]
            v = axis_v[index % 512]
            rank = torch.arange(16, device='cuda')[None, None].expand(sections, queries, -1) / 15
            truth = batch['centre'][row, pixel[..., 0], pixel[..., 1]]
            distance = (location - truth[:, :, None]).norm(dim=-1)
            minimum, nearest = distance.min(-1)
            available = minimum < 500
        logits = model(image_feature, atlas_feature, score.float(), pixel.float(), location, u, v, rank)
        conditional = F.cross_entropy(logits[..., :16].reshape(-1, 16),
                                      nearest.flatten(), reduction='none').reshape(sections, queries)
        match_loss = (conditional * available).sum() / available.sum().clamp_min(1)
        availability_logit = torch.logsumexp(logits[..., :16], -1) - logits[..., 16]
        availability_loss = F.binary_cross_entropy_with_logits(
            availability_logit, available.float(), pos_weight=logits.new_tensor(5.))
        coefficients = model.fit_plane(logits, pixel.float(), location)
        xy = pixel.flip(-1).float() / side - .5
        mapped = (coefficients[:, None, 0] + xy[..., 0, None] * coefficients[:, None, 1]
                  + xy[..., 1, None] * coefficients[:, None, 2])
        pose_loss = F.smooth_l1_loss((mapped - truth) / 1000, torch.zeros_like(mapped), beta=.5)
        loss = match_loss + availability_loss + pose_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        chosen = logits.detach().argmax(-1)
        chosen_distance = distance.gather(-1, chosen.clamp_max(15)[..., None]).squeeze(-1)
        training_row = {'step': step, 'synthetic_sections_total': step * sections,
                        'loss': float(loss.detach()), 'match_loss': float(match_loss.detach()),
                        'availability_loss': float(availability_loss.detach()),
                        'pose_loss': float(pose_loss.detach()),
                        'available_fraction': float(available.float().mean()),
                        'correct_available_fraction': float(((chosen < 16) & (chosen_distance < 500) & available).float().sum()
                                                            / available.sum().clamp_min(1)),
                        'false_unavailable_fraction': float(((chosen < 16) & ~available).float().sum()
                                                            / (~available).sum().clamp_min(1)),
                        'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(training_row) + '\n')
        if step == start + 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(training_row), flush=True)
        if step % 2000 == 1000:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'start_batch': start, 'additional_batches': extra, 'last_batch': start + extra,
    'accepted_new_synthetic_sections': extra * sections,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'last_batch': start + extra,
                  'seconds': time.perf_counter() - started}), flush=True)
