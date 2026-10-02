"""Train the final-lineage match-set component on fresh retrieved synthetic sections."""
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
checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
run = root / 'runs/atlas_learned_match_032'
seed, updates, sections, queries, side = 2026103200, 3000, 2, 32, 256
draw_seed = 20261032000000000
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

with (bank_dir / 'completed.json').open() as stream:
    bank_completion = json.load(stream)
for name, expected in bank_completion['files_sha256'].items():
    with (bank_dir / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
with (bank_dir / 'config.json').open() as stream:
    bank_config = json.load(stream)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == bank_config['checkpoint_sha256']
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
axis_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
axis_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
context = load_streaming_synthetic_v7_64(device='cuda')
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
descriptor = AtlasOrientedPatch025().cuda().eval()
descriptor.load_state_dict(saved['model'], strict=True)
del saved
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
model = AtlasLearnedMatch032().cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'updates': updates, 'sections_per_batch': sections,
          'queries_per_section': queries, 'candidate_matches_per_query': 16,
          'side': side, 'draw_seed_start': draw_seed, 'positive_distance_um': 500,
          'positive_loss_weight': 5., 'optimizer': 'AdamW 3e-4 weight_decay 1e-4',
          'atlas_bank_completed_sha256': hashlib.sha256((bank_dir / 'completed.json').read_bytes()).hexdigest(),
          'descriptor_checkpoint_sha256': bank_config['checkpoint_sha256'],
          'synthetic_provenance': context['provenance'],
          'training_queries': 'synthetic GT-valid tissue pixel selection, first curriculum stage only',
          'training_sections': 'fresh independent random synthetic TRAIN sections, one appearance each',
          'selector_random_initialization': True, 'external_pretrained_weights_features_pseudolabels': False,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_atlas_learned_match_032.py', 'atlas_learned_match_032.py',
                                         'atlas_oriented_patch_025.py', 'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'match_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
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
            target = torch.where(minimum < 500, nearest, 16)
        logits = model(image_feature, atlas_feature, score.float(), pixel.float(),
                       location, u, v, rank)
        loss_per_point = F.cross_entropy(logits.reshape(-1, 17), target.flatten(), reduction='none')
        weight = torch.where(target.flatten() == 16, 1., 5.)
        loss = (loss_per_point * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        predicted = logits.detach().argmax(-1)
        positive = target < 16
        training_row = {'step': step, 'synthetic_sections': step * sections,
                        'loss': float(loss.detach()), 'available_fraction': float(positive.float().mean()),
                        'correct_match_fraction': float(((predicted == target) & positive).float().mean()),
                        'false_match_fraction': float(((predicted < 16) & ~positive).float().mean()),
                        'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(training_row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(training_row), flush=True)
        if step % 1000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic_sections': updates * sections,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
