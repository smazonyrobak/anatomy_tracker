"""Matched 249 draw stream with only image-to-antipodal-normal supervision."""
import hashlib
import json
import math
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/pose_ce_only_250_train_001'
DOC = Path('docs/publication/POSE_CE_ONLY_250_PROTOCOL_20261012.md')
SEED, BATCHES, BATCH, SIDE = 2026101249, 3000, 4, 192
SAVES = (0, 500, 1500, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=True,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 100000

RUN.mkdir(parents=True, exist_ok=False)
source_names = (
    Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_stream.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_full_frame_primitives.py',
)
config = {
    'seed': SEED, 'batches': BATCHES, 'synthetic_per_batch': BATCH, 'side': SIDE,
    'saves': SAVES, 'full_model_parameters': sum(p.numel() for p in model.parameters()),
    'initialization': 'fresh random; no prior weights, features or pseudolabels',
    'training_stage': 'normal CE only; other heads remain random and are not trained',
    'loss': '15 degree antipodal 64-anchor soft normal CE',
    'matched_249': 'same model initialization and synthetic draw stream; only loss changed',
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in source_names},
    'real_labels_used': False, 'calibrated': False, 'public_benchmark_used': False,
}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'step': 0, 'config': config, 'calibrated': False}, RUN / 'joint_step_00000.pt')

started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, BATCHES + 1):
        accepted, pending = {}, list(range(BATCH))
        while pending:
            subjects = torch.randint(len(context['subjects']), (len(pending),),
                                     generator=subjects_rng).tolist()
            sampled = sample_one_shot_slide_artifacts_v7(context, subjects, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'batch': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in ('inputs', 'state')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)])
                 for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
        cosine = (normal @ model.normal_anchor_frames[:, :, 2].T).abs().clamp_max(1)
        angle = torch.acos(cosine.clamp_max(1 - 1e-7))
        target = F.softmax(-.5 * (angle / math.radians(15)) ** 2, -1)
        loss = -(target * prediction['log_mass']).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        selected = prediction['log_mass'].argmax(-1)
        record = {
            'batch': step, 'synthetic_presentations': step * BATCH,
            'normal_ce': float(loss.detach()),
            'normal_target_entropy': float(-(target * target.log()).sum(-1).mean()),
            'selected_normal_axis_deg': float(angle.gather(1, selected[:, None]).mean() * 180 / math.pi),
            'gradient': float(gradient), 'clipped': bool(gradient > 5),
            'seconds': time.perf_counter() - started,
        }
        log.write(json.dumps(record) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush(); draws.flush()
            print(json.dumps(record), flush=True)
        if step in SAVES:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'step': step, 'config': config, 'calibrated': False},
                       RUN / f'joint_step_{step:05d}.pt')

(RUN / 'completed.json').write_text(json.dumps({
    'batches': BATCHES, 'synthetic_presentations': BATCHES * BATCH,
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256((RUN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
                          for step in SAVES},
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'batches': BATCHES,
                  'seconds': time.perf_counter() - started}), flush=True)
