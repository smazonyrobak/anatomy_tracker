"""Initial random-start training of the final one-pass pose-and-warp architecture.

Each optimizer example draws a new arbitrary plane once; no stored section is
re-styled for another training example. Development sections are never trained.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/one_shot_joint_initial_001'
SYN_DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
SEED, UPDATES, TEACHER_UPDATES, BATCH, SIDE = 2026100102, 2500, 700, 4, 256
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(SEED)
RUN.mkdir(parents=True, exist_ok=False)
config = {'seed': SEED, 'updates': UPDATES, 'teacher_updates': TEACHER_UPDATES,
    'batch': BATCH, 'resolution': SIDE, 'training_anatomies': 64,
    'initialization': 'all learned weights randomly initialized; no earlier model checkpoint',
    'sampling': 'fresh continuous arbitrary plane, appearance, artifacts and background per draw; no deliberate repeated section',
    'teacher_stage': 'known plane trains local mapping while direct probabilistic pose learns from physical labels',
    'joint_stage': 'predicted plane and local map receive dense coordinate and differentiable atlas-fit losses',
    'development': 'one observation from each of 64 held-out synthetic planes and 64 weak-label real sections from six held-out donors',
    'calibrated': False, 'public_benchmark_used': False,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'synthetic_source': context['provenance']}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

synthetic_records = [json.loads(row) for row in (SYN_DEV / 'records.jsonl').read_text().splitlines()]
assert len(synthetic_records) == 64 and len({row['animal_id'] for row in synthetic_records}) == 4
synthetic_dev = []
for row in synthetic_records:
    with np.load(SYN_DEV / row['file']) as arrays:
        mode = row['section_index'] % 3
        synthetic_dev.append((torch.from_numpy(arrays['inputs'][mode].copy()),
                              torch.from_numpy(arrays['target_state'].copy()),
                              int(arrays['reflection']), row))
real_records = [json.loads(row) for row in (REAL / 'records.jsonl').read_text().splitlines()]
real_dev = [row for row in real_records if row['training_split'] == 'development']
assert len(real_dev) == 64 and len({row['animal_id'] for row in real_dev}) == 6
assert not {row['animal_id'] for row in real_dev} & {row['animal_id'] for row in real_records if row['training_split'] == 'train'}
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz') as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()


def points(state, reflection, side):
    centre, frame, basis = full_frame_state_to_components(state)
    uv = state.new_tensor([[0., 0.], [side - 1., 0.], [0., side - 1.],
                           [side - 1., side - 1.], [(side - 1) / 2, (side - 1) / 2]]) / side
    uv = uv.expand(*state.shape[:-1], -1, -1).clone()
    uv[..., 0] = torch.where(reflection[..., None].bool(), (side - 1) / side - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv - .5)


def dev_readout(step):
    model.eval()
    rows = []
    with torch.inference_mode():
        for label, dataset in (('synthetic', synthetic_dev), ('real_weak_allen', real_dev)):
            for row in dataset:
                if label == 'synthetic':
                    image, state, reflected, metadata = row
                    reference = points(state[None], torch.tensor([reflected]), 192)[0].cuda()
                else:
                    metadata = row
                    index = row['array_row_index']
                    image = torch.from_numpy(np.concatenate((real_images[index].astype(np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32))))
                    affine = torch.as_tensor(real_affines[index], device='cuda', dtype=torch.float32)
                    pixel = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                                          [95.5, 95.5]], device='cuda')
                    reference = affine[:, 2] + pixel[:, :1] * affine[:, 0] + pixel[:, 1:] * affine[:, 1]
                prediction = model.predict(image[None].cuda())
                candidate = points(prediction['state'][0, :, None].expand(-1, 2, -1),
                                   torch.tensor([[0, 1]], device='cuda').expand(model.modes, -1), 192)
                error = (candidate - reference).norm(dim=-1).mean(-1)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                selected = int(prior.flatten().argmax())
                rows.append({'step': step, 'set': label,
                             **{key: metadata[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                             'selected_error_um': float(error.flatten()[selected]),
                             'oracle_error_um': float(error.min())})
    with (RUN / 'development.jsonl').open('a', encoding='utf8') as output:
        output.writelines(json.dumps(row) + '\n' for row in rows)
    means = {label: float(np.mean([
        np.mean([row['selected_error_um'] for row in rows if row['set'] == label and row['animal_id'] == donor])
        for donor in {row['animal_id'] for row in rows if row['set'] == label}]))
        for label in ('synthetic', 'real_weak_allen')}
    print(json.dumps({'event': 'development', 'step': step, **means}), flush=True)
    model.train()
    return means


def save(step, name):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'config': config,
                'calibrated': False}, RUN / name)


save(0, 'initial.pt')
initial_metrics = dev_readout(0)
best = math.sqrt(initial_metrics['synthetic'] * initial_metrics['real_weak_allen'])
save(0, 'best_development.pt')
started, draw_seed = time.perf_counter(), SEED * 10000000
with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, UPDATES + 1):
        accepted = {}
        pending = list(range(BATCH))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sample = sample_one_shot_stream(context, virtual, draw_seed, side=SIDE)
            draw_seed += 1
            for row, record in enumerate(sample['provenance']):
                slot = pending[row]
                used = bool(sample['eligible'][row])
                draws.write(json.dumps({**record, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: value[row:row + 1] for key, value in sample.items()
                                      if key in ('inputs', 'state', 'reflection', 'offsets', 'weights', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)]) for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        truth, reflected = batch['state'], batch['reflection']
        target_points = points(truth, reflected, SIDE)
        candidate_points = points(prediction['state'], reflected[:, None].expand(-1, model.modes), SIDE)
        distance = (candidate_points - target_points[:, None]).norm(dim=-1).mean(-1)
        best_mode = distance.detach().argmin(-1)
        direct = distance.gather(1, best_mode[:, None]).mean() / 1000
        nll = -model.component_log_prob(prediction, truth, reflected).logsumexp(-1).mean()
        mass = F.nll_loss(prediction['log_mass'], best_mode)
        if step <= TEACHER_UPDATES:
            state = prediction['state'].scatter(1, best_mode[:, None, None].expand(-1, 1, 12), truth[:, None])
            mapping_prediction = {**prediction, 'state': state}
        else:
            mapping_prediction = prediction
        mapped = model.map(mapping_prediction, batch['offsets'], best_mode[:, None],
                           reflected[:, None], (SIDE, SIDE))
        valid = batch['valid_mask'].float()
        error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
        mapping = (error * valid).sum() / valid.sum().clamp_min(1) / 500
        field = mapped['local_displacement_um'][:, 0]
        smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                  + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
        mask = F.binary_cross_entropy_with_logits(mapped['correspondence_logit'][:, 0, 0], valid)
        fit = (model.atlas_fit_loss(batch['inputs'], mapped, context['atlas'],
                                   batch['weights'], batch['valid_mask']).mean()
               if step > TEACHER_UPDATES else direct.new_zeros(()))
        loss = direct + .1 * nll + .2 * mass + mapping + .02 * smooth + .1 * mask + .25 * fit
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        record = {'step': step, 'presentations': step * BATCH, 'direct_five_point_um': float(direct.detach() * 1000),
            'map_error_um': float(mapping.detach() * 500), 'nll': float(nll.detach()),
            'atlas_fit': float(fit.detach()), 'loss': float(loss.detach()),
            'gradient': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(record) + '\n')
        if step == 1 or step % 100 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(record), flush=True)
        if step % 500 == 0:
            save(step, f'joint_step_{step:05d}.pt')
            metrics = dev_readout(step)
            score = math.sqrt(metrics['synthetic'] * metrics['real_weak_allen'])
            if score < best:
                best = score
                save(step, 'best_development.pt')
save(UPDATES, 'latest.pt')
(RUN / 'completed.json').write_text(json.dumps({'updates': UPDATES, 'presentations': UPDATES * BATCH,
    'best_development_geometric_mean_um': best, 'seconds': time.perf_counter() - started,
    'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
