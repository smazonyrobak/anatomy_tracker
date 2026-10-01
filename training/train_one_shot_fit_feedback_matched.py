"""Matched one-shot continuation: geometry-soft 16-branch control versus atlas-fit pose feedback."""
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

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_joint_uncertainty import full_frame_residual
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT_RUN = ROOT / 'runs/one_shot_joint_mixed_real_003'
PARENT = PARENT_RUN / 'joint_step_20000.pt'
RUN = ROOT / 'runs/one_shot_fit_feedback_matched_001'
SYN_DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
REAL_DEV = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
SEED, UPDATES, SYNTHETIC_PER_BATCH, SIDE = 2026100301, 4000, 3, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert json.loads((PARENT_RUN / 'completed.json').read_text())['updates'] == 20000
synthetic = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
assert real['training_images'] == 263754 and len(real['donors']) == 1885
prior_rng = np.random.default_rng(2026100203)
remaining = [prior_rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
old_schedule = []
while len(old_schedule) < 20000:
    for donor in prior_rng.permutation(len(remaining)):
        if remaining[donor]:
            old_schedule.append((int(donor), int(remaining[donor].pop())))
            if len(old_schedule) == 20000:
                break
assert np.array_equal(np.asarray(old_schedule), np.load(PARENT_RUN / 'real_schedule.npy'))
rng = np.random.default_rng(SEED)
new_schedule = []
while len(new_schedule) < UPDATES:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            new_schedule.append((int(donor), int(remaining[donor].pop())))
            if len(new_schedule) == UPDATES:
                break
assert len(set(new_schedule)) == UPDATES and not set(new_schedule) & set(old_schedule)
RUN.mkdir(parents=True, exist_ok=False)
np.save(RUN / 'new_real_schedule.npy', np.asarray(new_schedule, dtype=np.int32))
config = {'seed': SEED, 'updates_per_arm': UPDATES, 'batch': [SYNTHETIC_PER_BATCH, 1],
          'arms': ['control', 'fit'], 'same_parent_and_optimizer': str(PARENT),
          'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
          'prior_real_schedule': str(PARENT_RUN / 'real_schedule.npy'),
          'real_schedule_sha256': hashlib.sha256((RUN / 'new_real_schedule.npy').read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'synthetic_provenance': synthetic['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'], 'real_prior_sections_reused': 0,
          'shared_loss': 'best physical five-point /1000 + .1 mixture NLL + .5 geometry-soft KL across 16 mode/reflection branches + correct-pose dense map and warp penalties',
          'fit_only_loss': '.25 atlas mismatch through predicted pose via detached 250um/2deg/2%-bounded truth-to-prediction geodesic, zero local warp and true synthetic valid mask',
          'development_selection': 'synthetic eligible sections only and all weak-Allen real development donors; all synthetic cases retained in raw rows',
          'scope': 'matched internal synthetic and weak-real TRAIN continuation, not calibration, independent animal validation, or deployment evidence',
          'calibrated': False, 'public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2))

landmarks = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device='cuda') / SIDE
axis = torch.arange(SIDE, device='cuda') / SIDE
y, x = torch.meshgrid(axis, axis, indexing='ij')
pixel = torch.stack((x, y), -1)


def five_points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = landmarks.expand(*state.shape[:-1], 5, 2).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


synthetic_dev = [json.loads(line) for line in (SYN_DEV / 'records.jsonl').read_text().splitlines()]
real_dev = [row for row in map(json.loads, (REAL_DEV / 'records.jsonl').read_text().splitlines())
            if row['training_split'] == 'development']
assert len(synthetic_dev) == len(real_dev) == 64
assert len({row['animal_id'] for row in synthetic_dev}) == 4
assert len({row['animal_id'] for row in real_dev}) == 6
assert not {base['lineage']['animal_id'] for base in synthetic['bases']} & {row['animal_id'] for row in synthetic_dev}
assert not {donor['animal_id'] for donor in real['donors']} & {row['animal_id'] for row in real_dev}
real_dev_images = np.load(REAL_DEV / 'images.npy', mmap_mode='r')
with np.load(REAL_DEV / 'geometry.npz') as arrays:
    real_dev_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()


def development(arm, step, model):
    model.eval()
    rows = []
    with torch.inference_mode():
        for split, records in (('synthetic', synthetic_dev), ('real_weak_allen', real_dev)):
            for record in records:
                if split == 'synthetic':
                    with np.load(SYN_DEV / record['file']) as arrays:
                        appearance = record['section_index'] % 3
                        image = torch.from_numpy(arrays['inputs'][appearance:appearance + 1].copy()).cuda()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        reflection = torch.tensor([int(arrays['reflection'])], device='cuda')
                        eligible = bool(arrays['eligible'][appearance])
                    reference = five_points(truth, reflection)[0]
                    normal = full_frame_state_to_components(truth)[1][0, :, 2]
                else:
                    eligible = None
                    index = record['array_row_index']
                    image = torch.from_numpy(np.concatenate((real_dev_images[index].astype(np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32))))[None].cuda()
                    affine = torch.as_tensor(real_dev_affines[index], device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + 192 * landmarks[:, :1] * affine[:, 0] + 192 * landmarks[:, 1:] * affine[:, 1]
                    normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                image = F.interpolate(image, (SIDE, SIDE), mode='bilinear', align_corners=False)
                prediction = model.predict(image)
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                flags = torch.tensor([[0, 1]], device='cuda').expand(model.modes, -1)
                error = (five_points(states, flags) - reference).norm(dim=-1).mean(-1)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                chosen = int(prior.flatten().argmax())
                predicted_normal = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                angle = torch.rad2deg((predicted_normal * normal).sum(-1).abs().clamp(0, 1).acos())
                rows.append({'arm': arm, 'step': step, 'set': split,
                    **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                    'synthetic_eligible': eligible,
                    'selected_five_point_um': float(error.flatten()[chosen]),
                    'oracle_five_point_um': float(error.min()),
                    'selected_normal_angle_deg': float(angle[chosen // 2])})
    with (RUN / arm / 'development.jsonl').open('a') as output:
        output.writelines(json.dumps(row) + '\n' for row in rows)
    means = {}
    for split, eligible_only in (('synthetic', True), ('synthetic_all', False), ('real_weak_allen', False)):
        subset = [row for row in rows if row['set'] == ('synthetic' if split == 'synthetic_all' else split)
                  and (not eligible_only or row['synthetic_eligible'])]
        means[split] = float(np.mean([np.mean([row['selected_five_point_um'] for row in subset
            if row['animal_id'] == donor]) for donor in {row['animal_id'] for row in subset}]))
    model.train()
    print(json.dumps({'arm': arm, 'event': 'development', 'step': step, **means}), flush=True)
    return means


matched_draws = []
for arm in ('control', 'fit'):
    directory = RUN / arm
    directory.mkdir()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    checkpoint = torch.load(PARENT, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 20000 and not checkpoint['calibrated']
    model = OneShotJointSliceModel().cuda()
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    subjects_rng = torch.Generator().manual_seed(SEED)
    draw_seed = SEED * 10000000
    started = time.perf_counter()
    initial = development(arm, 0, model)
    best = math.sqrt(initial['synthetic'] * initial['real_weak_allen'])
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': 0, 'arm': arm, 'config': config, 'calibrated': False}, directory / 'best_development.pt')
    with (directory / 'training.jsonl').open('w') as log, (directory / 'draws.jsonl').open('w') as draws:
        for step in range(1, UPDATES + 1):
            accepted, pending, signature = {}, list(range(SYNTHETIC_PER_BATCH)), []
            while pending:
                virtual = torch.randint(len(synthetic['subjects']), (len(pending),), generator=subjects_rng).tolist()
                sample = sample_one_shot_stream(synthetic, virtual, draw_seed, side=SIDE)
                draw_seed += 1
                for row, identity in enumerate(sample['provenance']):
                    slot = pending[row]
                    used = bool(sample['eligible'][row])
                    signature.append((slot, identity['physical_section_id'], used))
                    draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                    if used:
                        accepted[slot] = {key: value[row:row + 1] for key, value in sample.items()
                                          if key in ('inputs', 'state', 'reflection', 'offsets', 'weights', 'centre', 'valid_mask')}
                pending = [slot for slot in pending if slot not in accepted]
            if arm == 'control':
                matched_draws.append(signature)
            else:
                assert signature == matched_draws[step - 1]
            batch = {key: torch.cat([accepted[slot][key] for slot in range(SYNTHETIC_PER_BATCH)]) for key in accepted[0]}
            donor, section = new_schedule[step - 1]
            observation = sample_reserved_real_train(real, donor, [section], device='cuda')
            observation['inputs'] = F.interpolate(observation['inputs'], (SIDE, SIDE), mode='bilinear', align_corners=False)
            draws.write(json.dumps({**observation['identities'][0], 'step': step,
                                    'slot': SYNTHETIC_PER_BATCH, 'used': True,
                                    'label_role': real['label_role']}) + '\n')
            inputs = torch.cat((batch['inputs'], observation['inputs']))
            truth = torch.cat((batch['state'], observation['state']))
            reflection = torch.cat((batch['reflection'], observation['reflection']))
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(len(inputs), model.modes, -1)
            distance = (five_points(states, flags) - five_points(truth, reflection)[:, None, None]).norm(dim=-1).mean(-1)
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            best_branch = distance.detach().flatten(1).argmin(-1)
            direct = distance.flatten(1).gather(1, best_branch[:, None]).mean() / 1000
            nll = -model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean()
            target = (-distance.detach().flatten(1) / 2000).softmax(-1)
            rank = F.kl_div(prior.flatten(1), target, reduction='batchmean')
            correct_reflection_distance = distance[:SYNTHETIC_PER_BATCH].gather(
                2, batch['reflection'][:, None, None].expand(-1, model.modes, 1)).squeeze(-1)
            best_mode = correct_reflection_distance.detach().argmin(-1)
            mapping_prediction = {key: value[:SYNTHETIC_PER_BATCH] if torch.is_tensor(value) else value
                                  for key, value in prediction.items()}
            pose = mapping_prediction['state'].scatter(
                1, best_mode[:, None, None].expand(-1, 1, 12), batch['state'][:, None])
            mapped = model.map({**mapping_prediction, 'state': pose}, batch['offsets'],
                               best_mode[:, None], batch['reflection'][:, None], (SIDE, SIDE))
            valid = batch['valid_mask'].float()
            surface_error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
            mapping = (surface_error * valid).sum() / valid.sum().clamp_min(1) / 500
            field = mapped['local_displacement_um'][:, 0]
            magnitude = field.square().mean().sqrt() / 1000
            smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                      + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
            mask = F.binary_cross_entropy_with_logits(mapped['correspondence_logit'][:, 0, 0], valid)
            loss = direct + .1 * nll + .5 * rank + mapping + .1 * magnitude + .02 * smooth + .1 * mask
            fit, fit_gradient, bridge_error_um, bridge_fraction = None, None, None, None
            if arm == 'fit':
                selected = prediction['state'][:SYNTHETIC_PER_BATCH][
                    torch.arange(SYNTHETIC_PER_BATCH, device='cuda'), best_mode]
                residual = full_frame_residual(batch['state'], selected)
                with torch.no_grad():
                    bridge_fraction = torch.stack((torch.ones(SYNTHETIC_PER_BATCH, device='cuda'),
                        math.radians(2) / residual[:, :3].norm(dim=-1).clamp_min(1e-8),
                        250 / residual[:, 3:6].norm(dim=-1).clamp_min(1e-8),
                        math.log1p(.02) / residual[:, 6:8].abs().amax(-1).clamp_min(1e-8),
                        .02 / residual[:, 8].abs().clamp_min(1e-8)), -1).amin(-1)
                bridge = compose_full_frame_state(batch['state'], residual * bridge_fraction[:, None])
                centre, frame, basis = full_frame_state_to_components(bridge)
                chart = pixel[None].expand(SYNTHETIC_PER_BATCH, -1, -1, -1).clone()
                chart[..., 0] = torch.where(batch['reflection'][:, None, None].bool(),
                                            255 / SIDE - chart[..., 0], chart[..., 0])
                plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
                coordinates = plane[:, None] + batch['offsets'][:, :, None, None, None] * frame[:, None, None, None, :, 2]
                fit = model.atlas_fit_loss(batch['inputs'], {'coordinates': coordinates[:, None]},
                                           synthetic['atlas'], batch['weights'], batch['valid_mask']).mean()
                bridge_error_um = (five_points(bridge, batch['reflection'])
                                   - five_points(batch['state'], batch['reflection'])).norm(dim=-1).mean()
                if step == 1:
                    fit_gradient = float(torch.autograd.grad(.25 * fit, model.pose[-1].weight, retain_graph=True)[0].norm())
                loss = loss + .25 * fit
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            selected_error = distance.detach().flatten(1).gather(1, prior.detach().flatten(1).argmax(-1)[:, None])
            row = {'arm': arm, 'step': step, 'presentations': step * 4,
                'synthetic_selected_five_point_um': float(selected_error[:SYNTHETIC_PER_BATCH].mean()),
                'real_weak_selected_five_point_um': float(selected_error[-1]),
                'synthetic_oracle_five_point_um': float(distance[:SYNTHETIC_PER_BATCH].detach().flatten(1).min(-1).values.mean()),
                'direct_loss_um': float(direct.detach() * 1000), 'nll': float(nll.detach()),
                'joint_branch_rank': float(rank.detach()), 'teacher_map_error_um': float(mapping.detach() * 500),
                'fit': None if fit is None else float(fit.detach()),
                'bridge_five_point_um': None if bridge_error_um is None else float(bridge_error_um.detach()),
                'bridge_fraction_mean': None if bridge_fraction is None else float(bridge_fraction.mean()),
                'fit_to_pose_head_gradient_norm': fit_gradient if step == 1 else None,
                'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 200 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
            if step % 2000 == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                            'step': step, 'arm': arm, 'config': config, 'calibrated': False},
                           directory / f'joint_step_{step:05d}.pt')
                metrics = development(arm, step, model)
                score = math.sqrt(metrics['synthetic'] * metrics['real_weak_allen'])
                if score < best:
                    best = score
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step, 'arm': arm, 'config': config, 'calibrated': False},
                               directory / 'best_development.pt')
    (directory / 'completed.json').write_text(json.dumps({'arm': arm, 'updates': UPDATES,
        'presentations': UPDATES * 4, 'new_distinct_real_train_sections': UPDATES,
        'best_development_geometric_mean_um': best, 'seconds': time.perf_counter() - started,
        'calibrated': False, 'public_benchmark_used': False}, indent=2))
    del model, optimizer
print('Matched one-shot control and fit branches complete; frozen independent audit required', flush=True)
