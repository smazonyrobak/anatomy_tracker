"""Short matched test of map-quality gradients into the existing direct pose head."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_red_source_mix_311 import load_red_source_mix_311, sample_red_source_mix_311
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, BATCH, UPDATES = 20261015312, 192, 4, 256
RUN = ROOT / 'runs/paired_map_to_pose_312_001'
PARENT = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
PANELS = ROOT / 'runs/first_choice_exposure_309_001/panels.pt'
PROTOCOL = Path(__file__).resolve().parents[1] / 'docs/publication/PAIRED_MAP_TO_POSE_312_PROTOCOL_20261015.md'
assert Path(__file__).resolve().drive.upper() == RUN.drive.upper() == 'I:'
assert not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

context = load_red_source_mix_311('cuda')
model, parent_config = load_one_shot_checkpoint(PARENT, 'cuda')
assert model.modes == 80 and model.atlas_conditioning and model.spatial_feedback
initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
panel = torch.load(PANELS, map_location='cpu', weights_only=True)['v10']
assert len(panel) == 256
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
flags = torch.arange(2, device='cuda')[None, None]
eval_offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')
eval_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda') / 16


def evaluate():
    context['atlas'][0].copy_(context['source_intensities'][0])
    rigid, oracle, mapped_error = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(panel), BATCH):
            cases = panel[start:start + BATCH]
            inputs = torch.stack([case['inputs'] for case in cases]).cuda()
            chart = torch.stack([case['chart'] for case in cases]).cuda()
            truth = torch.stack([case['truth'] for case in cases]).cuda()
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(cases), model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            points = rigid_points_090(states, reflections, chart[:, None, None], (SIDE, SIDE))
            errors = (points - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            rigid.extend(errors.gather(1, first[:, None]).flatten().cpu().tolist())
            oracle.extend(errors.gather(1, mass.topk(16, -1).indices).min(-1).values.cpu().tolist())
            if start < 64:
                mapping = model.map(prediction, eval_offsets, first[:, None] // 2,
                                    first[:, None] % 2, (SIDE, SIDE),
                                    context['atlas'], eval_weights)
                pixels = (chart[..., 1] * SIDE).long() * SIDE + (chart[..., 0] * SIDE).long()
                sites = mapping['centre_surface_ccf_ap_dv_ml_um'][:, 0].flatten(1, 2).gather(
                    1, pixels[..., None].expand(-1, -1, 3))
                mapped_error.extend(((sites - truth).norm(dim=-1).mean(-1) / 1000).cpu().tolist())
    return {'n': len(rigid), 'first_rigid_mean_mm': float(np.mean(rigid)),
            'first_gt5mm': int(np.sum(np.asarray(rigid) > 5)),
            'top16_capture_le1p5mm': int(np.sum(np.asarray(oracle) <= 1.5)),
            'first_map_mean_mm_first64': float(np.mean(mapped_error)),
            'first_rigid_mm': rigid, 'top16_oracle_mm': oracle,
            'first_map_mm_first64': mapped_error}


def draw_batch(rng, step):
    accepted, pending = {}, list(range(BATCH))
    receipts = []
    while pending:
        virtual = [(int(rng.integers(56)) * VARIANTS + int(rng.integers(VARIANTS)))
                   for _ in pending]
        seed = int(rng.integers(2**63 - 1))
        with torch.no_grad():
            sample = sample_red_source_mix_311(context, virtual, seed, SIDE)
        for row, slot in enumerate(pending):
            identity = sample['provenance'][row]
            used = bool(sample['eligible'][row])
            receipts.append({'step': step, 'slot': slot, 'seed': seed,
                             'virtual': virtual[row], 'used': used, 'identity': identity})
            if used:
                accepted[slot] = {key: value[row:row + 1] for key, value in sample.items()
                                  if key in ('inputs', 'state', 'reflection', 'offsets',
                                             'weights', 'centre', 'valid_mask')}
        pending = [slot for slot in pending if slot not in accepted]
    batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)]) for key in accepted[0]}
    return batch, receipts


RUN.mkdir(parents=True)
with PARENT.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
config = {'seed': SEED, 'optimizer_batches_per_arm': UPDATES, 'synthetic_per_batch': BATCH,
          'parent': str(PARENT), 'parent_sha256': parent_sha,
          'parent_candidate': parent_config['candidate'], 'panel': str(PANELS),
          'panel_sha256': hashlib.file_digest(PANELS.open('rb'), 'sha256').hexdigest(),
          'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          'source_mix': context['provenance']['red_source_mix_311'],
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'evaluation_map_psf': '62.5 um fixed midpoint approximation; identical arms, frozen panel did not save PSF',
          'calibrated': False, 'public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

all_readouts, signatures = {}, None
for arm in ('detached_control', 'joint_feedback'):
    model.load_state_dict(initial)
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    map_modules = (model.warp_shared, model.warp_condition, model.warp,
                   model.atlas_encoder, model.pair)
    for module in (*pose_modules, *shared_modules, *map_modules):
        module.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': [parameter for module in pose_modules for parameter in module.parameters()], 'lr': 4e-6},
        {'params': [parameter for module in shared_modules for parameter in module.parameters()], 'lr': 1e-6},
        {'params': [parameter for module in map_modules for parameter in module.parameters()], 'lr': 1e-5},
    ], weight_decay=1e-4)
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    rng = np.random.default_rng(SEED + 1)
    sites_rng = torch.Generator(device='cuda').manual_seed(SEED + 2)
    arm_path = RUN / arm
    arm_path.mkdir()
    initial_readout = evaluate()
    started, draw_hashes = time.perf_counter(), []
    with (arm_path / 'training.jsonl').open('w', encoding='utf8') as log, \
         (arm_path / 'draws.jsonl').open('w', encoding='utf8') as draws:
        model.train()
        for step in range(1, UPDATES + 1):
            batch, receipts = draw_batch(rng, step)
            draw_hashes.append(hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest())
            for receipt in receipts:
                draws.write(json.dumps(receipt, allow_nan=False) + '\n')
            prediction = model.predict(batch['inputs'])
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(BATCH, model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            pixels = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128,
                                       replacement=True, generator=sites_rng)
            chart = torch.stack((pixels.remainder(SIDE),
                                 pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = batch['centre'].flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
            proposed = rigid_points_090(states, reflections, chart[:, None, None], (SIDE, SIDE))
            five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            true_five = rigid_points_090(batch['state'], batch['reflection'], corners, (SIDE, SIDE))
            tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            true_normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(BATCH, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            pose_loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                         + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
                         + .5 * physical[:, :2 * model.base_modes].min(-1).values
                         + (neighbourhood * near_cost).sum(-1)).mean()
            branches = torch.stack((prior.detach().argmax(-1), physical.detach().argmin(-1)), -1)
            mapping_prediction = (prediction if arm == 'joint_feedback' else
                                  {**prediction, 'state': prediction['state'].detach()})
            mapped = model.map(mapping_prediction, batch['offsets'], branches // 2,
                               branches % 2, (SIDE, SIDE), context['atlas'], batch['weights'])
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(2, 3)
            sites = surface.gather(2, pixels[:, None, :, None].expand(-1, 2, -1, 3))
            site_error = (sites - truth[:, None]).norm(dim=-1).mean(-1) / 1000
            field = mapped['local_displacement_um']
            magnitude = field.abs().mean((2, 3, 4)) / 1000
            smooth = ((field[..., 1:, :] - field[..., :-1, :]).abs().mean((2, 3, 4))
                      + (field[..., 1:] - field[..., :-1]).abs().mean((2, 3, 4))) / 1000
            fit_cost = site_error + .15 * magnitude + .1 * smooth
            selected_mass = prior.gather(1, branches)
            if arm == 'detached_control':
                selected_mass = selected_mass.detach()
            fit_loss = (-1.5 * torch.logsumexp(selected_mass - fit_cost / 1.5, -1)).mean()
            mask_loss = F.binary_cross_entropy_with_logits(
                mapped['correspondence_logit'][:, :, 0],
                batch['valid_mask'][:, None].float().expand(-1, 2, -1, -1))
            loss = pose_loss + .3 * fit_loss + .03 * mask_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            row = {'step': step, 'pose_loss': float(pose_loss.detach()),
                   'fit_loss': float(fit_loss.detach()), 'map_first_mm': float(site_error[:, 0].mean().detach()),
                   'map_near_mm': float(site_error[:, 1].mean().detach()),
                   'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if step % 64 == 0:
                log.flush()
                draws.flush()
                print(json.dumps({'event': 'batch_milestone', 'arm': arm, **row}), flush=True)
    if signatures is None:
        signatures = draw_hashes
    else:
        assert draw_hashes == signatures
    final_readout = evaluate()
    all_readouts[arm] = {'0': initial_readout, str(UPDATES): final_readout}
    (arm_path / 'readout.json').write_text(json.dumps(all_readouts[arm], indent=2), encoding='utf8')
    torch.save({'model': model.state_dict(), 'step': UPDATES, 'arm': arm,
                'config': config, 'calibrated': False}, arm_path / 'pose_local_00256.pt')
    print(json.dumps({'event': 'readout', 'arm': arm,
                      'step': UPDATES, 'metrics': {key: value for key, value in final_readout.items()
                                                 if not isinstance(value, list)}}), flush=True)

summary = {'arms': {arm: {stage: {key: value for key, value in readout.items()
                                if not isinstance(value, list)} for stage, readout in stages.items()}
                    for arm, stages in all_readouts.items()},
           'draws_identical': True, 'calibrated': False, 'public_benchmark_used': False,
           'preflight_only': True}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
