"""Scratch pose, source reliability, and exact-pose mapper stage for joint model 305."""
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
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, BATCH, UPDATES = 20261012305, 192, 4, 26000
READOUTS = (0, 2000, 8000, 16000, 26000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_PROTOCOL_20261012.md'
RUN = ROOT / 'runs/joint_spatial_feedback_305_pose_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
context = load_streaming_synthetic_v7_64('cuda')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = {'candidate': 305, 'stage': 'scratch_pose', 'seed': SEED, 'side': SIDE,
    'batch_size': BATCH, 'optimizer_batches': UPDATES, 'readouts': READOUTS,
    'model': {'modes': 16, 'normal_anchor_count': 64,
              'atlas_conditioning': True, 'vector_refinement': True,
              'spatial_feedback': True},
    'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'init': 'all trainable weights initialized by this run; no checkpoint loaded',
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'train_joint_spatial_pose_305.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py')},
    'no_external_weights_features_or_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

panel, attempts = [], 0
panel_rng = np.random.default_rng(SEED + 1)
with (RUN / 'panel_draws.jsonl').open('w', encoding='utf8') as output:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                attempts += 1
                variant = int(panel_rng.integers(VARIANTS))
                subject = base * VARIANTS + variant
                seed = SEED + 1_000_000_000 + attempts
                with torch.no_grad():
                    sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE)
                accepted = bool(sample['eligible'][0])
                output.write(json.dumps({'base': base, 'slot': slot, 'variant': variant,
                    'seed': seed, 'accepted': accepted, 'provenance': sample['provenance'][0]},
                    allow_nan=False) + '\n')
                if accepted:
                    break
                del sample
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero().flatten()
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(512, len(pixels)),
                device=pixels.device).long()]
            chart = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = sample['centre'][0].reshape(-1, 3)[pixels]
            normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            panel.append({'inputs': sample['inputs'][0].cpu(),
                'state': sample['state'][0].cpu(),
                'reflection': sample['reflection'][0].cpu(),
                'offsets': sample['offsets'][0].cpu(),
                'weights': sample['weights'][0].cpu(),
                'valid_mask': valid.cpu(), 'chart': chart.cpu(), 'truth': truth.cpu(),
                'base': base, 'slot': slot,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'damage_fraction': float((sample['fragment_displaced_mask'][0]
                    | sample['fragment_overlap_mask'][0] | sample['fold_occlusion_mask'][0]
                    | sample['fold_overlap_mask'][0] | sample['bubble_invalid_mask'][0]
                    ).float().mean()),
                'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
                'exterior_style': sample['provenance'][0][
                    'one_shot_slide_artifacts_v9']['exterior_style'],
                'provenance': sample['provenance'][0]})
            del sample
assert len(panel) == 64 and len({row['physical_section_id'] for row in panel}) == 64
torch.save(panel, RUN / 'inner_panel.pt')

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, vector_refinement=True, spatial_feedback=True).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
draw_rng = np.random.default_rng(SEED + 2)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
attempted = accepted = 0
started = time.perf_counter()
readouts = {}
with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
        (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
    for step in range(UPDATES + 1):
        if step:
            optimizer.param_groups[0]['lr'] = 2e-5 + 8e-5 * (1 + math.cos(math.pi * step / UPDATES)) / 2
            images, states, reflections, masks, centres, offsets, weights = [], [], [], [], [], [], []
            while len(images) < BATCH:
                attempted += 1
                base = int(draw_rng.integers(56))
                variant = int(draw_rng.integers(VARIANTS))
                subject = base * VARIANTS + variant
                seed = SEED + 2_000_000_000 + attempted
                with torch.no_grad():
                    sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE)
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'step': step, 'attempt': attempted,
                    'base': base, 'variant': variant, 'seed': seed, 'accepted': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][0].clone())
                    states.append(sample['state'][0].clone())
                    reflections.append(sample['reflection'][0].clone())
                    masks.append(sample['valid_mask'][0].clone())
                    centres.append(sample['centre'][0].clone())
                    offsets.append(sample['offsets'][0].clone())
                    weights.append(sample['weights'][0].clone())
                del sample
            inputs, states, reflections, valid, centres, axial, masses = map(torch.stack,
                (images, states, reflections, masks, centres, offsets, weights))
            pixels = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
            chart = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = centres.flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(inputs)
            proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
            proposed_reflections = reflection_options.expand(BATCH, model.modes, 2)
            prior = (prediction['log_mass'].float()[..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'].float()),
                F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
            proposed = rigid_points_090(proposed_states, proposed_reflections,
                chart[:, None, None], (SIDE, SIDE))
            five = rigid_points_090(proposed_states, proposed_reflections,
                corners, (SIDE, SIDE))
            true_five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            site_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
            true_normal = full_frame_state_to_components(states)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            physical = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(BATCH, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            direct_loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1)).mean()
            reliability_loss = model.source_reliability_loss(prediction, valid)
            loss = direct_loss + .1 * reliability_loss
            teacher_error = None
            if step % 4 == 0:
                teacher_prediction = {**prediction, 'state': states[:, None]}
                index = torch.zeros((BATCH, 1), device='cuda', dtype=torch.long)
                teacher = model.map(teacher_prediction, axial, index,
                    reflections[:, None], (96, 96), context['atlas'], masses,
                    feature_side=96, source_shape=(SIDE, SIDE), max_derivative=.35)
                surface = teacher['centre_surface_ccf_ap_dv_ml_um'][:, 0].permute(0, 3, 1, 2)
                grid = ((chart + .5 / SIDE) * 2 - 1).reshape(BATCH, 1, 128, 2)
                fitted = F.grid_sample(surface, grid, mode='bilinear',
                    padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)
                teacher_error = (fitted - truth).norm(dim=-1).mean() / 1000
                local = teacher['local_displacement_um'] / 1000
                loss = loss + 5 * teacher_error + .05 * local.square().mean()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            training.write(json.dumps({'step': step, 'attempted': attempted,
                'accepted': accepted, 'direct_loss': float(direct_loss.detach()),
                'reliability_loss': float(reliability_loss.detach()),
                'teacher_error_mm': None if teacher_error is None else float(teacher_error.detach()),
                'loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if step == 1 or step % 500 == 0:
                draws.flush(); training.flush()
                print(json.dumps({'event': 'pose_batch', 'step': step,
                    'direct_loss': float(direct_loss.detach()), 'accepted': accepted}), flush=True)
            del inputs, states, reflections, valid, centres, prediction, loss

        if step in READOUTS:
            draws.flush()
            training.flush()
            model.eval()
            rows = []
            with torch.inference_mode():
                for item in panel:
                    prediction = model.predict(item['inputs'][None].cuda())
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                    branches = prior.topk(16).indices
                    states = prediction['state'][0, branches // 2][None]
                    chart = item['chart'].cuda()
                    truth = item['truth'].cuda()
                    points = rigid_points_090(states, (branches % 2)[None],
                        chart[None], (SIDE, SIDE))[0]
                    error = (points - truth[None]).norm(dim=-1).mean(-1) / 1000
                    rows.append({'physical_section_id': item['physical_section_id'],
                        'base': item['base'], 'slot': item['slot'],
                        'obliquity_deg': item['obliquity_deg'],
                        'visible_fraction': item['visible_fraction'],
                        'damage_fraction': item['damage_fraction'],
                        'exterior_style': item['exterior_style'],
                        'branch_ids': branches.tolist(), 'rigid_site_error_mm': error.tolist(),
                        'source_reliability_loss': float(model.source_reliability_loss(
                            prediction, item['valid_mask'][None].cuda()))})
            capture = sum(min(row['rigid_site_error_mm']) <= 1.5 for row in rows)
            mean_oracle = float(np.mean([min(row['rigid_site_error_mm']) for row in rows]))
            mean_first = float(np.mean([row['rigid_site_error_mm'][0] for row in rows]))
            strata = {'obliquity_ge45': [row for row in rows if row['obliquity_deg'] >= 45],
                'visible_lt0p1': [row for row in rows if row['visible_fraction'] < .1],
                'damage_ge0p02': [row for row in rows if row['damage_fraction'] >= .02]}
            strata.update({style: [row for row in rows if row['exterior_style'] == style]
                for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
            strata_metrics = {name: {'n': len(group),
                'capture_1p5mm': sum(min(row['rigid_site_error_mm']) <= 1.5 for row in group),
                'oracle_top16_mean_error_mm': (float(np.mean([
                    min(row['rigid_site_error_mm']) for row in group])) if group else None),
                'first_choice_mean_error_mm': (float(np.mean([
                    row['rigid_site_error_mm'][0] for row in group])) if group else None)}
                for name, group in strata.items()}
            readouts[str(step)] = {'capture_1p5mm': capture,
                'oracle_top16_mean_error_mm': mean_oracle,
                'first_choice_mean_error_mm': mean_first, 'strata': strata_metrics}
            readout_file.write(json.dumps({'step': step, 'summary': readouts[str(step)],
                'rows': rows}, allow_nan=False) + '\n')
            readout_file.flush()
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'config': config, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'draw_rng_state': draw_rng.bit_generator.state,
                'calibrated': False}, RUN / f'pose_step_{step:05d}.pt')
            print(json.dumps({'event': 'pose_readout', 'step': step,
                **readouts[str(step)]}), flush=True)
            model.train()

selected = max(READOUTS, key=lambda step: (
    int(readouts[str(step)]['capture_1p5mm'] >= 45
        and readouts[str(step)]['oracle_top16_mean_error_mm'] <= 1.5),
    readouts[str(step)]['capture_1p5mm'],
    -readouts[str(step)]['oracle_top16_mean_error_mm'],
    -readouts[str(step)]['first_choice_mean_error_mm']))
summary = {'candidate': 305, 'stage': 'scratch_pose', 'selected_step': selected,
    'readouts': readouts, 'attempted': attempted, 'accepted': accepted,
    'joint_stage_capture_gate': (readouts[str(selected)]['capture_1p5mm'] >= 45
        and readouts[str(selected)]['oracle_top16_mean_error_mm'] <= 1.5),
    'calibrated': False, 'no_dev_cal_final_or_public_benchmark': True}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
receipt = {'summary': summary, 'output_sha256': {path.name: sha(path) for path in RUN.iterdir()
    if path.name != 'completed.json'}, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
