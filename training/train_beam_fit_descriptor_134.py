"""Train geometry-blind fitted-atlas descriptors on frozen 132 blind beams."""

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/BEAM_FIT_DESCRIPTOR_134_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/beam_fit_descriptor_134'
seed, side, map_side, descriptor_side, updates = 20261010134, 256, 96, 64, 4000
checkpoints = (0, 1000, 2000, 3000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert sha(parent_dir / 'draws.jsonl') == parent_done['draws_sha256']
assert sha(parent_dir / 'training.jsonl') == parent_done['training_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_done[key] for key in ('calibrated', 'expert_real_truth_used',
    'final_animals_used', 'public_benchmark_used', 'external_pretrained_weights_used') if key in parent_done)
context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance'] == parent_config['synthetic_provenance']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
atlas = context['atlas']

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
model.requires_grad_(False)
del saved

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
full = nn.ModuleDict({
    'source': nn.Sequential(nn.Conv2d(65, 64, 3, padding=1), nn.GELU(),
        nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GELU(),
        nn.Conv2d(64, 32, 3, padding=4, dilation=4)),
    'atlas': nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.GELU(),
        nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GELU(),
        nn.Conv2d(64, 32, 3, padding=4, dilation=4)),
})
heads = nn.ModuleDict({'full': full, 'support_only': copy.deepcopy(full)}).cuda()
optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=3e-4,
    weight_decay=1e-4) for arm in heads}
source_files = ('train_beam_fit_descriptor_134.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'image_side': side, 'map_side': map_side,
    'descriptor_side': descriptor_side, 'checkpoints': checkpoints,
    'appearance_schedule': 'v4,v4,v3 by accepted update',
    'beam': '132 exact blind16 (8 base + 6 anchor + 2 diverse unused anchor)',
    'eligible_target': 'softmax(-(observed-valid mean fitted 3D error mm - beam minimum)/0.5)',
    'eligible_gate': 'synthetic generator eligible and beam best fitted error <=1.5 mm',
    'score': 'fit_only=-3*energy; prior_plus_fit=frozen direct log prior-3*energy',
    'energy': 'reliability-weighted appearance + uncovered tissue + 0.25*local displacement mm + 0.25*local strain',
    'loss': 'equal-weight all-16 soft-label cross-entropy for fit-only and prior-plus-fit',
    'architecture': 'separate identically initialized 65->64->64->32 source and 2->64->64->32 atlas dilated convolutional heads; no coordinates',
    'optimizer': 'separate AdamW arms, lr 0.0003 cosine to 20%, weight decay 0.0001',
    'frozen': '132 pose, local map, atlas encoder, all prior model parameters',
    'trained': 'randomly initialized full-intensity and matched support-only descriptor heads',
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'allen_atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'allen_template_raw_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'allen_annotation_raw_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    path = run / f'descriptor_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'heads': heads.state_dict(),
        'optimizers': {arm: opt.state_dict() for arm, opt in optimizers.items()},
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
        'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
        'calibrated': False}, temporary)
    os.replace(temporary, path)


save(0)
started = time.perf_counter()
attempts = 0
accepted_ids = set()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        appearance = 'v3' if step % 3 == 0 else 'v4'
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        while True:
            virtual = int(rng.integers(len(context['subjects'])))
            draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            attempts += 1
            with torch.no_grad():
                sample = sampler(context, [virtual], draw_seed, side=side)
            record = sample['provenance'][0]
            if not bool(sample['eligible'][0]):
                draws.write(json.dumps({'update': step, 'attempt': attempts,
                    'appearance_version': appearance, 'used': False,
                    'reason': 'generator_ineligible', **record}, allow_nan=False) + '\n')
                continue

            with torch.no_grad():
                image, offsets, weights = sample['inputs'], sample['offsets'], sample['weights']
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                                  prior[:, 32:].topk(6, -1).indices + 32), -1)
                normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                for _ in range(2):
                    chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                    diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                    diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                    anchor = diversity.argmax(-1)
                    reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                    beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
                ids = beam[0]

                feature = F.adaptive_avg_pool2d(prediction['feature'], (descriptor_side, descriptor_side))
                gray = F.adaptive_avg_pool2d(image[:, :1], (descriptor_side, descriptor_side))
                gray_mean = F.avg_pool2d(gray, 5, 1, 2)
                texture = (F.avg_pool2d(gray.square(), 5, 1, 2) - gray_mean.square()).clamp_min(0).sqrt()
                reliability = .2 + .8 * (texture / .12).clamp(0, 1)
                reliability_mass = reliability.sum()
                reliability96 = F.interpolate(reliability, (map_side, map_side),
                    mode='bilinear', align_corners=False)
                atlas_pairs, surfaces, fixed_costs = [], [], []

                for part in ids.split(2):
                    count = len(part)
                    mapped = model.map(prediction, offsets, (part // 2)[None],
                        (part % 2)[None], (map_side, map_side), atlas, weights,
                        source_shape=(side, side))
                    surfaces.append(mapped['centre_surface_ccf_ap_dv_ml_um'][0])
                    coordinates = mapped['coordinates'][0].permute(0, 1, 4, 2, 3)
                    coordinates = F.interpolate(coordinates.reshape(count, -1, map_side, map_side),
                        (descriptor_side, descriptor_side), mode='bilinear', align_corners=False)
                    coordinates = coordinates.reshape(count, offsets.shape[-1], 3,
                        descriptor_side, descriptor_side).permute(0, 1, 3, 4, 2)
                    rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
                        (0., 0., 0.), (25., 25., 25.), weights.expand(count, -1))
                    support = rendered[:, 1:2].clamp(0, 1)
                    atlas_pairs.append(torch.cat((rendered[:, :1] / support.clamp_min(1e-4),
                        support), 1))

                    local = mapped['local_displacement_um'][0]
                    _, _, basis = full_frame_state_to_components(mapped['state'][0])
                    xstep = basis[..., :, 0].norm(dim=-1) / map_side
                    ystep = basis[..., :, 1].norm(dim=-1) / map_side
                    magnitude = local.norm(dim=1) / 1000
                    displacement = ((magnitude * reliability96[0, 0]).sum((-2, -1))
                        / reliability96.sum())
                    xweight = .5 * (reliability96[0, 0, :, 1:] + reliability96[0, 0, :, :-1])
                    yweight = .5 * (reliability96[0, 0, 1:, :] + reliability96[0, 0, :-1, :])
                    xstrain = (local[..., 1:] - local[..., :-1]).norm(dim=1) / xstep[:, None, None]
                    ystrain = (local[..., 1:, :] - local[..., :-1, :]).norm(dim=1) / ystep[:, None, None]
                    strain = ((xstrain * xweight).sum((-2, -1)) +
                        (ystrain * yweight).sum((-2, -1))) / (xweight.sum() + yweight.sum())
                    coverage = (reliability * (1 - support)).sum((1, 2, 3)) / reliability_mass
                    fixed_costs.append(coverage + .25 * displacement + .25 * strain)

                atlas_pair = torch.cat(atlas_pairs)
                support = atlas_pair[:, 1:2]
                fixed_cost = torch.cat(fixed_costs)
                surfaces = torch.cat(surfaces)
                upsampled = F.interpolate(surfaces.permute(0, 3, 1, 2), (side, side),
                    mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                valid = sample['valid_mask'][0]
                mapped_mm = ((upsampled - sample['centre'][0]).norm(dim=-1)[:, valid]
                    .mean(-1) / 1000)
                best_mm = float(mapped_mm.min())

            used = best_mm <= 1.5
            draws.write(json.dumps({'update': step, 'attempt': attempts,
                'appearance_version': appearance, 'used': used,
                'reason': 'accepted' if used else 'no_near_fitted_candidate',
                'beam_best_mapped_mm': best_mm, 'beam_branch_ids': ids.tolist(),
                **record}, allow_nan=False) + '\n')
            if used:
                accepted_ids.add(record['physical_section_id'])
                break

        target = F.softmax(-(mapped_mm - mapped_mm.min()) / .5, -1)
        source_input = torch.cat((feature, gray), 1)
        arm_metrics = {}
        lr = 3e-4 * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / updates)))
        for arm in ('full', 'support_only'):
            for group in optimizers[arm].param_groups:
                group['lr'] = lr
            target_input = atlas_pair if arm == 'full' else torch.cat((
                torch.zeros_like(atlas_pair[:, :1]), support), 1)
            source_descriptor = F.normalize(heads[arm]['source'](source_input), dim=1)
            atlas_descriptor = F.normalize(heads[arm]['atlas'](target_input), dim=1)
            cosine = (source_descriptor * atlas_descriptor).sum(1, keepdim=True)
            appearance_cost = (reliability * support * ((1 - cosine) / 2).clamp(0, 1)
                ).sum((1, 2, 3)) / reliability_mass
            energy = appearance_cost + fixed_cost
            fit_logits = -3 * energy
            prior_fit_logits = prior[0, ids] - 3 * energy
            fit_ce = -(target * F.log_softmax(fit_logits, -1)).sum()
            prior_fit_ce = -(target * F.log_softmax(prior_fit_logits, -1)).sum()
            loss = .5 * (fit_ce + prior_fit_ce)
            optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), 5.,
                error_if_nonfinite=True)
            optimizers[arm].step()
            fit_choice = int(fit_logits.detach().argmax())
            prior_fit_choice = int(prior_fit_logits.detach().argmax())
            arm_metrics[arm] = {'fit_ce': float(fit_ce.detach()),
                'prior_fit_ce': float(prior_fit_ce.detach()),
                'gradient_norm': float(gradient),
                'fit_only_selected_mapped_mm': float(mapped_mm[fit_choice]),
                'prior_fit_selected_mapped_mm': float(mapped_mm[prior_fit_choice]),
                'fit_only_selected_le_1p5': bool(mapped_mm[fit_choice] <= 1.5),
                'prior_fit_selected_le_1p5': bool(mapped_mm[prior_fit_choice] <= 1.5)}

        row = {'update': step, 'attempts': attempts, 'accepted_presentations': step,
            'appearance_version': appearance,
            'physical_section_id': record['physical_section_id'],
            'base_subject_id': record['base_lineage']['subject_id'],
            'beam_best_mapped_mm': best_mm,
            'prior_selected_mapped_mm': float(mapped_mm[prior[0, ids].argmax()]),
            'lr': lr, 'arms': arm_metrics,
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            draws.flush()
            log.flush()
            print(json.dumps({'event': 'training_milestone', **row}), flush=True)
        if step in checkpoints:
            save(step)

(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'draw_attempts': attempts, 'accepted_presentations': updates,
    'unique_accepted_physical_sections': len(accepted_ids),
    'parent_checkpoint_sha256': sha(parent), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'descriptor_step_{step:05d}.pt')
        for step in checkpoints}, 'calibrated': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
    'seconds': time.perf_counter() - started}), flush=True)
