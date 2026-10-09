"""TRAIN-only pilot: learn image-dependent atlas fit before pose feedback."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_FITTING_EVIDENCE_112_PROTOCOL_20261009.md'
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
run = root / 'runs/v3_fitting_evidence_112_pilot'
seed, updates, side, map_side = 20261009112, 512, 256, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
with parent.open('rb') as stream:
    parent_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
assert parent_sha256 == parent_receipt['checkpoint_sha256']['1959']
assert hashlib.sha256((parent_run / 'config.json').read_bytes()).hexdigest() == parent_receipt['config_sha256']
assert parent_receipt['updates'] == 1959 and not any(parent_receipt[key] for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used'))
for name, digest in parent_config['source_sha256'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
source_files = ('train_v3_fitting_evidence_112.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py')
config = {'seed': seed, 'updates': updates, 'image_side': side, 'map_side': map_side,
    'source': 'fresh independent v3 arbitrary-plane TRAIN synthetic draws; two different draws per batch; no orientation restriction',
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': parent_sha256,
    'parent_completion_sha256': hashlib.sha256((parent_run / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in source_files}, 'synthetic_provenance': context['provenance'],
    'trainable': ['warp_shared', 'warp_condition', 'warp', 'pair', 'atlas_encoder', 'fitted_matcher'],
    'frozen': 'encoder, lateral, pose, anchor pose/global, all other model heads',
    'candidates': 'exact physical plane, random local near-plane, support-matched hard top-16 frozen-pose branch',
    'fit_score': 'fitted candidate score with equal candidate priors; no pose prior can solve ranking',
    'loss': 'full valid-tissue exact/near dense coordinate error + local displacement/smoothness cost + exact/near versus hard fitted rank + matched-image versus swapped-image fitted rank',
    'map_supervision': 'synthetic observed-pixel CCF, not a refitted pose or segmentation label',
    'qualification': False, 'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1959 and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
trainable_modules = (model.warp_shared, model.warp_condition, model.warp,
    model.pair, model.atlas_encoder, model.fitted_matcher)
for module in trainable_modules:
    module.requires_grad_(True)
parameters = [parameter for module in trainable_modules for parameter in module.parameters()]
optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed, attempts = seed * 1000000, 0
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
y = (torch.arange(map_side, device='cuda') + .5) / map_side - .5 / side
x = (torch.arange(map_side, device='cuda') + .5) / map_side - .5 / side
yy, xx = torch.meshgrid(y, x, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)

run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
    'step': 0, 'config': config, 'subject_rng': subject_rng.get_state(),
    'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
    'cuda_rng': torch.cuda.get_rng_state_all(), 'calibrated': False},
    run / 'joint_step_00000.pt')

with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted = []
        while len(accepted) < 2:
            subject = int(torch.randint(len(context['subjects']), (1,), generator=subject_rng))
            with torch.no_grad():
                sample = sample_one_shot_slide_artifacts_v3(context, [subject], draw_seed, side)
            used = bool(sample['eligible'][0]) and int(sample['valid_mask'][0].sum()) >= side * side // 12
            attempts += 1
            draws.write(json.dumps({**sample['provenance'][0], 'step': step,
                'draw_attempt': attempts, 'used': used}) + '\n')
            draw_seed += 1
            if used:
                accepted.append(sample)
        batch = {key: torch.cat([sample[key] for sample in accepted]) for key in (
            'inputs', 'state', 'reflection', 'offsets', 'weights', 'centre', 'valid_mask')}
        prediction = model.predict(batch['inputs'])
        positive = {key: value[:1] for key, value in prediction.items() if torch.is_tensor(value)}
        swapped = {key: value[1:2] for key, value in prediction.items() if torch.is_tensor(value)}
        truth_state = batch['state'][:1]
        truth_reflection = batch['reflection'][:1]
        prior = (positive['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-positive['reflection_logit']),
            F.logsigmoid(positive['reflection_logit'])), -1)).flatten(1)

        with torch.no_grad():
            beam = prior.topk(16, -1).indices
            beam_states = positive['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            beam_reflection = beam % 2
            truth_five = rigid_points_090(truth_state, truth_reflection, corners)
            error_mm = (rigid_points_090(beam_states, beam_reflection, corners)
                - truth_five[:, None]).norm(dim=-1).mean(-1) / 1000
            beam_support = render_atlas_planes_090(context['atlas'], beam_states,
                beam_reflection, batch['offsets'][:1], batch['weights'][:1], candidate_chunk=2)
            true_support = render_atlas_planes_090(context['atlas'], truth_state[:, None],
                truth_reflection[:, None], batch['offsets'][:1], batch['weights'][:1])
            support_gap = (beam_support[:, :, 1].mean((-2, -1))
                - true_support[:, :, 1].mean((-2, -1))).abs()
            wrong = error_mm >= 1.5
            matched = wrong & (support_gap <= .10)
            eligible = matched if bool(matched.any()) else wrong
            hard_slot = (prior.gather(1, beam) - 5 * support_gap).masked_fill(
                ~eligible, -1e6).argmax(-1)
            if not bool(eligible.any()):
                hard_slot = error_mm.argmax(-1)
            hard_state = beam_states[torch.arange(1, device='cuda'), hard_slot]
            hard_reflection = beam_reflection[torch.arange(1, device='cuda'), hard_slot]
            hard_error_mm = error_mm.gather(1, hard_slot[:, None])[:, 0]
            hard_support_gap = support_gap.gather(1, hard_slot[:, None])[:, 0]

        local_update = torch.randn(1, 9, device='cuda') * torch.tensor(
            [.035, .035, .035, 250., 250., 150., .015, .015, .01], device='cuda')
        near_state = compose_full_frame_state(truth_state, local_update)
        states = torch.stack((truth_state, near_state, hard_state), 1)
        reflections = torch.stack((truth_reflection, truth_reflection,
            hard_reflection), 1)
        selected = {**positive, 'state': states,
            'log_mass': torch.zeros(1, 3, device='cuda'),
            'reflection_logit': torch.zeros(1, 3, device='cuda')}
        index = torch.arange(3, device='cuda')[None]
        mapped = model.map(selected, batch['offsets'][:1], index, reflections,
            (map_side, map_side), context['atlas'], batch['weights'][:1],
            feature_side=map_side, source_shape=(side, side))
        scores = model.score_fitted_candidates(batch['inputs'][:1], selected,
            mapped, context['atlas'], batch['weights'][:1])

        source_grid = torch.stack((2 * (xx + .5 / side) - 1,
            2 * (yy + .5 / side) - 1), -1)[None]
        valid = F.grid_sample(batch['valid_mask'][:1, None].float(),
            source_grid, align_corners=False)[:, 0] > .999
        target = F.grid_sample(batch['centre'][:1].permute(0, 3, 1, 2),
            source_grid, align_corners=False).permute(0, 2, 3, 1)
        predicted = mapped['centre_surface_ccf_ap_dv_ml_um']
        pixel_error_mm = (predicted[:, :2] - target[:, None]).norm(dim=-1) / 1000
        map_error_mm = (pixel_error_mm * valid[:, None]).sum((-2, -1)) / valid.sum().clamp_min(1)
        rigid = rigid_points_090(truth_state, truth_reflection, chart).reshape(
            1, map_side, map_side, 3)
        rigid_error_mm = (((rigid - target).norm(dim=-1) / 1000) * valid
            ).sum() / valid.sum().clamp_min(1)
        local = mapped['local_displacement_um'][:, :2] / 1000
        dx = local[..., 1:] - local[..., :-1]
        dy = local[..., 1:, :] - local[..., :-1, :]
        deformation_cost = .03 * local.square().mean() + .01 * (
            dx.square().mean() + dy.square().mean())

        swapped_selected = {**swapped, 'state': truth_state[:, None],
            'log_mass': torch.zeros(1, 1, device='cuda'),
            'reflection_logit': torch.zeros(1, 1, device='cuda')}
        swapped_map = model.map(swapped_selected, batch['offsets'][:1],
            torch.zeros(1, 1, device='cuda', dtype=torch.long), truth_reflection[:, None],
            (map_side, map_side), context['atlas'], batch['weights'][:1],
            feature_side=map_side, source_shape=(side, side))
        swapped_score = model.score_fitted_candidates(batch['inputs'][1:2],
            swapped_selected, swapped_map, context['atlas'], batch['weights'][:1])[0, 0]
        hard_rank = F.softplus(.5 + scores[0, 2] - scores[0, 0])
        near_rank = F.softplus(.25 + scores[0, 2] - scores[0, 1])
        image_rank = F.softplus(.5 + swapped_score - scores[0, 0])
        loss = map_error_mm[0, 0] + .25 * map_error_mm[0, 1] + deformation_cost \
            + .25 * (hard_rank + near_rank + image_rank)

        if step == 1 or step % 64 == 0:
            delta = torch.zeros(1, 9, device='cuda', requires_grad=True)
            probe = {**positive, 'state': compose_full_frame_state(hard_state, delta)[:, None],
                'log_mass': torch.zeros(1, 1, device='cuda'),
                'reflection_logit': torch.zeros(1, 1, device='cuda')}
            probe_map = model.map(probe, batch['offsets'][:1],
                torch.zeros(1, 1, device='cuda', dtype=torch.long),
                hard_reflection[:, None], (map_side, map_side), context['atlas'],
                batch['weights'][:1], feature_side=map_side, source_shape=(side, side))
            probe_score = model.score_fitted_candidates(batch['inputs'][:1],
                probe, probe_map, context['atlas'], batch['weights'][:1]).sum()
            fit_to_pose_gradient = torch.autograd.grad(probe_score, delta)[0].detach()
            fit_to_pose_gradient_norm = float(fit_to_pose_gradient.norm())
        else:
            fit_to_pose_gradient_norm = None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'draw_attempts': attempts,
            'positive_section_id': accepted[0]['provenance'][0]['physical_section_id'],
            'swapped_section_id': accepted[1]['provenance'][0]['physical_section_id'],
            'positive_mode': accepted[0]['provenance'][0]['mode'],
            'swapped_mode': accepted[1]['provenance'][0]['mode'],
            'hard_candidate_branch': int(beam.gather(1, hard_slot[:, None])[0, 0]),
            'hard_rigid_error_mm': float(hard_error_mm[0]),
            'hard_support_gap': float(hard_support_gap[0]),
            'hard_support_matched': bool(matched.gather(1, hard_slot[:, None])[0, 0]),
            'rigid_exact_error_mm': float(rigid_error_mm.detach()),
            'mapped_exact_error_mm': float(map_error_mm[0, 0].detach()),
            'mapped_near_error_mm': float(map_error_mm[0, 1].detach()),
            'fitted_scores_exact_near_hard': scores[0].detach().tolist(),
            'swapped_image_fitted_score': float(swapped_score.detach()),
            'deformation_cost': float(deformation_cost.detach()),
            'fit_to_pose_gradient_norm': fit_to_pose_gradient_norm,
            'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 64 == 0:
            print(json.dumps({key: row[key] for key in (
                'step', 'rigid_exact_error_mm', 'mapped_exact_error_mm',
                'mapped_near_error_mm', 'hard_rigid_error_mm',
                'hard_support_matched', 'fit_to_pose_gradient_norm',
                'peak_gpu_mb')}), flush=True)
        if step in (128, 512):
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'config': config, 'subject_rng': subject_rng.get_state(),
                'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'calibrated': False},
                run / f'joint_step_{step:05d}.pt')

with (run / 'draws.jsonl').open('rb') as stream:
    draws_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
with (run / 'training.jsonl').open('rb') as stream:
    training_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
completed = {'updates': updates, 'draw_attempts': attempts,
    'parent_checkpoint_sha256': parent_sha256,
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'draws_sha256': draws_sha256, 'training_sha256': training_sha256,
    'checkpoint_sha256': {}, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}
for step in (0, 128, 512):
    with (run / f'joint_step_{step:05d}.pt').open('rb') as stream:
        completed['checkpoint_sha256'][str(step)] = hashlib.file_digest(stream, 'sha256').hexdigest()
(run / 'completed.json').write_text(json.dumps(completed, indent=2))
