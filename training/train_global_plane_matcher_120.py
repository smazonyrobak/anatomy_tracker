"""Paired head-only v3 TRAIN gate for global plane correction; no deployment claim."""

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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match

source = Path(__file__).resolve().parent
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
run = root / 'runs/global_plane_matcher_120_pilot'
seed, updates, batch, side, coarse_side, sites = 20261009120, 512, 2, 256, 24, 128
original_beam, blind_count = 14, 16
checkpoints = (0, 256, 512)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert sha(parent) == parent_receipt['checkpoint_sha256']['1959']
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(source / 'arbitrary_plane_one_shot_model.py') == \
    parent_config['source_sha256']['arbitrary_plane_one_shot_model.py']
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1959 and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
attach_global_plane_matcher(model, enabled=True)
heads = {'atlas': model.global_plane_matcher_120,
         'support_only': copy.deepcopy(model.global_plane_matcher_120)}
assert all(torch.equal(left, right) for left, right in zip(
    heads['atlas'].state_dict().values(), heads['support_only'].state_dict().values()))
optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
              for arm, head in heads.items()}
parameters = {arm: list(head.parameters()) for arm, head in heads.items()}
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
source_files = ('train_global_plane_matcher_120.py', 'global_plane_matcher_120.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'global_atlas_contrast_090.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py')
config = {'seed': seed, 'updates': updates, 'batch': batch, 'side': side,
    'coarse_side': coarse_side, 'valid_tissue_sites': sites,
    'checkpoints': checkpoints, 'original_beam': '111 prior top 8 base + top 6 anchor branches',
    'blind_beam': 'original 14 branches plus 2 unused anchors most normal-diverse from current beam',
    'blind_count': blind_count,
    'actions': 'each blind slot offers original state/prior and bounded corrected state/prior+match logit',
    'training_augmentations': 'exact and locally jittered synthetic truth; excluded from blind selection',
    'jitter_limits': {'rotation_rad': .2, 'translation_um': 1500.,
        'basis_log': .08, 'shear': .05},
    'loss': '2 rigid pose Huber at valid-tissue chart points against true cutting-plane state (blind corrections weighted by frozen input error exp[-(mm/4)^2], plus exact/near training gates) + .5 blind action-cost KL + .25 blind expected cost + .1 exact no-shift gate + .1 near correction gate + .02 normalized update penalty; local deformation is reserved for joint stage',
    'optimizer': 'paired independent AdamW heads only, lr 1e-4 cosine to 2e-5, weight_decay 1e-4',
    'control': 'zero atlas intensity after differentiable finite-thickness render; retain support',
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_config_sha256': sha(parent_run / 'config.json'),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'heads': {arm: head.state_dict() for arm, head in heads.items()},
        'optimizers': {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'head_step_{step:05d}.pt')


save(0)
attempts = 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(batch))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                generator=subject_rng).tolist()
            with torch.no_grad():
                sampled = sample_one_shot_slide_artifacts_v3(context, virtual,
                    draw_seed, side=side)
            for row, record in enumerate(sampled['provenance']):
                slot, used = pending[row], bool(sampled['eligible'][row])
                attempts += 1
                draws.write(json.dumps({'update': step, 'slot': slot,
                    'draw_attempt': attempts, 'used': used, **record}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'valid_mask',
                         'offsets', 'weights')}
                    accepted[slot]['record'] = record
            pending = [slot for slot in pending if slot not in accepted]
            draw_seed += 1
        synthetic = {key: torch.cat([accepted[slot][key] for slot in range(batch)])
                     for key in ('inputs', 'state', 'reflection',
                                 'valid_mask', 'offsets', 'weights')}
        with torch.no_grad():
            prediction = model.predict(synthetic['inputs'])
            branch_prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((branch_prior[:, :32].topk(8, -1).indices,
                branch_prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            row = torch.arange(batch, device='cuda')
            for _ in range(blind_count - original_beam):
                chosen = normals[row[:, None], beam // 2]
                similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity = -similarity
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = branch_prior[:, 32:].reshape(batch, 64, 2)[row, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            mode_index, reflection = beam // 2, beam % 2
            blind_state = prediction['state'][row[:, None], mode_index]
            jitter_limits = synthetic['state'].new_tensor(
                (.2, .2, .2, 1500., 1500., 1500., .08, .08, .05))
            jitter = (2 * torch.rand(batch, 9, generator=subject_rng).cuda() - 1) * jitter_limits
            near_state = compose_full_frame_state(synthetic['state'], jitter)
            all_state = torch.cat((blind_state, synthetic['state'][:, None],
                                   near_state[:, None]), 1)
            all_mode = torch.cat((mode_index, mode_index[:, :1].expand(-1, 2)), 1)
            all_reflection = torch.cat((reflection,
                synthetic['reflection'][:, None].expand(-1, 2)), 1)
            pixel = torch.multinomial(synthetic['valid_mask'].flatten(1).float(),
                sites, replacement=True)
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            target = rigid_points_090(synthetic['state'], synthetic['reflection'], chart)

        metrics = {}
        for arm, head in heads.items():
            output = global_plane_match(model, prediction, all_mode, all_reflection,
                synthetic['offsets'], synthetic['weights'], context['atlas'],
                (side, side), side=coarse_side, support_only=arm == 'support_only',
                candidate_state=all_state, matcher=head)
            original = rigid_points_090(output['input_state'], all_reflection,
                chart[:, None])
            corrected = rigid_points_090(output['state'], all_reflection,
                chart[:, None])
            original_cost = ((original - target[:, None]) / 1000).norm(dim=-1).mean(-1)
            residual = (corrected - target[:, None]) / 1000
            point_huber = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                reduction='none', beta=.5).sum(-1).mean(-1)
            pose_weight = (-(original_cost[:, :blind_count].detach() / 4).square()).exp()
            pose_loss = ((point_huber[:, :blind_count] * pose_weight).sum(-1)
                / pose_weight.sum(-1).clamp_min(1e-4)).mean()
            pose_loss = pose_loss + .25 * point_huber[:, blind_count].mean() \
                + 2 * point_huber[:, blind_count + 1].mean()
            corrected_cost = residual.norm(dim=-1).mean(-1)
            action_score = torch.stack((output['input_score'][:, :blind_count],
                output['score'][:, :blind_count]), -1).flatten(1)
            action_cost = torch.stack((original_cost[:, :blind_count],
                corrected_cost[:, :blind_count]), -1).flatten(1).detach()
            target_mass = F.softmax(-action_cost / .7, -1)
            score_loss = F.kl_div(F.log_softmax(action_score, -1), target_mass,
                reduction='batchmean')
            expected_cost = (F.softmax(action_score, -1) * action_cost).sum(-1).mean()
            exact_gate = F.softplus(output['match_logit'][:, blind_count]).mean()
            near_gate = F.softplus(-output['match_logit'][:, blind_count + 1]).mean()
            normalized_update = output['update'] / jitter_limits
            update_penalty = normalized_update.square().mean()
            loss = (2 * pose_loss + .5 * score_loss + .25 * expected_cost
                + .1 * exact_gate + .1 * near_gate + .02 * update_penalty)
            decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
            optimizer = optimizers[arm]
            optimizer.param_groups[0]['lr'] = 1e-4 * decay
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters[arm], 5.,
                error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                input_choice = output['input_score'][:, :blind_count].argmax(-1)
                corrected_choice = output['score'][:, :blind_count].argmax(-1)
                joint_choice = action_score.argmax(-1)
                metrics[arm] = {'pose_huber_mm': float(pose_loss),
                    'score_kl': float(score_loss), 'expected_cost_mm': float(expected_cost),
                    'exact_no_shift_gate': float(exact_gate),
                    'near_correction_gate': float(near_gate),
                    'update_penalty': float(update_penalty), 'loss': float(loss),
                    'input14_selected_mm': float(original_cost[:, :original_beam].gather(
                        1, output['input_score'][:, :original_beam].argmax(-1)[:, None]).mean()),
                    'input16_selected_mm': float(original_cost[:, :blind_count].gather(
                        1, input_choice[:, None]).mean()),
                    'input16_best_mm': float(original_cost[:, :blind_count].min(-1).values.mean()),
                    'corrected16_selected_mm': float(corrected_cost[:, :blind_count].gather(
                        1, corrected_choice[:, None]).mean()),
                    'corrected16_best_mm': float(corrected_cost[:, :blind_count].min(-1).values.mean()),
                    'joint16_selected_mm': float(action_cost.gather(
                        1, joint_choice[:, None]).mean()),
                    'joint_corrected_fraction': float((joint_choice % 2).float().mean()),
                    'gradient_norm': float(gradient)}
        log.write(json.dumps({'update': step,
            'physical_section_ids': [accepted[slot]['record']['physical_section_id']
                for slot in range(batch)], 'blind_branch_ids': beam.tolist(),
            'jitter_local_updates': jitter.tolist(),
            'valid_site_indices_sha256': hashlib.sha256(pixel.cpu().numpy().tobytes()).hexdigest(),
            'arms': metrics, 'seconds': time.perf_counter() - started},
            allow_nan=False) + '\n')
        if step % 64 == 0:
            log.flush()
            draws.flush()
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': batch * updates, 'draw_attempts': attempts,
    'parent_checkpoint_sha256': sha(parent), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'head_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
