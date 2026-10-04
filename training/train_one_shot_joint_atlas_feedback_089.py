"""Train direct pose and bounded atlas-conditioned deformation on predicted branches."""
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
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

parent_run = root / 'runs/allbeam_fitted_ranker_085_pilot'
parent = parent_run / 'ranker_step_01000.pt'
run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
seed, batches, side, pixels = 2026100489, 6000, 256, 128
checkpoints = (0, 1000, 3000, 6000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :2] @ basis, xy - .5)


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert receipt['batches'] == 1000 and not receipt['calibrated']
assert not receipt['public_benchmark_used']
assert sha(parent) == receipt['checkpoint_sha256']['1000']
assert sha(parent_run / 'config.json') == receipt['config_sha256']
assert sha(parent_run / 'draws.jsonl') == receipt['draws_sha256']
assert sha(parent_run / 'training.jsonl') == receipt['training_sha256']
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
head = WholeSliceAtlasFeedback083().cuda().train()
head.load_state_dict(checkpoint['feedback'], strict=True)
del checkpoint
model.requires_grad_(False)
head.requires_grad_(False)
for layer in (model.pose[-1], model.anchor_pose, model.anchor_global,
              model.warp_shared, model.warp_condition, model.warp, model.pair,
              head.spatial, head.summary, head.pose, head.mapper_feature):
    layer.requires_grad_(True)
groups = (
    (model.pose[-1].parameters(), 1e-5),
    (model.anchor_pose.parameters(), 2e-5),
    (model.anchor_global.parameters(), 2e-5),
    (model.warp_shared.parameters(), 1e-5),
    (model.warp_condition.parameters(), 1e-5),
    (model.warp.parameters(), 1e-5),
    (model.pair.parameters(), 1e-5),
    (head.spatial.parameters(), 3e-5),
    (head.summary.parameters(), 3e-5),
    (head.pose.parameters(), 3e-5),
    (head.mapper_feature.parameters(), 3e-5),
)
optimizer = torch.optim.AdamW([{'params': list(parameters), 'lr': rate}
                               for parameters, rate in groups], weight_decay=1e-4)
rates = [rate for _, rate in groups]
trainable = [parameter for group in optimizer.param_groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
fit_weight = None
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')
local_index = torch.arange(2, device='cuda')[None]

run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'accepted_synthetic_per_batch': 1, 'candidate_beam': {'old': 8, 'anchor': 6},
    'trained_pair': 'prior-top and physical-best existing beam branch; prior runner-up if identical',
    'truth_role': 'TRAIN physical target and existing-branch choice only; never an inference candidate',
    'parent_085': str(parent), 'parent_085_sha256': sha(parent),
    'parent_085_config_sha256': sha(parent_run / 'config.json'),
    'parent_085_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_083_checkpoint_sha256': parent_config['pilot_083_checkpoint_6000_sha256'],
    'trainable': ['model.pose[-1]', 'model.anchor_pose', 'model.anchor_global',
                  'model.warp_shared', 'model.warp_condition', 'model.warp', 'model.pair',
                  'head.spatial', 'head.summary', 'head.pose', 'head.mapper_feature'],
    'frozen': ['encoder/lateral', '083 image/atlas embeddings and temperature',
               'atlas_encoder', 'pose_refiner', 'quality/scorers', 'uncertainty'],
    'fit_weight': 'monotone cap every 100 batches: min 0.2*base/fit gradient-norm ratio over nonzero old-pose, anchor-pose/global, warp, feedback-pose and feedback-map groups; cap 0.5, no positive floor',
    'native_fit_gate': 'detached sigmoid((2 - rigid_physical_mm)/0.5), multiplied without renormalization',
    'map': 'two 083 passes; differentiable 64-grid summary and 96-grid bounded affine-free warp',
    'synthetic_provenance': context['provenance'],
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        'train_one_shot_joint_atlas_feedback_089.py', 'arbitrary_plane_one_shot_model.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py')},
    'calibrated': False, 'public_benchmark_used': False, 'real_labels_used': False,
    'external_pretrained_weights_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'feedback': head.state_dict(),
        'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.get_state(),
        'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'fit_weight': fit_weight,
        'step': step, 'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            used = bool(sampled['eligible'][0])
            draws.write(json.dumps({**sampled['provenance'][0], 'step': step,
                                     'used': used}) + '\n')
            if used:
                accepted = {key: sampled[key] for key in
                    ('inputs', 'state', 'reflection', 'offsets', 'weights',
                     'centre', 'valid_mask')}
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict(accepted['inputs'])
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflection = flags[None, None].expand(1, model.modes, 2)
        reference = points(accepted['state'], accepted['reflection'], corners)
        five = (points(states, reflection, corners) - reference[:, None, None]).norm(
            dim=-1).mean(-1)
        pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                                  pixels, replacement=True)
        target = accepted['centre'].reshape(1, -1, 3).gather(
            1, pixel[..., None].expand(-1, -1, 3))
        chart = torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                            -1).float() / side
        dense = (points(states, reflection, chart[:, None, None]) - target[:, None, None]
                 ).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        physical_mm = (.75 * dense + .25 * five + normal_penalty[..., None]).flatten(1) / 1000
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = physical_mm.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
        target_mass = F.softmax(-physical_mm.detach() / .7, -1)
        direct = (-1.5 * torch.logsumexp(prior - physical_mm / 1.5, -1).mean()
                  + .5 * F.kl_div(prior, target_mass, reduction='batchmean')
                  + .5 * physical_mm[:, :2 * model.base_modes].min(-1).values.mean()
                  + (neighbourhood * near_distance).sum(-1).mean())
        with torch.no_grad():
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            ranked = prior.gather(1, beam).argsort(-1, descending=True)
            lead = ranked[:, 0]
            best = physical_mm.gather(1, beam).argmin(-1)
            other = torch.where(lead == best, ranked[:, 1], best)
            choice = beam.gather(1, torch.stack((lead, other), -1))
        reflected = choice % 2
        selected = {**prediction,
            'state': prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12)),
            'log_mass': prediction['log_mass'].gather(1, choice // 2),
            'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
        first = head(prediction['feature'], selected['state'], reflected,
                     context['atlas'], accepted['offsets'], accepted['weights'])
        mapped64 = model.map({**selected, 'state': first['state']}, accepted['offsets'],
            local_index, reflected, (64, 64), context['atlas'], accepted['weights'],
            return_refinement_feature=True, feature_side=64, source_shape=(side, side),
            spatial_evidence=first['spatial_evidence'])
        second = head(prediction['feature'], first['state'], reflected,
                      context['atlas'], accepted['offsets'], accepted['weights'],
                      fit_summary=fit_summary(mapped64))
        mapped96 = model.map({**selected, 'state': second['state']}, accepted['offsets'],
            local_index, reflected, (96, 96), context['atlas'], accepted['weights'],
            feature_side=96, source_shape=(side, side),
            spatial_evidence=second['spatial_evidence'])
        grid = (torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                            -1).float() + .5) * (2 / side) - 1
        surface = mapped96['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid[:, None].expand(2, -1, -1, -1),
                               padding_mode='border', align_corners=False).reshape(
                                   2, 3, pixels).transpose(1, 2)
        mapped_error_mm = (fitted - target[0, None]).norm(dim=-1).mean(-1) / 1000
        local = mapped96['local_displacement_um'] / 1000
        warp_penalty = (.03 * local.square().mean()
                        + .01 * (local[..., 1:, :] - local[..., :-1, :]).square().mean()
                        + .01 * (local[..., 1:] - local[..., :-1]).square().mean())
        base_loss = direct + mapped_error_mm.mean() + warp_penalty
        fit_map = model.atlas_fit_loss(accepted['inputs'], mapped96, context['atlas'],
                                       accepted['weights'], accepted['valid_mask'])
        gate = torch.sigmoid((2 - physical_mm.gather(1, choice).detach()) / .5)
        fit_term = (gate * fit_map).mean()
        audit = None
        if step == 1 or step % 100 == 0:
            names = ('old_pose', 'anchor_pose', 'anchor_global', 'warp',
                     'head_pose', 'head_mapper')
            parameters = (model.pose[-1].weight, model.anchor_pose[-1].weight,
                          model.anchor_global[-1].weight, model.warp[-1].weight,
                          head.pose.weight, head.mapper_feature.weight)
            fit_grads = torch.autograd.grad(fit_term, parameters, retain_graph=True,
                                            allow_unused=True)
            fit_norms = {name: 0. if grad is None else float(grad.norm())
                         for name, grad in zip(names, fit_grads)}
            audit = {'fit_gradient_norms': fit_norms}
            base_grads = torch.autograd.grad(base_loss, parameters, retain_graph=True,
                                             allow_unused=True)
            base_norms = {name: 0. if grad is None else float(grad.norm())
                          for name, grad in zip(names, base_grads)}
            ratios = [.2 * base_norms[name] / fit_norms[name]
                      for name in names if base_norms[name] > 0 and fit_norms[name] > 0]
            cap = min(.5, min(ratios, default=0.))
            fit_weight = cap if fit_weight is None else min(fit_weight, cap)
            audit['base_gradient_norms'] = base_norms
            audit['fit_weight_cap'] = cap
            if step == 1:
                assert fit_weight > 0 and fit_norms['warp'] > 0
                assert fit_norms['old_pose'] + fit_norms['anchor_pose'] + fit_norms['anchor_global'] > 0
                assert base_norms['head_pose'] > 0 and base_norms['head_mapper'] > 0
        loss = base_loss + fit_weight * fit_term
        progress = step / batches
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * (.2 + .8 * .5 * (1 + math.cos(math.pi * progress)))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'batch': step, 'synthetic_presentations': step,
                   'direct': float(direct.detach()),
                   'mapped96_mm': float(mapped_error_mm.detach().mean()),
                   'mapped96_by_candidate_mm': mapped_error_mm.detach().tolist(),
                   'rigid_by_candidate_mm': physical_mm.gather(1, choice).detach().tolist(),
                   'warp_penalty': float(warp_penalty.detach()),
                   'native_fit': float(fit_term.detach()), 'fit_weight': fit_weight,
                   'fit_gate': gate.detach().tolist(), 'gradient_norm': float(gradient),
                   'seconds': time.perf_counter() - started, **audit}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches,
    'parent_085_sha256': sha(parent),
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
