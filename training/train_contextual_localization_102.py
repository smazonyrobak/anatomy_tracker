"""Train the 099 context residuals to localize supported atlas matches."""
import hashlib
import json
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.contextual_matcher_099 import WholeSliceAtlasFeedback099
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/CONTEXTUAL_LOCALIZATION_102_PROTOCOL_20261008.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent = parent_run / 'joint_step_20000.pt'
descriptor_run = root / 'runs/contextual_match_099_pilot'
descriptor = descriptor_run / 'contextual_step_04000.pt'
run = root / 'runs/contextual_localization_102_pilot'
seed, batches, side, pixels = 20261008102, 4000, 256, 1024
checkpoints = (0, 2000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '47af4b65b70b755b9349b20975bc29f2df20dfeb13245ae5667b46c4b7f7de6b'
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
descriptor_receipt = json.loads((descriptor_run / 'completed.json').read_text())
descriptor_config = json.loads((descriptor_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent) == parent_receipt['checkpoint_sha256']['20000'] \
    == '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert descriptor_receipt['batches'] == 4000
assert sha(descriptor) == descriptor_receipt['checkpoint_sha256']['4000'] \
    == 'c4a85f1141109460eb2257ae77bb170bf8a650cd41a1d4d1dc550e84b913ec61'
assert sha(descriptor_run / 'config.json') == descriptor_receipt['config_sha256']
assert sha(descriptor_run / 'training.jsonl') == descriptor_receipt['training_sha256']
assert sha(descriptor_run / 'draws.jsonl') == descriptor_receipt['draws_sha256']
assert all(sha(source / name) == digest for name, digest in descriptor_config['source_sha256'].items())
assert descriptor_config['parent_094_sha256'] == sha(parent)
assert not any(receipt[key] for receipt in (parent_receipt, descriptor_receipt)
    for key in ('calibrated', 'public_benchmark_used', 'real_labels_used',
                'external_pretrained_weights_used'))

context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and not saved['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
teacher = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
teacher.load_state_dict(saved['spatial_fit'], strict=True)
del saved
saved = torch.load(descriptor, map_location='cpu', weights_only=True)
assert saved['step'] == 4000 and not saved['calibrated']
student = WholeSliceAtlasFeedback099().cuda().train().requires_grad_(False)
student.load_state_dict(saved['matcher'], strict=True)
del saved
student.image.context.requires_grad_(True)
student.atlas.context.requires_grad_(True)
assert all(parameter.requires_grad == name.startswith(('image.context.', 'atlas.context.'))
    for name, parameter in student.named_parameters())
trainable = list(student.image.context.parameters()) + list(student.atlas.context.parameters())
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
seen_sections = set()
attempts = 0
flags = torch.tensor([0, 1], device='cuda')

config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'sampling': 'one independent geometry and appearance per sample_one_shot_stream draw; 4096 TRAIN virtual subjects from 64 base deformation plans; rejected attempts retained; never deliberate paired variants',
    'draw_seed_start': draw_seed, 'image_side': side, 'rigid_error_pixels': pixels,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 4,
    'candidate_rule': 'unchanged 099 frozen-094 prior beam: physical best, closest initial atlas support at least 0.5mm worse, highest-prior at least 1mm worse, distinct random fallback',
    'frozen_geometry': '094 predictor and fitter; coarse on initial candidate, fine on 094 fitted candidate; no true-plane inference candidate',
    'matcher_initialization': 'authenticated 099 step4000; only image.context and atlas.context update',
    'loss': 'equal fine/coarse means of supported-bin distance-soft-target CE plus 0.2 times Huber of exact 094 argmax-neighborhood decoded CCF distance in mm',
    'fine': {'grid': 32, 'classes': 225, 'sigma_um': 500, 'nearest_eligible_um': 1500},
    'coarse': {'grid': 16, 'classes': 729, 'sigma_um': 1000, 'nearest_eligible_um': 2000},
    'support_threshold': .5, 'coordinate_loss_weight': .2, 'huber_beta_mm': 1.,
    'learning_rate': 1e-4, 'weight_decay': 1e-4,
    'parent_094': str(parent), 'parent_094_sha256': sha(parent),
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_config_sha256': sha(parent_run / 'config.json'),
    'parent_099': str(descriptor), 'parent_099_sha256': sha(descriptor),
    'parent_099_completed_sha256': sha(descriptor_run / 'completed.json'),
    'parent_099_config_sha256': sha(descriptor_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'development_panel_plan_indices': list(range(10200, 10208)),
    'source_sha256': {name: sha(source / name) for name in (
        'train_contextual_localization_102.py', 'train_contextual_match_099.py',
        'contextual_matcher_099.py', 'spatial_joint_fit_094.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_model.py', 'global_atlas_contrast_090.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
        'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'synthetic_provenance': context['provenance'], 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'step': step, 'matcher': student.state_dict(),
        'config': config, 'calibrated': False},
        run / f'localization_step_{step:05d}.pt')


save(0)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    step = 0
    while step < batches:
        step += 1
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            record = sampled['provenance'][0]
            attempts += 1
            draw_seed += 1
            if bool(sampled['eligible'][0]):
                accepted = {key: sampled[key] for key in (
                    'inputs', 'state', 'reflection', 'offsets', 'weights',
                    'centre', 'valid_mask')}
            else:
                draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                    'used': False, 'reason': 'generator_ineligible'}) + '\n')

        with torch.no_grad():
            prediction = model.predict(accepted['inputs'])
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags[None, None].expand(1, model.modes, 2)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                pixels, replacement=True)
            target = accepted['centre'].reshape(1, -1, 3).gather(
                1, pixel[..., None].expand(-1, -1, 3))
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            rigid_mm = (rigid_points_090(states, reflections, chart) - target[:, None, None]
                ).norm(dim=-1).mean(-1).flatten(1) / 1000
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            initial = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            atlas_pair = render_atlas_planes_090(context['atlas'], initial, beam % 2,
                accepted['offsets'], accepted['weights'], candidate_chunk=2)
            support = atlas_pair[:, :, 1].mean((-2, -1))[0]
            beam_error = rigid_mm.gather(1, beam)[0]
            beam_prior = prior.gather(1, beam)[0]
            best = int(beam_error.argmin())
            gap = beam_error - beam_error[best]
            remaining = [index for index in range(14) if index != best]
            close = [index for index in remaining if gap[index] >= .5]
            support_fallback = not close
            if not close:
                close = remaining
            support_choice = min(close, key=lambda index: float(abs(support[index] - support[best])))
            remaining.remove(support_choice)
            wrong = [index for index in remaining if gap[index] >= 1.]
            prior_fallback = not wrong
            if not wrong:
                wrong = remaining
            prior_choice = max(wrong, key=lambda index: float(beam_prior[index]))
            remaining.remove(prior_choice)
            fourth = remaining[int(torch.randint(len(remaining), (1,), generator=subject_rng))]
            slots = torch.tensor([best, support_choice, prior_choice, fourth], device='cuda')
            choice = beam[:, slots]
            reflected = choice % 2
            chosen_state = prediction['state'].gather(1,
                (choice // 2)[..., None].expand(-1, -1, 12))
            fitted_state = teacher(prediction['feature'], chosen_state, reflected,
                context['atlas'], accepted['offsets'], accepted['weights'],
                source_shape=(side, side))['state'].detach()
            geometry = {}
            for name, grid, radius, sigma, tolerance in (
                ('fine', 32, 2, 500, 1500), ('coarse', 16, 4, 1000, 2000)):
                state = fitted_state if name == 'fine' else chosen_state
                centre, frame, basis = full_frame_state_to_components(state)
                edges = frame[..., :, :2] @ basis
                truth = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                    (grid, grid), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                visible = F.interpolate(accepted['valid_mask'][:, None].float(),
                    (grid, grid), mode='bilinear', align_corners=False)[:, 0] == 1
                axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
                yy, xx = torch.meshgrid(axis, axis, indexing='ij')
                chart_x = torch.where(reflected.bool()[..., None, None],
                    (side - 1) / side - xx, xx)
                query_chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
                base = centre[..., None, None, :] + torch.einsum(
                    'bkij,bkhwj->bkhwi', edges, query_chart - .5)
                width = 2 * radius + 1
                cells = width ** 2
                index = torch.arange(9 * cells, device='cuda')
                depth = index // cells
                dy = index // width % width - radius
                dx = index % width - radius
                lateral = torch.stack((torch.where(reflected.bool()[..., None], -dx, dx),
                    dy[None, None].expand(1, 4, -1)), -1).float() / grid
                bin_ccf = (base[:, :, None]
                    + torch.einsum('bkij,bkcj->bkci', edges, lateral)[:, :, :, None, None]
                    + 1500 * (depth - 4)[None, None, :, None, None, None]
                    * frame[..., :, 2][:, :, None, None, None, :])
                distance2 = (bin_ccf - truth[:, None, None]).square().sum(-1)
                geometry[name] = (visible, truth, distance2, centre, frame, edges,
                    query_chart, depth, dy, dx, sigma, tolerance)

        coarse = student(prediction['feature'], chosen_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side), match_only=True)
        fine = student(prediction['feature'], fitted_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side), match_only=True)
        matched = {'coarse': (coarse['coarse_match_logits'], coarse['coarse_match_support']),
                   'fine': (fine['fine_match_logits'], fine['fine_match_support'])}
        del coarse, fine
        objective = {}
        for name in ('fine', 'coarse'):
            visible, truth, distance2, centre, frame, edges, query_chart, depth, dy, dx, sigma, tolerance = geometry[name]
            logits, atlas_support = matched[name]
            supported = atlas_support >= .5
            eligible = visible[:, None] & distance2.masked_fill(~supported, torch.inf).amin(2).le(tolerance ** 2)
            target_weight = torch.exp(-distance2 / (2 * sigma ** 2)) * supported
            target_probability = target_weight / target_weight.sum(2, keepdim=True).clamp_min(1e-20)
            masked = logits.masked_fill(~supported, -1e4)
            cross_entropy = -(target_probability * masked.log_softmax(2)).sum(2)
            probability = masked.softmax(2)
            peak = probability.argmax(2)
            local_region = (
                ((depth[None, None, :, None, None] - depth[peak][:, :, None]).abs() <= 1)
                & ((dy[None, None, :, None, None] - dy[peak][:, :, None]).abs() <= 1)
                & ((dx[None, None, :, None, None] - dx[peak][:, :, None]).abs() <= 1))
            local = masked.masked_fill(~local_region, -1e4).softmax(2)
            mx = (local * dx[None, None, :, None, None]).sum(2)
            my = (local * dy[None, None, :, None, None]).sum(2)
            md = (local * depth[None, None, :, None, None]).sum(2)
            grid = logits.shape[-1]
            shift = torch.stack((torch.where(reflected.bool()[..., None, None], -mx, mx) / grid,
                                 my / grid), -1)
            decoded = (centre[..., None, None, :]
                + torch.einsum('bkij,bkhwj->bkhwi', edges, query_chart - .5 + shift)
                + 1500 * (md - 4)[..., None] * frame[:, :, None, None, :, 2])
            error_mm = (decoded - truth[:, None]).norm(dim=-1) / 1000
            huber = F.smooth_l1_loss(error_mm, torch.zeros_like(error_mm),
                beta=1., reduction='none')
            objective[name] = (eligible, cross_entropy, huber, error_mm, supported)

        if not all(bool(objective[name][0].any()) for name in ('fine', 'coarse')):
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                'used': False, 'reason': 'no_supported_near_truth_fine_or_coarse_site'}) + '\n')
            step -= 1
            continue
        assert record['physical_section_id'] not in seen_sections
        seen_sections.add(record['physical_section_id'])
        draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
            'used': True, 'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': beam_error[slots].tolist(),
            'initial_support': support[slots].tolist(),
            'support_fallback': support_fallback,
            'prior_fallback': prior_fallback}, allow_nan=False) + '\n')
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name in ('fine', 'coarse'):
            eligible, cross_entropy, huber, error_mm, _ = objective[name]
            losses[name] = ((cross_entropy + .2 * huber) * eligible).sum() / eligible.sum()
        loss = (losses['fine'] + losses['coarse']) / 2
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'virtual_subject_id': record['virtual_subject_id'],
            'animal_id': record['base_lineage']['animal_id'],
            'specimen_id': record['base_lineage']['specimen_id'],
            'experiment_id': record['base_lineage']['experiment_id'],
            'synthetic_animal_id': record['base_lineage']['synthetic_animal_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': beam_error[slots].tolist(),
            'initial_support': support[slots].tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'fine_eligible_candidate_sites': int(objective['fine'][0].sum()),
            'coarse_eligible_candidate_sites': int(objective['coarse'][0].sum()),
            'fine_candidates_with_eligible_sites': int(objective['fine'][0].any((-2, -1)).sum()),
            'coarse_candidates_with_eligible_sites': int(objective['coarse'][0].any((-2, -1)).sum()),
            'fine_supported_bins': int((objective['fine'][4] & objective['fine'][0][:, :, None]).sum()),
            'coarse_supported_bins': int((objective['coarse'][4] & objective['coarse'][0][:, :, None]).sum()),
            'fine_loss': float(losses['fine'].detach()),
            'coarse_loss': float(losses['coarse'].detach()),
            'fine_decoded_error_mm': float(objective['fine'][3][objective['fine'][0]].mean().detach()),
            'coarse_decoded_error_mm': float(objective['coarse'][3][objective['coarse'][0]].mean().detach()),
            'loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'fine_eligible_candidate_sites',
                'coarse_eligible_candidate_sites', 'loss', 'fine_decoded_error_mm',
                'coarse_decoded_error_mm', 'peak_gpu_mb', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches,
    'distinct_physical_sections': len(seen_sections), 'draw_attempts': attempts,
    'protocol_sha256': sha(protocol),
    'parent_094_checkpoint_sha256': sha(parent),
    'parent_099_checkpoint_sha256': sha(descriptor),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'localization_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
    'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
