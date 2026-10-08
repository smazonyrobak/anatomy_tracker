"""TRAIN-only 099 multiscale descriptor pilot on the frozen 094 candidate geometry."""
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
protocol = source.parent / 'docs/publication/CONTEXTUAL_MATCH_099_PROTOCOL_20261008.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent = parent_run / 'joint_step_20000.pt'
run = root / 'runs/contextual_match_099_pilot'
seed, batches, side, pixels = 2026100899, 4000, 256, 1024
checkpoints = (0, 4000)
separation, sharpness = .1, 10.
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '0aa387fd8db0517ebe10271c75da8610ccdbe84f79c07f9ed07a0f9dba52d974'
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent) == parent_receipt['checkpoint_sha256']['20000'] \
    == '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))

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
student = WholeSliceAtlasFeedback099().cuda().train().requires_grad_(False)
parent_matcher = {key.removeprefix('matcher.'): value for key, value in
    saved['spatial_fit'].items() if key.startswith('matcher.')}
mapped = {}
for key, value in parent_matcher.items():
    if key.startswith(('image.', 'atlas.')):
        stem, tail = key.split('.', 1)
        key = f'{stem}.base.{tail}'
    mapped[key] = value
missing, unexpected = student.load_state_dict(mapped, strict=False)
assert not unexpected and set(missing) == {
    key for key in student.state_dict() if '.context.' in key}
assert all(torch.count_nonzero(student.state_dict()[key]) == 0
           for key in missing if key.endswith(('out.weight', 'out.bias')))
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
    'sampling': 'one sample_one_shot_stream draw (one random plane geometry and one appearance) per attempt; CPU subject_rng selects one of 4096 TRAIN virtual subjects rooted in 64 base deformations; draw_seed increments on every attempt; reject generator-ineligible and draws without fine or coarse positive/negative sites; assert unique physical_section_id for each accepted section',
    'draw_seed_start': draw_seed, 'image_side': side, 'rigid_error_pixels': pixels,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 4,
    'candidate_rule': 'unchanged 097/094: prior=log_mass+reflection logsigmoid; top 8 IDs 0:31 and top 6 IDs 32:63; rigid error from 1024 replacement-sampled valid pixels; first minimum rigid error; second nearest initial atlas support among candidates >=0.5 mm worse (fallback remaining); third highest prior among remaining >=1 mm worse (fallback remaining); fourth uniform random distinct remaining using subject_rng; all candidate IDs/errors/support/fallbacks logged',
    'frozen_geometry': '094 step20000 frozen predictor plus 094 SpatialJointFit094 teacher on the four selected states; coarse descriptors/physical bins use chosen initial states and fine descriptors/physical bins use teacher fitted states, matching 094 inference; no true-pose teacher',
    'matcher_initialization': '094 matcher projection image/atlas weights loaded into frozen .base submodules; all other 083/094 matcher weights unchanged; 097 descriptor checkpoint not loaded; zero-initialized residual .context output yields step0 094 logits',
    'trainable': ['image.context', 'atlas.context'],
    'descriptor_channels': 16, 'atlas_depths': 9,
    'context': 'three 2x spatial pyramid levels at 1/2, 1/4, 1/8 resolution, near/mid/far widths 32/48/64 and skip-up convolutions; atlas 3D convolutions preserve all 9 depth slices',
    'loss': 'equal mean of fine and coarse candidate-site losses; each candidate-site uses its best supported positive cosine and hardest supported negative cosine; softplus(sharpness*(hard_negative-best_positive+separation))/sharpness; candidate-sites need both classes',
    'fine': {'grid': 32, 'classes': 225, 'positive_um': 1500, 'negative_um': 2000},
    'coarse': {'grid': 16, 'classes': 729, 'positive_um': 2000, 'negative_um': 3000},
    'support_threshold': .5, 'support_penalty_removed': 4.,
    'separation_cosine': separation, 'softplus_sharpness': sharpness,
    'learning_rate': 1e-4, 'weight_decay': 1e-4,
    'parent_094': str(parent), 'parent_094_sha256': sha(parent),
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_contextual_match_099.py', 'contextual_matcher_099.py',
        'train_contrastive_match_097.py', 'spatial_joint_fit_094.py',
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
        run / f'contextual_step_{step:05d}.pt')


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
            used = bool(sampled['eligible'][0])
            attempts += 1
            draw_seed += 1
            if used:
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
            for name, grid, radius, positive_um, negative_um in (
                ('fine', 32, 2, 1500, 2000), ('coarse', 16, 4, 2000, 3000)):
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
                cells = (2 * radius + 1) ** 2
                index = torch.arange(9 * cells, device='cuda')
                depth_um = 1500 * (index // cells - 4)
                dy = (index // (2 * radius + 1) % (2 * radius + 1) - radius).float()
                dx = (index % (2 * radius + 1) - radius).float()
                shift_x = torch.where(reflected.bool()[..., None], -dx, dx)
                shift = torch.stack((shift_x, dy[None, None].expand_as(shift_x)), -1) / grid
                lateral = torch.einsum('bkij,bkcj->bkci', edges, shift)
                bin_ccf = (base[:, :, None] + lateral[:, :, :, None, None, :]
                    + depth_um[None, None, :, None, None, None]
                    * frame[..., :, 2][:, :, None, None, None, :])
                squared_distance = (bin_ccf - truth[:, None, None]).square().sum(-1)
                geometry[name] = (visible, squared_distance <= positive_um ** 2,
                    squared_distance >= negative_um ** 2)

        coarse_match = student(prediction['feature'], chosen_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side), match_only=True)
        matched = {'coarse': (coarse_match['coarse_match_logits'],
                              coarse_match['coarse_match_support'])}
        del coarse_match
        fine_match = student(prediction['feature'], fitted_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side), match_only=True)
        matched['fine'] = (fine_match['fine_match_logits'],
                           fine_match['fine_match_support'])
        del fine_match
        temperature = student.log_temperature.exp().clamp(2, 20)
        objective = {}
        for name, grid, classes in (('fine', 32, 225), ('coarse', 16, 729)):
            visible, positive_distance, negative_distance = geometry[name]
            logits, atlas_support = matched[name]
            supported = atlas_support >= .5
            positive = supported & positive_distance
            negative = supported & negative_distance
            cosine = (logits + 4 * (1 - atlas_support)) / temperature
            scores = cosine.permute(0, 1, 3, 4, 2).reshape(-1, classes)
            positives = positive.permute(0, 1, 3, 4, 2).reshape_as(scores)
            negatives = negative.permute(0, 1, 3, 4, 2).reshape_as(scores)
            eligible = (visible[:, None].expand(-1, 4, -1, -1).reshape(-1)
                & positives.any(-1) & negatives.any(-1))
            objective[name] = (visible, positive, negative, scores, positives, negatives, eligible)
        if not all(bool(objective[name][-1].any()) for name in ('fine', 'coarse')):
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                'used': False, 'reason': 'no_fine_or_coarse_contrastive_eligible_site'}) + '\n')
            step -= 1
            continue
        assert record['physical_section_id'] not in seen_sections
        seen_sections.add(record['physical_section_id'])
        draw_row = {**record, 'step': step, 'draw_attempt': attempts, 'used': True,
            'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': beam_error[slots].tolist(),
            'initial_support': support[slots].tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback}
        draws.write(json.dumps(draw_row, allow_nan=False) + '\n')
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        margins = {}
        for name in ('fine', 'coarse'):
            visible, positive, negative, scores, positives, negatives, eligible = objective[name]
            active = scores[eligible]
            best_positive = active.masked_fill(~positives[eligible], -torch.inf).amax(-1)
            hardest_negative = active.masked_fill(~negatives[eligible], -torch.inf).amax(-1)
            margins[name] = (best_positive - hardest_negative).mean()
            losses[name] = (F.softplus(sharpness * (
                hardest_negative - best_positive + separation)) / sharpness).mean()
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
            'fine_visible_sites': int(objective['fine'][0].sum()),
            'coarse_visible_sites': int(objective['coarse'][0].sum()),
            'fine_eligible_sites': int(objective['fine'][-1].reshape(-1, 4, 32, 32).any(1).sum()),
            'coarse_eligible_sites': int(objective['coarse'][-1].reshape(-1, 4, 16, 16).any(1).sum()),
            'fine_eligible_candidate_sites': int(objective['fine'][-1].sum()),
            'coarse_eligible_candidate_sites': int(objective['coarse'][-1].sum()),
            'fine_candidates_with_eligible_sites': int(objective['fine'][-1].reshape(-1, 4, 32, 32).any((-2, -1)).sum()),
            'coarse_candidates_with_eligible_sites': int(objective['coarse'][-1].reshape(-1, 4, 16, 16).any((-2, -1)).sum()),
            'fine_positive_bins': int((objective['fine'][1] & objective['fine'][0][:, None, None]).sum()),
            'fine_negative_bins': int((objective['fine'][2] & objective['fine'][0][:, None, None]).sum()),
            'coarse_positive_bins': int((objective['coarse'][1] & objective['coarse'][0][:, None, None]).sum()),
            'coarse_negative_bins': int((objective['coarse'][2] & objective['coarse'][0][:, None, None]).sum()),
            'fine_loss': float(losses['fine'].detach()),
            'coarse_loss': float(losses['coarse'].detach()),
            'fine_hard_margin_cosine': float(margins['fine'].detach()),
            'coarse_hard_margin_cosine': float(margins['coarse'].detach()),
            'loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'fine_eligible_candidate_sites',
                'coarse_eligible_candidate_sites', 'loss', 'fine_hard_margin_cosine',
                'coarse_hard_margin_cosine', 'peak_gpu_mb', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'distinct_physical_sections': len(seen_sections),
    'draw_attempts': attempts, 'protocol_sha256': sha(protocol),
    'parent_094_checkpoint_sha256': sha(parent),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'contextual_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}),
      flush=True)
