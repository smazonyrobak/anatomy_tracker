"""TRAIN-only anatomical contrast for the frozen 094 fine matcher."""
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
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/CONTRASTIVE_MATCH_097_PROTOCOL_20261005.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent = parent_run / 'joint_step_20000.pt'
run = root / 'runs/contrastive_match_097_pilot'
seed, batches, side, pixels = 2026100597, 4000, 256, 1024
checkpoints = (0, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '0a39656bce38a3364d7f1a2aab03b7e9a9d1077a271765da3f0bf926bba493fe'
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
student = WholeSliceAtlasFeedback083().cuda().train().requires_grad_(False)
student.load_state_dict({key.removeprefix('matcher.'): value for key, value in
    saved['spatial_fit'].items() if key.startswith('matcher.')}, strict=True)
del saved
student.image.requires_grad_(True)
student.atlas.requires_grad_(True)
trainable = list(student.image.parameters()) + list(student.atlas.parameters())
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
seen_sections = set()
attempts = 0
flags = torch.tensor([0, 1], device='cuda')
axis = (torch.arange(32, device='cuda') + .5) / 32 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
index = torch.arange(225, device='cuda')
depth_um = 1500 * (index // 25 - 4)
dy = (index // 5 % 5 - 2).float()
dx = (index % 5 - 2).float()

config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 4,
    'candidate_rule': '094 rigid-best, support-nearest >=0.5mm worse, highest-prior >=1mm worse, distinct random; logged fallbacks',
    'frozen_geometry': 'teacher 094 step20000 fitted state; only student matcher.image and matcher.atlas train',
    'loss': 'mean visible eligible query logsumexp(P union N raw logits) - logsumexp(P raw logits)',
    'positive_um': 1500, 'negative_um': 2000, 'fine_classes': 225,
    'support_threshold': .5, 'support_penalty_removed': 4.,
    'learning_rate': 1e-4, 'weight_decay': 1e-4,
    'parent_094': str(parent), 'parent_094_sha256': sha(parent),
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
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
        run / f'descriptor_step_{step:05d}.pt')


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
            truth = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                (32, 32), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            visible = F.interpolate(accepted['valid_mask'][:, None].float(),
                (32, 32), mode='bilinear', align_corners=False)[:, 0] == 1
            centre, frame, basis = full_frame_state_to_components(fitted_state)
            edges = frame[..., :, :2] @ basis
            chart_x = torch.where(reflected.bool()[..., None, None],
                                  (side - 1) / side - xx, xx)
            query_chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
            base = centre[..., None, None, :] + torch.einsum(
                'bkij,bkhwj->bkhwi', edges, query_chart - .5)
            shift_x = torch.where(reflected.bool()[..., None], -dx, dx)
            shift = torch.stack((shift_x, dy[None, None].expand_as(shift_x)), -1) / 32
            lateral = torch.einsum('bkij,bkcj->bkci', edges, shift)
            bin_ccf = (base[:, :, None] + lateral[:, :, :, None, None, :]
                + depth_um[None, None, :, None, None, None]
                * frame[..., :, 2][:, :, None, None, None, :])
            squared_distance = (bin_ccf - truth[:, None, None]).square().sum(-1)

        matched = student(prediction['feature'], fitted_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side), match_only=True)
        atlas_support = matched['fine_match_support']
        supported = atlas_support >= .5
        positive = supported & (squared_distance <= 1500 ** 2)
        negative = supported & (squared_distance >= 2000 ** 2)
        raw = matched['fine_match_logits'] + 4 * (1 - atlas_support)
        scores = raw.permute(0, 3, 4, 1, 2).reshape(-1, 4 * 225)
        positives = positive.permute(0, 3, 4, 1, 2).reshape_as(scores)
        negatives = negative.permute(0, 3, 4, 1, 2).reshape_as(scores)
        eligible = visible.flatten() & positives.any(-1) & negatives.any(-1)
        if not bool(eligible.any()):
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                                    'used': False, 'reason': 'no_contrastive_eligible_site'}) + '\n')
            step -= 1
            continue
        assert record['physical_section_id'] not in seen_sections
        seen_sections.add(record['physical_section_id'])
        draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                                'used': True}) + '\n')
        optimizer.zero_grad(set_to_none=True)
        active = scores[eligible]
        pos = positives[eligible]
        neg = negatives[eligible]
        pos_scores = active.masked_fill(~pos, -1e4)
        neg_scores = active.masked_fill(~neg, -1e4)
        loss = (torch.logsumexp(active.masked_fill(~(pos | neg), -1e4), -1)
                - torch.logsumexp(pos_scores, -1)).mean()
        margin = (pos_scores.amax(-1) - neg_scores.amax(-1)).mean()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        loss_value, margin_value, gradient_value = (
            float(loss.detach()), float(margin.detach()), float(gradient))
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'virtual_subject_id': record['virtual_subject_id'],
            'animal_id': record['base_lineage']['animal_id'],
            'subject_plan_id': record['base_lineage']['subject_deformation_plan_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': beam_error[slots].tolist(),
            'initial_support': support[slots].tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'visible_sites': int(visible.sum()), 'eligible_sites': int(eligible.sum()),
            'positive_bins': int((positive & visible[:, None, None]).sum()),
            'negative_bins': int((negative & visible[:, None, None]).sum()),
            'contrastive_loss': loss_value, 'hard_negative_margin': margin_value,
            'gradient_norm': gradient_value,
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'eligible_sites',
                'contrastive_loss', 'hard_negative_margin', 'peak_gpu_mb', 'seconds')}),
                flush=True)
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
    'checkpoint_sha256': {str(step): sha(run / f'descriptor_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}),
      flush=True)
