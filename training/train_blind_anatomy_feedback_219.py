"""TRAIN-only blind-pose anatomy feedback pilot; no public or reserved evaluation."""

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

import numpy as np
import torch
import torch.nn.functional as F

from training.anatomy_correspondence_feedback_218 import AnatomyCorrespondenceFeedback218
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
uncentered = os.environ.get('JOINT219_ABLATION') == 'uncentered'
run = root / ('runs/blind_anatomy_feedback_220_uncentered_train_001' if uncentered
    else 'runs/blind_anatomy_feedback_219_train_001')
seed, side, bank_side, updates = 20261011219, 256, 24, 300
milestones = (0, 100, 300)
assert source.drive.upper() == parent.drive.upper() == run.drive.upper() == 'I:'
assert not run.exists() and torch.cuda.is_available()
assert sha(parent) == json.loads((parent_run / 'completed.json').read_text())['checkpoint_sha256']['800']
torch.set_num_threads(4)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
assert saved['step'] == 800 and saved['calibrated'] is False
del saved
matcher = AnatomyCorrespondenceFeedback218(side=bank_side,
    center_similarity=not uncentered).cuda()
optimizer = torch.optim.AdamW(matcher.parameters(), lr=2e-4, weight_decay=1e-4)
axis = (torch.arange(bank_side, device='cuda') + .5) / bank_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
native_axis = ((torch.arange(bank_side, device='cuda') + .5) * side / bank_side - .5).round().long()
native_y, native_x = torch.meshgrid(native_axis, native_axis, indexing='ij')
native_index = (native_y * side + native_x).flatten()


def draw(index, inner=False):
    base = 60 + index % 4 if inner else (37 * index) % 60
    draw_seed = seed + (9000000 if inner else 0) + 1009 * index
    with torch.no_grad():
        source_sample = sample_panel_geometry_train_166(context, [base], draw_seed, side)
        framed = sample_support_aware_framing_217(source_sample, draw_seed)
        sample = sample_one_shot_slide_artifacts_v4(context, [base], draw_seed,
            side, source_sample=framed)
    return sample, {'draw_index': index, 'base_index': base, 'draw_seed': draw_seed,
        'source_eligible': bool(source_sample['eligible'][0]),
        'framed_eligible': bool(framed['eligible'][0]),
        'used': bool(sample['eligible'][0]), 'valid_pixels': int(sample['valid_mask'].sum()),
        'provenance': sample['provenance'][0]}


def render_bank(state, reflection, sample):
    count = len(state)
    plane = rigid_points_090(state, reflection, chart, (side, side)).reshape(
        count, bank_side, bank_side, 3)
    normal = full_frame_state_to_components(state)[1][..., :, 2]
    depths = matcher.depth_offsets_um
    offsets = sample['offsets'].expand(count, -1)
    weights = sample['weights'].expand(count, -1)
    axial = offsets[:, None, :] + depths[None, :, None]
    world = plane[:, None, None] + axial[..., None, None, None] * normal[:, None, None, None, None, :]
    rendered = render_finite_thickness_coordinate_grid(atlas, world.flatten(0, 1),
        (0., 0., 0.), (25., 25., 25.), weights[:, None].expand(-1, len(depths), -1).flatten(0, 1))
    rendered = rendered.reshape(count, len(depths), 2, bank_side, bank_side)
    support = rendered[:, :, 1:2].clamp(0, 1)
    intensity = (rendered[:, :, :1] / support.clamp_min(1e-4)).clamp(0, 1)
    keys = plane[:, None] + depths[None, :, None, None, None] * normal[:, None, None, None, :]
    return intensity, support, keys


def branches(sample):
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        state = prediction['state'][0].repeat_interleave(2, 0)
        reflection = torch.arange(2, device='cuda').repeat(model.modes)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device='cuda').long()]
        sites = torch.stack((pixels.remainder(side), pixels.div(side, rounding_mode='floor')), -1).float() / side
        truth = sample['centre'][0].reshape(-1, 3)[pixels]
        proposed = rigid_points_090(state, reflection, sites, (side, side))
        error = (proposed - truth[None]).norm(dim=-1).mean(-1) / 1000
    return state, reflection, prior, error, sites, truth


def compare(sample, state, reflection, encoded=None, return_logits=False):
    intensity, support, keys = render_bank(state, reflection, sample)
    if encoded is None:
        encoded = matcher.encode_source(sample['inputs'])
    repeated = tuple(value.expand(len(state), *value.shape[1:]) for value in encoded)
    result = matcher(sample['inputs'].expand(len(state), -1, -1, -1),
        intensity, support, keys, state, reflection, source_encoded=repeated,
        return_logits=return_logits)
    return result, intensity, support, keys


def one_case(sample, index):
    state, reflection, prior, error, sites, truth = branches(sample)
    best = int(error.argmin())
    first = int(prior.argmax())
    top = prior.topk(16).indices
    wrong = top[(error[top] > error[best] + .8).nonzero()[:, 0]]
    wrong = int(wrong[0]) if len(wrong) else int(error.argmax())
    chosen = torch.tensor([best, first, wrong], device='cuda')
    candidate_state = torch.cat((sample['state'], state[chosen]))
    candidate_reflection = torch.cat((sample['reflection'], reflection[chosen]))
    encoded = matcher.encode_source(sample['inputs'])
    result, intensity, support, keys = compare(sample, candidate_state,
        candidate_reflection, encoded, return_logits=True)
    scores = result['anatomy_logit']

    target = sample['centre'][0].reshape(-1, 3)[native_index]
    valid = sample['valid_mask'][0].flatten()[native_index]
    distance = torch.cdist(target[None], keys[:1].reshape(1, -1, 3))[0]
    admissible = result['correspondence_valid'][0]
    distance = distance.masked_fill(~admissible, float('inf'))
    nearest_distance, label = distance.min(-1)
    positive = valid & (nearest_distance < 650)
    negative = ~valid | (nearest_distance > 1200)
    logits = result['correspondence_logits'][0]
    ce = (F.cross_entropy(logits[positive], label[positive]) if positive.any()
          else scores.new_zeros(()))
    probability = result['match_probability'][0].flatten().clamp(1e-5, 1 - 1e-5)
    yes = -probability[positive].log().mean() if positive.any() else scores.new_zeros(())
    no = -(1 - probability[negative]).log().mean() if negative.any() else scores.new_zeros(())
    source_tissue = sample['visible'][0].flatten()[native_index] > .25
    reliability = result['source_weight'][0].flatten().clamp(1e-5, 1 - 1e-5)
    tissue_loss = (-reliability[source_tissue].log().mean()
        if source_tissue.any() else scores.new_zeros(()))
    background_loss = (-(1 - reliability[~source_tissue]).log().mean()
        if (~source_tissue).any() else scores.new_zeros(()))
    rank = F.softplus(.3 + scores[3] - scores[1]) + F.softplus(.3 + scores[3] - scores[0])
    if first != best and float(error[first] - error[best]) > .5:
        rank = rank + F.softplus(.2 + scores[2] - scores[1])
    fitted = rigid_points_090(result['fitted_state'][1:2],
        candidate_reflection[1:2], sites, (side, side))[0]
    fit_mm = (fitted - truth).norm(dim=-1).mean() / 1000
    stay = rigid_points_090(result['fitted_state'][:1],
        candidate_reflection[:1], sites, (side, side))[0]
    original = rigid_points_090(sample['state'], sample['reflection'], sites, (side, side))[0]
    stay_mm = (stay - original).norm(dim=-1).mean() / 1000
    common = ((support[:1] > .5) & (support[3:4] > .5)).float()
    amount = common.sum((2, 3, 4)).clamp_min(1)
    true_mean = (intensity[:1] * common).sum((2, 3, 4)) / amount
    wrong_mean = (intensity[3:4] * common).sum((2, 3, 4)) / amount
    true_std = (((intensity[:1] - true_mean[:, :, None, None, None]).square() * common
        ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    wrong_std = (((intensity[3:4] - wrong_mean[:, :, None, None, None]).square() * common
        ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    matched_wrong = ((intensity[3:4] - wrong_mean[:, :, None, None, None])
        * true_std[:, :, None, None, None] / wrong_std[:, :, None, None, None]
        + true_mean[:, :, None, None, None]).clamp(0, 1)
    changed = torch.where(common.bool(), matched_wrong, intensity[:1])
    fake = matcher(sample['inputs'], changed, support[:1], keys[:1],
        sample['state'], sample['reflection'], source_encoded=encoded)['anatomy_logit'][0]
    counterfactual = (F.softplus(.15 + fake - scores[0])
        if int(common[:, len(matcher.depth_offsets_um) // 2].sum()) >= 12
        else scores.new_zeros(()))
    loss = (.12 * ce + .35 * (yes + no) + .15 * (tissue_loss + background_loss)
        + rank + .25 * counterfactual + .2 * fit_mm + .05 * stay_mm)
    return loss, {'ce': float(ce.detach()), 'positive_sites': int(positive.sum()),
        'negative_sites': int(negative.sum()), 'match_yes': float(yes.detach()),
        'match_no': float(no.detach()), 'rank': float(rank.detach()),
        'tissue_loss': float(tissue_loss.detach()),
        'background_loss': float(background_loss.detach()),
        'counterfactual': float(counterfactual.detach()),
        'blind_prior_mm': float(error[first]), 'blind_best_of_160_mm': float(error[best]),
        'blind_prior_is_best': first == best, 'fit_before_mm': float(error[best]),
        'fit_after_mm': float(fit_mm.detach()), 'truth_stay_mm': float(stay_mm.detach()),
        'truth_score': float(scores[0].detach()),
        'best_score': float(scores[1].detach()), 'wrong_score': float(scores[3].detach()),
        'loss': float(loss.detach())}


def inner_readout(step):
    matcher.eval()
    cases = []
    with torch.no_grad():
        for index in range(8):
            sample, draw_record = draw(index, inner=True)
            if not draw_record['used']:
                cases.append({**draw_record, 'scored': False})
                continue
            state, reflection, prior, error, sites, truth = branches(sample)
            encoded = matcher.encode_source(sample['inputs'])
            scores, fitted_error = [], []
            for start in range(0, len(state), 4):
                stop = start + 4
                result, _, _, _ = compare(sample, state[start:stop],
                    reflection[start:stop], encoded)
                scores.append(result['anatomy_logit'])
                fitted = rigid_points_090(result['fitted_state'],
                    reflection[start:stop], sites, (side, side))
                fitted_error.append((fitted - truth[None]).norm(dim=-1).mean(-1) / 1000)
            scores = torch.cat(scores)
            fitted_error = torch.cat(fitted_error)
            prior_index = int(prior.argmax())
            anatomy_index = int(scores.argmax())
            fused_index = int((scores + .3 * prior).argmax())
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            truth_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            angle = torch.rad2deg(torch.acos((normal @ truth_normal).abs().clamp_max(1)))
            cases.append({**draw_record, 'scored': True,
                'visible_fraction': float(sample['valid_mask'].float().mean()),
                'prior_mm': float(error[prior_index]),
                'prior_axis_deg': float(angle[prior_index]),
                'anatomy_mm': float(error[anatomy_index]),
                'anatomy_axis_deg': float(angle[anatomy_index]),
                'fused_mm': float(error[fused_index]),
                'fused_axis_deg': float(angle[fused_index]),
                'fused_after_fit_mm': float(fitted_error[fused_index]),
                'best_of_160_mm': float(error.min()),
                'best_of_160_axis_deg': float(angle[error.argmin()]),
                'top10_recall_2mm': bool((error[(scores + .3 * prior).topk(10).indices] < 2).any())})
    scored = [case for case in cases if case['scored']]
    summary = {'step': step, 'scored_n': len(scored),
        'prior_median_mm': float(np.median([case['prior_mm'] for case in scored])),
        'anatomy_median_mm': float(np.median([case['anatomy_mm'] for case in scored])),
        'fused_median_mm': float(np.median([case['fused_mm'] for case in scored])),
        'prior_axis_median_deg': float(np.median([case['prior_axis_deg'] for case in scored])),
        'fused_axis_median_deg': float(np.median([case['fused_axis_deg'] for case in scored])),
        'fused_after_fit_median_mm': float(np.median([case['fused_after_fit_mm'] for case in scored])),
        'best_of_160_median_mm': float(np.median([case['best_of_160_mm'] for case in scored])),
        'top10_recall_2mm': float(np.mean([case['top10_recall_2mm'] for case in scored]))}
    (run / f'inner_step_{step:05d}.json').write_text(json.dumps(
        {'summary': summary, 'cases': cases}, allow_nan=False))
    path = run / f'joint_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'matcher': matcher.state_dict(),
        'optimizer': optimizer.state_dict(), 'calibrated': False}, temporary)
    os.replace(temporary, path)
    print(json.dumps({'event': 'inner_checkpoint', **summary}), flush=True)
    matcher.train()


config = {'seed': seed, 'updates': updates, 'milestones': milestones,
    'center_similarity': not uncentered,
    'parent_checkpoint': str(parent), 'parent_sha256': sha(parent),
    'source_sha256': {name: sha(source.with_name(name)) for name in (
        source.name, 'anatomy_correspondence_feedback_218.py',
        'support_aware_framing_217.py', 'arbitrary_plane_panel_geometry_train_166.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py')},
    'synthetic_provenance': context['provenance'],
    'train_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'selection': 'full 160-branch blind ranking on 8 inner TRAIN-only draws',
    'external_pretrained_weights': False, 'posterior_calibrated': False,
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False}
if os.environ.get('JOINT219_MODE') == 'preflight':
    for preflight_index in range(10):
        preflight_sample, preflight_record = draw(preflight_index)
        if preflight_record['used']:
            break
    assert preflight_record['used']
    preflight_loss, preflight_measures = one_case(preflight_sample, preflight_index)
    preflight_loss.backward()
    print(json.dumps({'event': 'preflight', 'draw': preflight_record,
        'gradient_norm': float(torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5.)),
        **preflight_measures}, allow_nan=False), flush=True)
    sys.exit(0)
assert not run.exists()
run.mkdir(parents=True)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
started = time.perf_counter()
inner_readout(0)
accepted = 0
with (run / 'training.jsonl').open('w') as stream:
    for index in range(1000):
        sample, draw_record = draw(index)
        if not draw_record['used']:
            stream.write(json.dumps({**draw_record, 'update': None}, allow_nan=False) + '\n')
            continue
        optimizer.zero_grad(set_to_none=True)
        loss, measures = one_case(sample, index)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        accepted += 1
        stream.write(json.dumps({**draw_record, 'update': accepted,
            'gradient_norm': float(gradient), **measures}, allow_nan=False) + '\n')
        if accepted == 1 or accepted % 25 == 0:
            stream.flush()
            print(json.dumps({'event': 'train_milestone', 'attempt': index + 1,
                'update': accepted, **measures}), flush=True)
        if accepted in milestones:
            inner_readout(accepted)
        if accepted == updates:
            break

done = {'updates': accepted, 'attempts': index + 1,
    'elapsed_seconds': time.perf_counter() - started,
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in milestones},
    'output_sha256': {path.name: sha(path) for path in run.iterdir()
        if path.is_file() and path.name != 'completed.json'},
    'calibrated': False, 'reserved_dev_cal_final_used': False,
    'public_benchmark_used': False}
(run / 'completed.json').write_text(json.dumps(done, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', **done}), flush=True)
