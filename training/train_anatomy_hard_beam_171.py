"""Bounded TRAIN-only hard-beam test of atlas-correspondence feedback."""

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
import torch.nn.functional as F

from training.anatomy_correspondence_feedback_168 import AnatomyCorrespondenceFeedback
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ANATOMY_HARD_BEAM_171_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
previous = root / 'runs/anatomy_correspondence_169_train_001'
previous_path = previous / 'joint_step_06000.pt'
out = root / 'runs/anatomy_hard_beam_171_train_001'
seed, panel_seed, side, bank_side = 202610111710, 202610111711, 256, 48
panel_attempts, train_attempts, final_updates = 128, 2200, 1600
checkpoints = (0, 400, 800, 1600)
assert source.drive.upper() == out.drive.upper() == 'I:'
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

previous_done = json.loads((previous / 'completed.json').read_text())
previous_config = json.loads((previous / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert previous_done['optimizer_updates'] == 6000
assert sha(previous_path) == previous_done['checkpoint_sha256']['6000']
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000'] == previous_config['parent_checkpoint_sha256']
assert all(sha(source / name) == digest for name, digest in previous_config['source_sha256'].items())
assert seed != previous_config['seed'] and panel_seed != seed

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == previous_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    previous_model = torch.load(previous_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
assert previous_model['step'] == 6000 and previous_model['calibrated'] is False
assert all(torch.equal(previous_model['model'][key], value)
           for key, value in parent['model'].items())
model.load_state_dict(parent['model'], strict=True)
del parent
matcher = AnatomyCorrespondenceFeedback(side=bank_side, radius=3).cuda()
prefix = 'anatomy_correspondence_168.'
matcher.load_state_dict({key[len(prefix):]: value for key, value
    in previous_model['model'].items() if key.startswith(prefix)}, strict=True)
del previous_model
optimizer = torch.optim.AdamW(matcher.parameters(), lr=1e-4, weight_decay=1e-4)
depths = torch.tensor([-500., -250., 0., 250., 500.], device='cuda')
ax = (torch.arange(bank_side, device='cuda') + .5) / bank_side - .5 / side
yy, xx = torch.meshgrid(ax, ax, indexing='ij')
chart_bank = torch.stack((xx, yy), -1).reshape(-1, 2)


def draw(index, inner=False):
    rng = np.random.default_rng(np.random.SeedSequence([panel_seed if inner else seed, index]))
    base = int(rng.integers(60, 64) if inner else rng.integers(0, 60))
    geometry = sample_panel_geometry_train_151 if rng.random() < .2 else sample_panel_geometry_train_166
    stratum = ('heavy_oblique_45_55' if inner and index >= 96 else
               'unconditional_inner' if inner else 'unconditional_train')
    trials = 0
    while True:
        draw_seed = int(rng.integers(0, 2**31 - 1))
        trials += 1
        if stratum != 'heavy_oblique_45_55':
            break
        orientation = np.random.default_rng(np.random.SeedSequence(
            [draw_seed, base, 0, 0])).normal(size=3)
        orientation /= np.linalg.norm(orientation)
        if math.degrees(math.acos(float(np.abs(orientation).max()))) >= 45:
            break
    with torch.no_grad():
        original = geometry(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(context, [base], draw_seed,
            side, source_sample=original)
    return sample, {'index': index, 'base_index': base, 'draw_seed': draw_seed,
        'geometry': geometry.__name__, 'valid_pixels': int(sample['valid_mask'].sum()),
        'panel_stratum': stratum, 'orientation_seed_trials': trials,
        'provenance': sample['provenance'][0]}


def beam_for(prediction):
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
    return prior, beam


def render_bank(state, reflection, offsets, weights):
    count = len(state)
    plane = rigid_points_090(state, reflection, chart_bank, (side, side)).reshape(
        count, bank_side, bank_side, 3)
    normal = full_frame_state_to_components(state)[1][..., :, 2]
    axial = offsets.expand(count, -1)[:, None, :] + depths[None, :, None]
    coords = (plane[:, None, None] + axial[:, :, :, None, None, None]
              * normal[:, None, None, None, None, :])
    rendered = render_finite_thickness_coordinate_grid(atlas,
        coords.flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
        weights.expand(count, -1)[:, None].expand(-1, len(depths), -1).flatten(0, 1))
    rendered = rendered.reshape(count, len(depths), 2, bank_side, bank_side)
    support = rendered[:, :, 1:2].clamp(0, 1)
    return rendered[:, :, :1] / support.clamp_min(1e-4), support


def counterfactual(true_intensity, true_support, wrong_intensity, wrong_support):
    common = ((true_support > .5) & (wrong_support > .5)).float()
    interior = (F.avg_pool2d(common.flatten(0, 1), 7, 1, 3)
                .reshape_as(common) > .999).float()
    amount = interior.sum((2, 3, 4)).clamp_min(1)
    mean_t = (true_intensity * interior).sum((2, 3, 4)) / amount
    mean_w = (wrong_intensity * interior).sum((2, 3, 4)) / amount
    std_t = (((true_intensity - mean_t[:, :, None, None, None]).square() * interior
              ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    std_w = (((wrong_intensity - mean_w[:, :, None, None, None]).square() * interior
              ).sum((2, 3, 4)) / amount + 1e-4).sqrt()
    matched = ((wrong_intensity - mean_w[:, :, None, None, None])
               * std_t[:, :, None, None, None] / std_w[:, :, None, None, None]
               + mean_t[:, :, None, None, None]).clamp(0, 1)
    return torch.where(interior.bool(), matched, true_intensity), interior


def chart_for(valid, count):
    pixels = valid[0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(count, len(pixels)),
        device=pixels.device).long()]
    return torch.stack((pixels.remainder(side),
        pixels.div(side, rounding_mode='floor')), -1).float() / side


def train_case(sample, index):
    # The beam and prior are fixed without touching section truth.
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        states = prediction['state'][0, ids // 2].detach()
        flags = ids.remainder(2)
        source_for_mining = matcher.encode_source(sample['inputs'])
        mine_scores = []
        for start in range(0, 16, 4):
            candidate = states[start:start + 4]
            candidate_flags = flags[start:start + 4]
            intensity, support = render_bank(candidate, candidate_flags,
                sample['offsets'], sample['weights'])
            source = source_for_mining.expand(len(candidate), -1, -1, -1)
            first = matcher.compare(source, matcher.encode_atlas(intensity), support, depths)
            fitted = compose_full_frame_state(candidate, first['update'])
            fitted_intensity, fitted_support = render_bank(fitted, candidate_flags,
                sample['offsets'], sample['weights'])
            second = matcher.compare(source, matcher.encode_atlas(fitted_intensity),
                fitted_support, depths)
            effort = .04 * (first['update'] / matcher.update_limits).square().mean(-1)
            mine_scores.append(second['score'] - effort)
        mine_scores = torch.cat(mine_scores)
        # The blind beam and its anatomy scores now exist; truth labels follow.
        chart = chart_for(sample['valid_mask'], 128)
        truth = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
        errors = (rigid_points_090(states, flags, chart) - truth).norm(dim=-1).mean(-1) / 1000
        best = int(errors.argmin())
        prior_slot = int(prior[0, ids].argmax())
        wrong = (errors >= errors[best] + .75).nonzero()[:, 0]
        if len(wrong) == 0:
            wrong = errors.argmax().reshape(1)
        hard = int(wrong[prior[0, ids[wrong]].argmax()])
        anatomy_hard = int(wrong[mine_scores[wrong].argmax()])
        near = int(wrong[errors[wrong].argmin()])
        selected = list(dict.fromkeys((best, prior_slot, hard, anatomy_hard, near)))[:4]
        for slot in prior[0, ids].argsort(descending=True):
            if len(selected) == 4:
                break
            if int(slot) not in selected:
                selected.append(int(slot))
        initial = errors[selected]
        candidate = states[selected]
        reflection = flags[selected]
        selected_prior = prior[0, ids[selected]]
    source_desc = matcher.encode_source(sample['inputs'])
    intensity, support = render_bank(candidate, reflection,
        sample['offsets'], sample['weights'])
    first = matcher.compare(source_desc.expand(len(selected), -1, -1, -1),
        matcher.encode_atlas(intensity), support, depths)
    fitted = compose_full_frame_state(candidate, first['update'])
    fitted_intensity, fitted_support = render_bank(fitted, reflection,
        sample['offsets'], sample['weights'])
    second = matcher.compare(source_desc.expand(len(selected), -1, -1, -1),
        matcher.encode_atlas(fitted_intensity), fitted_support, depths)
    effort = (first['update'] / matcher.update_limits).square().mean(-1)
    score = second['score'] - .04 * effort
    target = (-initial / .5).softmax(0)
    rank_loss = -(target * F.log_softmax(selected_prior - selected_prior.max()
        + score / .02, dim=0)).sum()
    best_after = rigid_points_090(fitted[:1], reflection[:1], chart)
    correction_mm = (best_after - truth[None]).norm(dim=-1).mean() / 1000
    with torch.no_grad():
        true_intensity, true_support = render_bank(sample['state'],
            sample['reflection'], sample['offsets'], sample['weights'])
        shift = torch.zeros((1, 9), device='cuda')
        shift[0, 5] = 1200. if index % 2 else -1200.
        wrong_state = compose_full_frame_state(sample['state'], shift)
        wrong_intensity, wrong_support = render_bank(wrong_state,
            sample['reflection'], sample['offsets'], sample['weights'])
        changed, common = counterfactual(true_intensity, true_support,
            wrong_intensity, wrong_support)
    common_sites = int(common[:, len(depths) // 2].sum())
    if common_sites >= 12:
        true_score = matcher.compare(source_desc,
            matcher.encode_atlas(true_intensity), common, depths)['score']
        wrong_score = matcher.compare(source_desc,
            matcher.encode_atlas(changed), common, depths)['score']
        cf_loss = F.softplus(.06 + wrong_score - true_score).mean()
    else:
        cf_loss = rank_loss.new_zeros(())
    loss = rank_loss + .2 * cf_loss + .2 * correction_mm + .02 * effort.mean()
    return loss, {'rank_loss': float(rank_loss.detach()),
        'correction_mm': float(correction_mm.detach()),
        'best_initial_mm': float(initial[0]), 'cf_sites': common_sites,
        'cf_loss': float(cf_loss.detach()), 'selected_slots': selected,
        'selected_branch_ids': ids[selected].cpu().tolist(),
        'selected_initial_mm': initial.cpu().tolist(),
        'selected_support_fraction': support[:, len(depths) // 2].mean((1, 2, 3)).detach().cpu().tolist(),
        'loss': float(loss.detach())}


def freeze_case(sample, record):
    item = dict(record)
    if not record['valid_pixels']:
        return item
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        chosen = ids // 2
        flags = (ids % 2)[None]
        states = prediction['state'][:, chosen]
        native_score, native_state = [], []
        for start in range(0, 16, 4):
            fit = global_plane_match(model, prediction, chosen[None, start:start + 4],
                flags[:, start:start + 4], sample['offsets'], sample['weights'],
                atlas, (side, side), side=24)
            action_score = torch.stack((fit['input_score'][0], fit['score'][0]), -1)
            action = action_score.argmax(-1)
            native_score.append(action_score.max(-1).values)
            native_state.append(torch.where(action[:, None] == 0,
                fit['input_state'][0], fit['state'][0]))
        # All candidate identities, prior scores and native fitter actions are frozen.
        chart = chart_for(sample['valid_mask'], 1024)
        item.update({'inputs': sample['inputs'][0].cpu(),
            'offsets': sample['offsets'][0].cpu(), 'weights': sample['weights'][0].cpu(),
            'branch_ids': ids.cpu(), 'reflection': flags[0].cpu(),
            'initial_state': states[0].cpu(), 'prior_score': prior[0, ids].cpu(),
            'native_score': torch.cat(native_score).cpu(),
            'native_state': torch.cat(native_state).cpu(), 'chart': chart.cpu(),
            'truth_state': sample['state'][0].cpu(),
            'truth_reflection': sample['reflection'][0].cpu()})
    return item


def evaluate_case(item):
    record = {key: item[key] for key in ('index', 'base_index', 'draw_seed',
        'geometry', 'valid_pixels', 'panel_stratum', 'orientation_seed_trials',
        'provenance')}
    record['scored'] = 'inputs' in item
    if not record['scored']:
        return record
    image = item['inputs'][None].cuda()
    source_desc = matcher.encode_source(image)
    states = item['initial_state'].cuda()
    reflection = item['reflection'].cuda()
    offsets = item['offsets'][None].cuda()
    weights = item['weights'][None].cuda()
    scores = {name: [] for name in ('anatomy0', 'anatomy1', 'null0', 'null1')}
    corrected, support_area = [], []
    for start in range(0, 16, 4):
        state = states[start:start + 4]
        flags = reflection[start:start + 4]
        intensity, support = render_bank(state, flags, offsets, weights)
        source = source_desc.expand(len(state), -1, -1, -1)
        first = matcher.compare(source, matcher.encode_atlas(intensity), support, depths)
        null_first = matcher.compare(source,
            matcher.encode_atlas(torch.zeros_like(intensity)), support, depths)
        fitted = compose_full_frame_state(state, first['update'])
        fitted_intensity, fitted_support = render_bank(fitted, flags, offsets, weights)
        second = matcher.compare(source,
            matcher.encode_atlas(fitted_intensity), fitted_support, depths)
        null_second = matcher.compare(source,
            matcher.encode_atlas(torch.zeros_like(fitted_intensity)), fitted_support, depths)
        effort = .04 * (first['update'] / matcher.update_limits).square().mean(-1)
        scores['anatomy0'].append(first['score'])
        scores['anatomy1'].append(second['score'] - effort)
        scores['null0'].append(null_first['score'])
        scores['null1'].append(null_second['score'] - effort)
        corrected.append(fitted)
        support_area.append(support[:, len(depths) // 2].mean((1, 2, 3)))
    scores = {name: torch.cat(values).cpu().tolist() for name, values in scores.items()}
    prior = item['prior_score'].tolist()
    scores.update({'prior': prior, 'native121': item['native_score'].tolist(),
        'fused1': (item['prior_score'] + torch.tensor(scores['anatomy1']) / .02).tolist(),
        'fused_null1': (item['prior_score'] + torch.tensor(scores['null1']) / .02).tolist()})
    selected = {name: int(np.argmax(value)) for name, value in scores.items()}
    # Plane truth is read only after every inference score and chosen slot is fixed.
    chart = item['chart'].cuda()
    truth = rigid_points_090(item['truth_state'][None].cuda(),
        item['truth_reflection'][None].cuda(), chart)[0]
    initial_mm = ((rigid_points_090(states, reflection, chart) - truth[None]
        ).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
    anatomy_mm = ((rigid_points_090(torch.cat(corrected), reflection, chart)
        - truth[None]).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
    native_mm = ((rigid_points_090(item['native_state'].cuda(), reflection, chart)
        - truth[None]).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
    provenance = item['provenance']
    normal = np.abs(provenance['physical_unit_normal_ap_dv_ml'])
    angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
    record.update({'animal_id': provenance['base_lineage']['animal_id'],
        'specimen_id': provenance['base_lineage']['specimen_id'],
        'experiment_id': provenance['base_lineage']['experiment_id'],
        'physical_section_id': provenance['physical_section_id'],
        'appearance_mode': provenance['mode'],
        'nearest_cardinal_angle_deg': angle,
        'valid_target_fraction': item['valid_pixels'] / (side * side),
        'candidate_atlas_support_fraction': torch.cat(support_area).cpu().tolist(),
        'branch_ids': item['branch_ids'].tolist(),
        'candidate_reflection': reflection.cpu().tolist(),
        'truth_reflection': int(item['truth_reflection']),
        'scores': scores, 'chosen_slot': selected,
        'initial_rigid_mm': initial_mm, 'anatomy_fitted_rigid_mm': anatomy_mm,
        'native_fitted_rigid_mm': native_mm})
    return record


def readout(group):
    names = ('prior', 'native121', 'anatomy0', 'anatomy1', 'null0', 'null1',
             'fused1', 'fused_null1')
    result = {'sections': len(group)}
    for name in names:
        errors = np.asarray([row['initial_rigid_mm'][row['chosen_slot'][name]] for row in group])
        result[name] = {'mean_initial_mm': float(errors.mean()) if len(errors) else None,
            'capture_le_1p5': float((errors <= 1.5).mean()) if len(errors) else None}
    oracle = np.asarray([min(row['initial_rigid_mm']) for row in group])
    result['oracle16'] = {'mean_initial_mm': float(oracle.mean()) if len(oracle) else None,
        'capture_le_1p5': float((oracle <= 1.5).mean()) if len(oracle) else None}
    result['selected_fitted_mean_mm'] = {name: float(np.mean([
        row['anatomy_fitted_rigid_mm'][row['chosen_slot'][name]] for row in group]))
        if group else None for name in ('anatomy1', 'fused1')}
    result['selected_fitted_mean_mm']['native121'] = float(np.mean([
        row['native_fitted_rigid_mm'][row['chosen_slot']['native121']]
        for row in group])) if group else None
    return result


def evaluate_panel(panel, step):
    matcher.eval()
    with torch.inference_mode(), (out / f'inner_step_{step:05d}.jsonl').open('w') as stream:
        rows = []
        for item in panel:
            row = evaluate_case(item)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    scored = [row for row in rows if row['scored']]
    summary = {'step': step, 'all': readout(scored),
        'panel_stratum': {key: readout([row for row in scored if row['panel_stratum'] == key])
            for key in ('unconditional_inner', 'heavy_oblique_45_55')},
        'valid_target_fraction': {key: readout([row for row in scored if low <= row['valid_target_fraction'] < high])
            for key, low, high in (('0-.10', 0, .10), ('.10-.25', .10, .25),
                                   ('.25-.50', .25, .50), ('.50-1', .50, 1.01))},
        'angle': {key: readout([row for row in scored if low <= row['nearest_cardinal_angle_deg'] < high])
            for key, low, high in (('0-15', 0, 15), ('15-30', 15, 30),
                                   ('30-45', 30, 45), ('45-55', 45, 55))},
        'appearance': {key: readout([row for row in scored if row['appearance_mode'] == key])
            for key in ('raw', 'exact_black', 'imperfect_brush')}}
    (out / f'inner_step_{step:05d}.json').write_text(json.dumps(summary, indent=2,
        allow_nan=False))
    path = out / f'matcher_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'matcher': matcher.state_dict(),
        'optimizer': optimizer.state_dict(), 'config_sha256': sha(out / 'config.json'),
        'calibrated': False}, temp)
    os.replace(temp, path)
    print(json.dumps({'event': 'inner_checkpoint', 'step': step,
        'all': summary['all']}, allow_nan=False), flush=True)
    matcher.train()


if os.environ.get('JOINT171_MODE') == 'preflight':
    sample, record = draw(99999999)
    assert record['valid_pixels']
    loss, measures = train_case(sample, 99999999)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5.,
        error_if_nonfinite=True)
    frozen = freeze_case(sample, record)
    matcher.eval()
    with torch.inference_mode():
        evaluated = evaluate_case(frozen)
    oblique_sample, oblique_record = draw(96, inner=True)
    normal = np.abs(oblique_record['provenance']['physical_unit_normal_ap_dv_ml'])
    oblique_angle = math.degrees(math.acos(float(np.max(normal))))
    assert oblique_angle >= 45
    print(json.dumps({'event': 'preflight', 'gradient_norm': float(gradient),
        'train': measures, 'blind16': {key: evaluated[key] for key in
            ('scored', 'chosen_slot', 'initial_rigid_mm')},
        'oblique_inner': {'angle_deg': oblique_angle,
            'orientation_seed_trials': oblique_record['orientation_seed_trials'],
            'valid_pixels': oblique_record['valid_pixels']},
        'source_sha256': sha(__file__)}, allow_nan=False), flush=True)
    sys.exit(0)

assert not out.exists()
out.mkdir(parents=True, exist_ok=False)
panel = []
for index in range(panel_attempts):
    sample, record = draw(index, inner=True)
    panel.append(freeze_case(sample, record))
    if (index + 1) % 32 == 0:
        print(json.dumps({'event': 'panel_frozen', 'attempts': index + 1,
            'nonblank': sum('inputs' in row for row in panel)}), flush=True)
panel_path = out / 'panel.pt'
temp = panel_path.with_suffix('.tmp')
torch.save(panel, temp)
os.replace(temp, panel_path)
config = {'scope': 'TRAIN-only hard model-proposed blind-beam diagnostic',
    'seed': seed, 'panel_seed': panel_seed, 'panel_attempts': panel_attempts,
    'panel_nonblank': sum('inputs' in row for row in panel),
    'train_attempts': train_attempts, 'final_updates': final_updates,
    'checkpoints': checkpoints, 'train_bases': list(range(60)),
    'inner_bases': list(range(60, 64)),
    'panel_strata': '96 unconditional draws plus 32 independently oriented 45-55 degree draws; angle-only seed conditioning, blanks retained',
    'geometry': 'independent 166 80% / 151 20% plus v4 artifacts; no blank retry',
    'beam_policy': 'frozen 121 blind16; no truth before proposal or native score',
    'loss_policy': 'four hard beam candidates; prior+anatomy/.02 listwise target softmax(-initial_mm/.5), same-support CF, best-candidate correction',
    'score_policy': 'anatomy0/1, prior+anatomy1/.02, zero-intensity score-stage nulls; fixed blind16',
    'source_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
    'previous_config_sha256': sha(previous / 'config.json'),
    'previous_completion_sha256': sha(previous / 'completed.json'),
    'previous_matcher_checkpoint_sha256': sha(previous_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_path), 'panel_sha256': sha(panel_path),
    'synthetic_provenance': context['provenance'],
    'no_external_pretrained_weights': True, 'posterior_calibrated': False,
    'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

started = time.perf_counter()
updates = 0
evaluate_panel(panel, 0)
with (out / 'training.jsonl').open('w') as stream:
    for index in range(train_attempts):
        sample, record = draw(index)
        if not record['valid_pixels']:
            stream.write(json.dumps({**record, 'optimizer_update': None,
                'skip_reason': 'blank_plane'}, allow_nan=False) + '\n')
            continue
        optimizer.zero_grad(set_to_none=True)
        loss, measures = train_case(sample, index)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(matcher.parameters(), 5.,
            error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        stream.write(json.dumps({**record, 'optimizer_update': updates,
            'gradient_norm': float(gradient), **measures}, allow_nan=False) + '\n')
        if updates % 100 == 0:
            stream.flush()
            print(json.dumps({'event': 'update', 'updates': updates,
                'attempts_seen': index + 1, **measures}, allow_nan=False), flush=True)
        if updates in checkpoints:
            evaluate_panel(panel, updates)
        if updates == final_updates:
            break
if updates != final_updates:
    failure = {'status': 'insufficient_nonblank_attempts',
        'attempts_seen': train_attempts, 'optimizer_updates': updates,
        'required_optimizer_updates': final_updates,
        'config_sha256': sha(out / 'config.json'),
        'panel_sha256': sha(panel_path),
        'training_sha256': sha(out / 'training.jsonl')}
    (out / 'failed.json').write_text(json.dumps(failure, indent=2, allow_nan=False))
    print(json.dumps({'event': 'failed', **failure}, allow_nan=False), flush=True)
    sys.exit(2)
done = {'optimizer_updates': updates, 'attempts_seen': index + 1,
    'elapsed_seconds': time.perf_counter() - started,
    'output_sha256': {path.name: sha(path) for path in out.iterdir()
                      if path.is_file() and path.name != 'completed.json'},
    'posterior_calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}
(out / 'completed.json').write_text(json.dumps(done, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', **done}, allow_nan=False), flush=True)
