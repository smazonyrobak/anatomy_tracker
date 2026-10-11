"""Fresh TRAIN-only blind-16 candidate ranking of frozen 169 checkpoints."""

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
protocol = source.parent / 'docs/publication/ANATOMY_CORRESPONDENCE_170_BLIND_TRAIN_PROTOCOL_20261011.md'
train = root / 'runs/anatomy_correspondence_169_train_001'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/anatomy_correspondence_170_blind_train_002'
steps = (0, 1500, 3000, 4500, 6000)
seed, attempts, side, bank_side = 202610121700, 64, 256, 48
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

done = json.loads((train / 'completed.json').read_text())
training_config = json.loads((train / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert done['optimizer_updates'] == 6000
assert tuple(training_config['checkpoints']) == steps
assert (seed - training_config['seed']) % 1009 != 0
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000'] == training_config['parent_checkpoint_sha256']
assert all(sha(train / name) == digest for name, digest in done['output_sha256'].items())
assert all(sha(source / name) == digest for name, digest in training_config['source_sha256'].items())
checkpoints = {step: train / f'joint_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == done['checkpoint_sha256'][str(step)] for step, path in checkpoints.items())

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == training_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
parent_state = saved['model']
del saved

depths = torch.tensor([-500., -250., 0., 250., 500.], device='cuda')
ax = (torch.arange(bank_side, device='cuda') + .5) / bank_side - .5 / side
yy, xx = torch.meshgrid(ax, ax, indexing='ij')
chart_bank = torch.stack((xx, yy), -1).reshape(-1, 2)


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


def readout(group):
    names = ('prior', 'native121', 'anatomy0', 'anatomy1', 'null0', 'null1')
    result = {'sections': len(group)}
    for name in names:
        errors = [row['initial_rigid_mm'][row['chosen_slot'][name]] for row in group]
        result[name] = {'mean_initial_mm': float(np.mean(errors)) if errors else None,
            'capture_le_1p5': float(np.mean(np.asarray(errors) <= 1.5)) if errors else None}
    result['oracle16'] = {'mean_initial_mm': float(np.mean([min(row['initial_rigid_mm'])
        for row in group])) if group else None,
        'capture_le_1p5': float(np.mean([min(row['initial_rigid_mm']) <= 1.5
        for row in group])) if group else None}
    result['anatomy1_fitted_mean_mm'] = float(np.mean([
        row['anatomy_fitted_rigid_mm'][row['chosen_slot']['anatomy1']]
        for row in group])) if group else None
    result['native121_fitted_mean_mm'] = float(np.mean([
        row['native_fitted_rigid_mm'][row['chosen_slot']['native121']]
        for row in group])) if group else None
    result['anatomy1_minus_native121_fitted_gain_mm'] = float(np.mean([
        row['native_fitted_rigid_mm'][row['chosen_slot']['native121']]
        - row['anatomy_fitted_rigid_mm'][row['chosen_slot']['anatomy1']]
        for row in group])) if group else None
    result['paired_gain_mm'] = {f'anatomy1_minus_{name}': float(np.mean([
        row['initial_rigid_mm'][row['chosen_slot'][name]]
        - row['initial_rigid_mm'][row['chosen_slot']['anatomy1']]
        for row in group])) if group else None
        for name in ('prior', 'native121', 'anatomy0', 'null1')}
    result['paired_capture_gain'] = {f'anatomy1_minus_{name}': float(np.mean([
        int(row['initial_rigid_mm'][row['chosen_slot']['anatomy1']] <= 1.5)
        - int(row['initial_rigid_mm'][row['chosen_slot'][name]] <= 1.5)
        for row in group])) if group else None
        for name in ('prior', 'native121', 'anatomy0', 'null1')}
    return result


# One fixed draw panel, never replenished for blank sections. No truth enters the beam or fitter.
panel = []
with torch.inference_mode():
    for index in range(attempts):
        base = (37 * index + 17) % 60
        draw_seed = seed + 1009 * index
        geometry = sample_panel_geometry_train_151 if index % 5 == 0 else sample_panel_geometry_train_166
        original = geometry(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(context, [base], draw_seed,
            side, source_sample=original)
        provenance = sample['provenance'][0]
        valid = sample['valid_mask'][0].flatten().bool()
        item = {'index': index, 'base_index': base, 'draw_seed': draw_seed,
            'geometry': geometry.__name__, 'provenance': provenance,
            'valid_pixels': int(valid.sum())}
        if not item['valid_pixels']:
            panel.append(item)
            continue
        image = sample['inputs']
        prediction = model.predict(image)
        prior, beam = beam_for(prediction)
        ids = beam[0]
        flags = (ids % 2)[None]
        chosen = ids // 2
        candidate = prediction['state'][:, chosen]
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
        pixels = valid.nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(1024, len(pixels)),
            device='cuda').long()]
        chart = torch.stack((pixels.remainder(side),
            pixels.div(side, rounding_mode='floor')), -1).float() / side
        item.update({'inputs': image[0].cpu(), 'offsets': sample['offsets'][0].cpu(),
            'weights': sample['weights'][0].cpu(), 'truth_state': sample['state'][0].cpu(),
            'truth_reflection': sample['reflection'][0].cpu(), 'chart': chart.cpu(),
            'branch_ids': ids.cpu().tolist(), 'reflection': flags[0].cpu(),
            'initial_state': candidate[0].cpu(),
            'prior_score': prior[0, ids].cpu().tolist(),
            'native_score': torch.cat(native_score).cpu().tolist(),
            'native_state': torch.cat(native_state).cpu()})
        panel.append(item)
        if (index + 1) % 16 == 0:
            print(json.dumps({'event': 'panel', 'attempts': index + 1,
                'nonblank': sum('inputs' in row for row in panel)}), flush=True)

out.mkdir(parents=True, exist_ok=False)
config = {'scope': 'fresh TRAIN-only blind-16, not animal-held-out DEV or final test',
    'seed': seed, 'attempts': attempts, 'nonblank': sum('inputs' in row for row in panel),
    'train_bases': list(range(60)), 'geometry': '166 80%, 151 20%, v4 artifact draw; no blank retry',
    'steps': steps, 'branch_policy': 'frozen 121 blind16 top8 base, top6 anchor, two normal-diversity',
    'score_policy': 'prior, native121 best original/corrected action, anatomy0, anatomy1, matched atlas-intensity-null0/1; no truth used for ranking',
    'source_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
    'training_completion_sha256': sha(train / 'completed.json'),
    'training_config_sha256': sha(train / 'config.json'),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_path),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoints.items()},
    'synthetic_provenance': context['provenance'], 'calibrated': False,
    'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

matcher = AnatomyCorrespondenceFeedback(side=bank_side, radius=3).cuda().eval().requires_grad_(False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step, path in checkpoints.items():
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step and json.loads(json.dumps(saved['config'])) == training_config
        assert saved['calibrated'] is False
        assert all(torch.equal(saved['model'][key], value) for key, value in parent_state.items())
        prefix = 'anatomy_correspondence_168.'
        matcher.load_state_dict({key[len(prefix):]: value for key, value
            in saved['model'].items() if key.startswith(prefix)}, strict=True)
        del saved
        for item in panel:
            record = {key: item[key] for key in ('index', 'base_index', 'draw_seed',
                'geometry', 'provenance', 'valid_pixels')}
            record['step'] = step
            if 'inputs' not in item:
                record['scored'] = False
                rows.append(record)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                continue
            image = item['inputs'][None].cuda()
            source_desc = matcher.encode_source(image)
            states = item['initial_state'].cuda()
            reflection = item['reflection'].cuda()
            offsets = item['offsets'][None].cuda()
            weights = item['weights'][None].cuda()
            scores = {name: [] for name in ('anatomy0', 'anatomy1', 'null0', 'null1')}
            corrected = []
            for start in range(0, 16, 4):
                state = states[start:start + 4]
                flags = reflection[start:start + 4]
                intensity, support = render_bank(state, flags, offsets, weights)
                first = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
                    matcher.encode_atlas(intensity), support, depths)
                null_first = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
                    matcher.encode_atlas(torch.zeros_like(intensity)), support, depths)
                fitted = compose_full_frame_state(state, first['update'])
                fitted_intensity, fitted_support = render_bank(fitted, flags, offsets, weights)
                second = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
                    matcher.encode_atlas(fitted_intensity), fitted_support, depths)
                null_second = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
                    matcher.encode_atlas(torch.zeros_like(fitted_intensity)), fitted_support, depths)
                effort = .04 * (first['update'] / matcher.update_limits).square().mean(-1)
                scores['anatomy0'].append(first['score'])
                scores['anatomy1'].append(second['score'] - effort)
                scores['null0'].append(null_first['score'])
                scores['null1'].append(null_second['score'] - effort)
                corrected.append(fitted)
            scores = {name: torch.cat(values).cpu().tolist() for name, values in scores.items()}
            corrected = torch.cat(corrected)
            # Truth is read only after every candidate identity and score is fixed.
            chart = item['chart'].cuda()
            truth = rigid_points_090(item['truth_state'][None].cuda(),
                item['truth_reflection'][None].cuda(), chart)[0]
            target = truth[None]
            initial_mm = ((rigid_points_090(states, reflection, chart) - target
                ).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
            anatomy_mm = ((rigid_points_090(corrected, reflection, chart) - target
                ).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
            native_mm = ((rigid_points_090(item['native_state'].cuda(), reflection, chart)
                - target).norm(dim=-1).mean(-1) / 1000).cpu().tolist()
            all_scores = {'prior': item['prior_score'], 'native121': item['native_score'],
                **scores}
            selected = {name: int(np.argmax(value)) for name, value in all_scores.items()}
            normal = np.abs(item['provenance']['physical_unit_normal_ap_dv_ml'])
            angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
            support_fraction = item['valid_pixels'] / (side * side)
            record.update({'scored': True,
                'animal_id': item['provenance']['base_lineage']['animal_id'],
                'specimen_id': item['provenance']['base_lineage']['specimen_id'],
                'experiment_id': item['provenance']['base_lineage']['experiment_id'],
                'physical_section_id': item['provenance']['physical_section_id'],
                'appearance_mode': item['provenance']['mode'],
                'nearest_cardinal_angle_deg': angle,
                'angle_bin': '0-15' if angle < 15 else '15-30' if angle < 30
                    else '30-45' if angle < 45 else '45-55',
                'support_fraction': support_fraction,
                'support_bin': '0-.10' if support_fraction < .10 else '.10-.25'
                    if support_fraction < .25 else '.25-.50' if support_fraction < .50
                    else '.50-1',
                'branch_ids': item['branch_ids'], 'scores': all_scores,
                'chosen_slot': selected, 'initial_rigid_mm': initial_mm,
                'anatomy_fitted_rigid_mm': anatomy_mm,
                'native_fitted_rigid_mm': native_mm})
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'event': 'checkpoint_scored', 'step': step,
            'nonblank': sum(row['scored'] for row in rows if row['step'] == step)}), flush=True)

summary = {}
for step in steps:
    group = [row for row in rows if row['step'] == step and row['scored']]
    summary[str(step)] = {'all': readout(group),
        'angle': {key: readout([row for row in group if row['angle_bin'] == key])
            for key in ('0-15', '15-30', '30-45', '45-55')},
        'support': {key: readout([row for row in group if row['support_bin'] == key])
            for key in ('0-.10', '.10-.25', '.25-.50', '.50-1')},
        'appearance': {key: readout([row for row in group if row['appearance_mode'] == key])
            for key in ('raw', 'exact_black', 'imperfect_brush')}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'attempts': attempts,
    'nonblank': config['nonblank'], 'rows': len(rows),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'all': {step: result['all']
    for step, result in summary.items()}}, allow_nan=False), flush=True)
