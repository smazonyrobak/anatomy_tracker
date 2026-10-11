"""Frozen TRAIN-only local atlas-fit direction on acquired fluorescence."""

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_from_components,
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_joint_uncertainty import full_frame_residual
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fit_score(image, atlas_images, mask):
    source = F.avg_pool2d(image, 3, 1, 1)[0, 0]
    target = F.avg_pool2d(atlas_images, 3, 1, 1)[:, 0]
    area = mask.sum()
    a = (source * mask).sum() / area
    b = (target * mask).sum((-2, -1)) / area
    u = (source - a) * mask
    v = (target - b[:, None, None]) * mask
    correlation = (u[None] * v).sum((-2, -1)) / (
        (u.square().sum() * v.square().sum((-2, -1))).sqrt().clamp_min(1e-8))
    return (correlation.square() + 1e-4).sqrt()


def permute_intensity(image, pixels, shifted):
    original = image.flatten(1)
    permuted = original.clone()
    permuted[:, pixels] = original[:, shifted]
    return permuted.reshape_as(image)


split_path = Path(__file__).with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
train_run = root / 'runs/lupori_fluorescence_pose_188_all_train'
checkpoint = train_run / 'joint_step_01200.pt'
protocol = Path(__file__).resolve().parents[1] / 'docs/publication/LUPORI_ACQUIRED_ATLAS_FIT_DIRECTION_189_PROTOCOL_20261011.md'
out = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'
assert Path(__file__).drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert archive.stat().st_size == split['archive_bytes']
completed = json.loads((train_run / 'completed.json').read_text())
assert sha(checkpoint) == completed['checkpoint_sha256']['1200']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 1200 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

selected = []
with ZipFile(archive) as source_zip:
    for animal in split['train']:
        info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        rows = [row for row in visualign['slices']
                if Path(row['filename']).name.removesuffix('-thumb.png') in valid]
        rows.sort(key=lambda row: hashlib.sha256((animal + '/' +
            Path(row['filename']).name.removesuffix('-thumb.png')).encode()).hexdigest())
        selected += [(animal, row) for row in rows[:16]]
assert len(selected) == 64

side, grid_side = 256, 48
axis = (torch.arange(grid_side, device='cuda') + .5) / grid_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')


def render(states, reflections):
    plane = rigid_points_090(states, reflections, chart).reshape(-1, grid_side, grid_side, 3)
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    world = plane[:, None] + offsets[None, :, None, None, None] * normal[:, None, None, None, :]
    sampled = render_finite_thickness_coordinate_grid(atlas, world, (0., 0., 0.),
        (25., 25., 25.), weights.expand(len(states), -1))
    support = sampled[:, 1:2].clamp(0, 1)
    return sampled[:, :1] / support.clamp_min(1e-4), support


config = {'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'split_sha256': sha(split_path), 'protocol_sha256': sha(protocol),
    'source_sha256': sha(__file__), 'checkpoint_sha256': sha(checkpoint),
    'train_completion_sha256': sha(train_run / 'completed.json'),
    'animals': split['train'], 'sections_per_animal': 16,
    'section_selection': 'smallest SHA256(animal_id/section_id) among valid',
    'grid_side': grid_side, 'top_proposals': 16,
    'atlas_offsets_um': offsets.tolist(), 'atlas_weights': weights.tolist(),
    'mask': 'initial proposal support eroded 7x7, never source tissue mask',
    'truth_role': 'retrospective diagnostic direction/error only',
    'calibrated': False, 'public_benchmark_used': False,
    'heldout_development_calibration_final_read': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))

cases = []
with ZipFile(archive) as source_zip, (out / 'cases.jsonl').open('w') as stream:
    for animal, row in selected:
        section_id = Path(row['filename']).name.removesuffix('-thumb.png')
        image_bytes = source_zip.read(f'{animal}/{row["filename"]}')
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        image = torch.from_numpy(gray[None, None]).cuda()
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[:, :1] = image
        source_ouv = torch.tensor(row['anchoring'], device='cuda', dtype=torch.float32).reshape(3, 3)
        physical_ouv = torch.stack((quicknii_to_allen_points(source_ouv[0]),
            quicknii_to_allen_vectors(source_ouv[1]),
            quicknii_to_allen_vectors(source_ouv[2]))) * 25
        truth = physical_ouv[0] + five[:, :1] * physical_ouv[1] + five[:, 1:] * physical_ouv[2]
        with torch.no_grad():
            prediction = model.predict(inputs)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            ids = prior.topk(16).indices
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            physical = (rigid_points_090(states, flags, five) - truth[None]
                ).norm(dim=-1).mean(-1) / 1000
            near = ((physical >= .5) & (physical <= 3.)).nonzero()[:, 0]
            near_slot = int(near[physical[near].argmin()]) if len(near) else None
        case = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'section_id': section_id, 'source_thumbnail': f'{animal}/{row["filename"]}',
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'candidate_ids': ids.tolist(), 'prior_log_mass': prior[ids].tolist(),
            'candidate_five_point_mm': physical.tolist(), 'top16_near_le_1p5':
            bool((physical <= 1.5).any()), 'branches': {}}
        for name, slot in (('prior_top', 0), ('near_diagnostic', near_slot)):
            if slot is None or not (.5 <= float(physical[slot]) <= 3.):
                case['branches'][name] = {'eligible': False, 'reason': 'no_captured_candidate'}
                continue
            state, flag = states[slot].detach(), flags[slot]
            target_ouv = physical_ouv.clone()
            if bool(flag):
                target_ouv[0] += (side - 1) / side * target_ouv[1]
                target_ouv[1] *= -1
            target_state = full_frame_state_from_components(*physical_ouv_to_frame(target_ouv))
            residual = full_frame_residual(state[None], target_state[None])[0].detach()
            with torch.no_grad():
                trial_states = compose_full_frame_state(state.expand(3, -1),
                    torch.tensor([.2, 0., -.2], device='cuda')[:, None] * residual)
                rendered, support = render(trial_states, flag.expand(3))
                common = (support[1, 0] > .5).float()
                common = (F.avg_pool2d(common[None, None], 7, 1, 3)[0, 0] > .999).float()
                sites = int(common.sum())
                if sites < 100:
                    case['branches'][name] = {'eligible': False,
                        'reason': 'insufficient_atlas_support', 'shared_sites': sites}
                    continue
                pixels = common.flatten().nonzero()[:, 0]
                shifted = pixels.roll(max(1, sites // 3))
                full = fit_score(F.interpolate(image, (grid_side, grid_side), mode='area'),
                    rendered, common)
                null = fit_score(F.interpolate(image, (grid_side, grid_side), mode='area'),
                    permute_intensity(rendered, pixels, shifted), common)
                moved = rigid_points_090(trial_states, flag.expand(3), five)
                errors = (moved - truth[None]).norm(dim=-1).mean(-1) / 1000
                retention = [float(((support[j, 0] > .5).float() * common).sum()
                    / common.sum()) for j in range(3)]
            direction = torch.zeros((), device='cuda', requires_grad=True)
            current_state = compose_full_frame_state(state[None], (direction * residual)[None])
            current, _ = render(current_state, flag[None])
            down = F.interpolate(image, (grid_side, grid_side), mode='area')
            score = fit_score(down, current, common)[0]
            null_score = fit_score(down, permute_intensity(current, pixels, shifted), common)[0]
            full_gradient = torch.autograd.grad(score, direction, retain_graph=True)[0]
            null_gradient = torch.autograd.grad(null_score, direction)[0]
            case['branches'][name] = {'eligible': True, 'slot': slot,
                'branch_id': int(ids[slot]), 'shared_sites': sites,
                'physical_mm_toward_center_away': errors.tolist(),
                'physical_move_monotonic': bool(errors[0] < errors[1] < errors[2]),
                'support_retention_toward_center_away': retention,
                'fit_score_toward_center_away': full.tolist(),
                'null_score_toward_center_away': null.tolist(),
                'fit_direction_positive': bool(full[0] > full[2]),
                'null_direction_positive': bool(null[0] > null[2]),
                'fit_derivative_positive': bool(full_gradient > 0),
                'null_derivative_positive': bool(null_gradient > 0),
                'fit_derivative': float(full_gradient),
                'null_derivative': float(null_gradient)}
        cases.append(case)
        stream.write(json.dumps(case, allow_nan=False) + '\n')
        if len(cases) % 16 == 0:
            stream.flush()
            print(json.dumps({'event': 'animal_milestone', 'sections': len(cases),
                'animal_id': animal}), flush=True)

summary = {'sections': len(cases), 'top16_near_le_1p5': sum(
    case['top16_near_le_1p5'] for case in cases), 'groups': {}}
for name in ('prior_top', 'near_diagnostic'):
    for animal in ('all', *split['train']):
        group = [case['branches'][name] for case in cases if animal == 'all'
            or case['animal_id'] == animal]
        eligible = [branch for branch in group if branch['eligible']]
        summary['groups'][f'{name}/{animal}'] = {'eligible': len(eligible),
            'fit_forward': sum(branch['fit_direction_positive'] for branch in eligible),
            'null_forward': sum(branch['null_direction_positive'] for branch in eligible),
            'fit_gradient_forward': sum(branch['fit_derivative_positive'] for branch in eligible),
            'monotonic_physical': sum(branch['physical_move_monotonic'] for branch in eligible),
            'minimum_support_retention': min((min(branch[
                'support_retention_toward_center_away']) for branch in eligible), default=None)}
captured = summary['groups']['near_diagnostic/all']
animal_wins = sum(summary['groups'][f'near_diagnostic/{animal}']['fit_forward'] >
    .5 * summary['groups'][f'near_diagnostic/{animal}']['eligible'] and
    summary['groups'][f'near_diagnostic/{animal}']['eligible'] > 0
    for animal in split['train'])
summary['predeclared_gate'] = {'passed': bool(captured['eligible'] >= 32 and
    captured['fit_forward'] / max(1, captured['eligible']) >= .65 and
    (captured['fit_forward'] - captured['null_forward']) /
        max(1, captured['eligible']) >= .10 and
    captured['fit_gradient_forward'] / max(1, captured['eligible']) >= .60 and
    animal_wins >= 3), 'animals_above_half': animal_wins}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'source_sha256': sha(__file__),
    'checkpoint_sha256': sha(checkpoint), 'sections': len(cases),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'cases.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}, allow_nan=False), flush=True)
