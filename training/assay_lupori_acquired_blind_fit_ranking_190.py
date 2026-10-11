"""TRAIN-only rigid atlas-image ranking of blind fluorescence pose proposals."""

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
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
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(image, target, mask):
    source = F.avg_pool2d(image, 3, 1, 1)[0, 0]
    target = F.avg_pool2d(target, 3, 1, 1)[:, 0]
    count = mask.sum()
    u = (source - (source * mask).sum() / count) * mask
    v = (target - (target * mask).sum((-2, -1))[:, None, None] / count) * mask
    correlation = (u[None] * v).sum((-2, -1)) / (
        (u.square().sum() * v.square().sum((-2, -1))).sqrt().clamp_min(1e-8))
    return (correlation.square() + 1e-4).sqrt()


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/LUPORI_ACQUIRED_BLIND_FIT_RANKING_190_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
checkpoint = root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt'
previous = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'
out = root / 'runs/lupori_acquired_blind_fit_ranking_190_train_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert sha(checkpoint) == json.loads((root /
    'runs/lupori_fluorescence_pose_188_all_train/completed.json').read_text())[
    'checkpoint_sha256']['1200']
previous_done = json.loads((previous / 'completed.json').read_text())
assert sha(previous / 'cases.jsonl') == previous_done['output_sha256']['cases.jsonl']
selected = [json.loads(line) for line in (previous / 'cases.jsonl').read_text().splitlines()]
assert len(selected) == 64 and sorted({row['animal_id'] for row in selected}) == sorted(split['train'])
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
    'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'checkpoint_sha256': sha(checkpoint), 'previous_189_completion_sha256':
    sha(previous / 'completed.json'), 'previous_189_cases_sha256': sha(previous / 'cases.jsonl'),
    'split_sha256': sha(split_path), 'animals': split['train'],
    'input': 'whole RGB 256, max(red,green)/255, image-wide p99, four channels zero',
    'grid_side': grid_side, 'top_proposals': 16,
    'atlas_offsets_um': offsets.tolist(), 'atlas_weights': weights.tolist(),
    'trained': False, 'calibrated': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    for previous_row in selected:
        animal = previous_row['animal_id']
        image_bytes = source_zip.read(previous_row['source_thumbnail'])
        assert hashlib.sha256(image_bytes).hexdigest() == previous_row['source_thumbnail_sha256']
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        image = torch.from_numpy(gray[None, None]).cuda()
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[:, :1] = image
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        metadata = next(row for row in visualign['slices'] if
            f'{animal}/{row["filename"]}' == previous_row['source_thumbnail'])
        quicknii = torch.tensor(metadata['anchoring'], device='cuda', dtype=torch.float32).reshape(3, 3)
        o = quicknii_to_allen_points(quicknii[0]) * 25
        u = quicknii_to_allen_vectors(quicknii[1]) * 25
        v = quicknii_to_allen_vectors(quicknii[2]) * 25
        truth = o + five[:, :1] * u + five[:, 1:] * v
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
            assert ids.tolist() == previous_row['candidate_ids']
            assert max(abs(a - b) for a, b in zip(physical.tolist(),
                previous_row['candidate_five_point_mm'])) < 1e-4
            rendered, support = [], []
            for start in range(0, 16, 4):
                chunk_image, chunk_support = render(states[start:start + 4], flags[start:start + 4])
                rendered.append(chunk_image)
                support.append(chunk_support)
            rendered, support = torch.cat(rendered), torch.cat(support)
            down = F.interpolate(image, (grid_side, grid_side), mode='area')
            individual = []
            individual_sites = []
            for branch in range(16):
                mask = (support[branch, 0] > .5).float()
                mask = (F.avg_pool2d(mask[None, None], 7, 1, 3)[0, 0] > .999).float()
                sites = int(mask.sum())
                individual_sites.append(sites)
                individual.append(float(score(down, rendered[branch:branch + 1], mask)[0])
                                  if sites >= 100 else None)
            valid_scores = [(index, value) for index, value in enumerate(individual)
                            if value is not None]
            blind_index = max(valid_scores, key=lambda item: item[1])[0] if valid_scores else None
            near_choices = (physical <= 1.5).nonzero()[:, 0]
            far_choices = (physical >= 3.).nonzero()[:, 0]
            near = int(near_choices[physical[near_choices].argmin()]) if len(near_choices) else None
            far = int(far_choices[0]) if len(far_choices) else None
            pair = {'eligible': False, 'reason': 'missing_near_or_far'}
            if near is not None and far is not None:
                mask = ((support[near, 0] > .5) & (support[far, 0] > .5)).float()
                mask = (F.avg_pool2d(mask[None, None], 7, 1, 3)[0, 0] > .999).float()
                sites = int(mask.sum())
                pair = {'eligible': False, 'reason': 'insufficient_common_support',
                        'near_slot': near, 'far_slot': far, 'common_sites': sites}
                if sites >= 100:
                    match = score(down, rendered[[near, far]], mask)
                    pixels = mask.flatten().nonzero()[:, 0]
                    shifted = pixels.roll(max(1, sites // 3))
                    base = rendered[[near, far]].flatten(1)
                    permuted = base.clone()
                    permuted[:, pixels] = base[:, shifted]
                    null = score(down, permuted.reshape(2, 1, grid_side, grid_side), mask)
                    pair = {'eligible': True, 'near_slot': near, 'far_slot': far,
                        'common_sites': sites, 'real_near_far': match.tolist(),
                        'null_near_far': null.tolist(),
                        'real_near_wins': bool(match[0] > match[1]),
                        'null_near_wins': bool(null[0] > null[1]),
                        'prior_near_wins': bool(near < far),
                        'common_support_fraction': float(mask.mean())}
        record = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'section_id': previous_row['section_id'],
            'source_thumbnail': previous_row['source_thumbnail'],
            'source_thumbnail_sha256': previous_row['source_thumbnail_sha256'],
            'candidate_ids': ids.tolist(), 'prior_log_mass': prior[ids].tolist(),
            'candidate_five_point_mm': physical.tolist(), 'individual_fit_score': individual,
            'individual_supported_sites': individual_sites, 'blind_fit_slot': blind_index,
            'prior_selected_mm': float(physical[0]),
            'blind_fit_selected_mm': float(physical[blind_index]) if blind_index is not None else None,
            'pair': pair}
        rows.append(record)
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        if len(rows) % 16 == 0:
            stream.flush()
            print(json.dumps({'event': 'animal_milestone', 'sections': len(rows),
                'animal_id': animal}), flush=True)

group = [row for row in rows if row['pair']['eligible']]
pair = {'eligible': len(group),
    'real_near_wins': sum(row['pair']['real_near_wins'] for row in group),
    'null_near_wins': sum(row['pair']['null_near_wins'] for row in group),
    'prior_near_wins': sum(row['pair']['prior_near_wins'] for row in group)}
by_animal = {}
for animal in split['train']:
    animal_rows = [row for row in rows if row['animal_id'] == animal]
    paired = [row for row in animal_rows if row['pair']['eligible']]
    by_animal[animal] = {'pair_eligible': len(paired),
        'real_near_wins': sum(row['pair']['real_near_wins'] for row in paired),
        'null_near_wins': sum(row['pair']['null_near_wins'] for row in paired),
        'prior_selected_mean_mm': float(np.mean([row['prior_selected_mm'] for row in animal_rows])),
        'blind_fit_selected_mean_mm': float(np.mean([row['blind_fit_selected_mm']
            if row['blind_fit_selected_mm'] is not None else row['prior_selected_mm']
            for row in animal_rows]))}
missing = sum(row['blind_fit_slot'] is None for row in rows)
pair_gate = bool(pair['eligible'] >= 32 and
    pair['real_near_wins'] / max(1, pair['eligible']) >= .70 and
    (pair['real_near_wins'] - pair['null_near_wins']) /
        max(1, pair['eligible']) >= .10 and
    sum(by_animal[animal]['pair_eligible'] > 0 and
        by_animal[animal]['real_near_wins'] > .5 * by_animal[animal]['pair_eligible']
        for animal in split['train']) >= 3)
prior_mean = float(np.mean([by_animal[animal]['prior_selected_mean_mm']
    for animal in split['train']]))
blind_mean = float(np.mean([by_animal[animal]['blind_fit_selected_mean_mm']
    for animal in split['train']]))
blind_gate = bool(prior_mean - blind_mean >= .15 and missing <= 4 and
    all(by_animal[animal]['blind_fit_selected_mean_mm'] -
        by_animal[animal]['prior_selected_mean_mm'] <= .2 for animal in split['train']))
summary = {'sections': len(rows), 'pair': pair, 'animal': by_animal,
    'individual_score_missing': missing, 'prior_equal_animal_mean_mm': prior_mean,
    'blind_fit_equal_animal_mean_mm': blind_mean,
    'predeclared_pair_gate': pair_gate, 'predeclared_blind_gate': blind_gate}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': sha(source), 'checkpoint_sha256': sha(checkpoint),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'trained': False, 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}, allow_nan=False), flush=True)
