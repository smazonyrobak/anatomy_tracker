"""No-update AL1A check of atlas-fit ranking before and after 193 map adaptation."""

import copy
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
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(source, target, mask):
    source = F.avg_pool2d(source, 3, 1, 1)[0, 0]
    target = F.avg_pool2d(target, 3, 1, 1)[:, 0]
    count = mask.sum()
    u = (source - (source * mask).sum() / count) * mask
    v = (target - (target * mask).sum((-2, -1))[:, None, None] / count) * mask
    correlation = (u[None] * v).sum((-2, -1)) / (
        (u.square().sum() * v.square().sum((-2, -1))).sqrt().clamp_min(1e-8))
    return (correlation.square() + 1e-4).sqrt()


source = Path(__file__).resolve()
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent_path = parent_run / 'joint_step_00800.pt'
map_run = root / 'runs/lupori_local_map_193_inner_AL1A'
map_path = map_run / 'map_step_00600.pt'
out = root / 'runs/lupori_fit_rank_202_train_001'
assert not out.exists() and split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert sha(parent_path) == json.loads((parent_run / 'completed.json').read_text())['checkpoint_sha256']['800']
map_done = json.loads((map_run / 'completed.json').read_text())
assert not map_done['guard_animal_used_in_training']
assert sha(map_path) == map_done['checkpoint_sha256']['600']
assert json.loads((map_run / 'config.json').read_text())['parent_checkpoint_sha256'] == sha(parent_path)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 800
model.load_state_dict(saved['model'], strict=True)
adapted = copy.deepcopy(model).eval().requires_grad_(False)
saved = torch.load(map_path, map_location='cpu', weights_only=True)
assert saved['step'] == 600
for name, weights in saved['modules'].items():
    getattr(adapted, name).load_state_dict(weights, strict=True)

side, grid = 256, 96
axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
config = {'animal': 'AL1A', 'scope': 'TRAIN inner-animal guard; no updates',
    'parent_checkpoint_sha256': sha(parent_path), 'adapted_map_sha256': sha(map_path),
    'source_sha256': sha(source), 'split_sha256': sha(split_path),
    'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'input': 'whole RGB 256, max(red,green)/255, image-wide p99, four channels zero',
    'beam': '187 direct prior top 16 of 160 reflection branches',
    'pair': 'lowest five-point-error branch <=1.5 mm versus highest-prior branch >=3 mm',
    'mask': 'detached common rigid atlas support >0.5, eroded 7x7; >=400 pixels',
    'score': 'smooth absolute Pearson after 3x3 smoothing on identical fixed mask; higher is better',
    'null': 'cyclic one-third within-mask permutation of each rendered atlas intensity at score stage',
    'atlas_offsets_um': offsets.tolist(), 'atlas_weights': weights.tolist(),
    'map_side': grid, 'trained': False, 'calibrated': False,
    'dev_cal_final_read': False, 'public_benchmark_used': False}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    animal = 'AL1A'
    info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
    valid = {row.findtext('name') for row in info.findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
    for metadata in visualign['slices']:
        section = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        image_bytes = source_zip.read(f'{animal}/{metadata["filename"]}')
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = torch.from_numpy(gray).cuda()
        q = torch.tensor(metadata['anchoring'], device='cuda', dtype=torch.float32).reshape(3, 3)
        o = quicknii_to_allen_points(q[0]) * 25
        u = quicknii_to_allen_vectors(q[1]) * 25
        v = quicknii_to_allen_vectors(q[2]) * 25
        truth = o + five[:, :1] * u + five[:, 1:] * v
        with torch.inference_mode():
            prediction = model.predict(inputs)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            ids = prior.topk(16).indices
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            error = (rigid_points_090(states, flags, five) - truth[None]
                ).norm(dim=-1).mean(-1) / 1000
            near_choices = (error <= 1.5).nonzero()[:, 0]
            far_choices = (error >= 3.).nonzero()[:, 0]
            record = {'animal_id': animal, 'section_id': section,
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'candidate_ids': ids.tolist(), 'candidate_five_point_mm': error.tolist(),
                'prior_selected_mm': float(error[0]), 'eligible': False}
            if len(near_choices) and len(far_choices):
                near = int(near_choices[error[near_choices].argmin()])
                far = int(far_choices[0])
                pair_ids = ids[[near, far]]
                pair_state = states[[near, far]]
                pair_flags = flags[[near, far]]
                plane = rigid_points_090(pair_state, pair_flags, chart).reshape(2, grid, grid, 3)
                normal = full_frame_state_to_components(pair_state)[1][..., :, 2]
                world = plane[:, None] + offsets[None, :, None, None, None] * normal[:, None, None, None, :]
                rigid_render = render_finite_thickness_coordinate_grid(atlas, world,
                    (0., 0., 0.), (25., 25., 25.), weights.expand(2, -1))
                rigid_support = rigid_render[:, 1:2].clamp(0, 1)
                mask = (rigid_support[:, 0] > .5).all(0).float()
                mask = (F.avg_pool2d(mask[None, None], 7, 1, 3)[0, 0] > .999).float()
                sites = int(mask.sum())
                record.update({'near_slot': near, 'far_slot': far, 'common_sites': sites})
                if sites >= 400:
                    observed = F.interpolate(inputs[:, :1], (grid, grid), mode='area')
                    renderings = {'zero': rigid_render}
                    for name, mapper in (('parent', model), ('adapted', adapted)):
                        mapped = mapper.map(prediction, offsets, (pair_ids // 2)[None],
                            pair_flags[None], (grid, grid), atlas, weights,
                            feature_side=grid, source_shape=(side, side))
                        renderings[name] = render_finite_thickness_coordinate_grid(
                            atlas, mapped['coordinates'][0], (0., 0., 0.),
                            (25., 25., 25.), weights.expand(2, -1))
                        record[name + '_warp_rms_mm'] = (mapped['local_displacement_um'][0]
                            .square().sum(1).mean((1, 2)).sqrt() / 1000).tolist()
                    for name, rendered in renderings.items():
                        support = rendered[:, 1:2].clamp(0, 1)
                        atlas_image = rendered[:, :1] / support.clamp_min(1e-4)
                        pixels = mask.flatten().nonzero()[:, 0]
                        shifted = pixels.roll(max(1, sites // 3))
                        permuted = atlas_image.flatten(1).clone()
                        permuted[:, pixels] = permuted[:, shifted]
                        record[name + '_score'] = score(observed, atlas_image, mask).tolist()
                        record[name + '_null_score'] = score(observed,
                            permuted.reshape_as(atlas_image), mask).tolist()
                        record[name + '_support_retention'] = (
                            (support[:, 0] * mask).sum((1, 2)) / mask.sum()).tolist()
                    record['eligible'] = True
                    record['prior_near_wins'] = near < far
        rows.append(record)
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        if len(rows) % 13 == 0:
            stream.flush()
            print(json.dumps({'sections': len(rows)}), flush=True)

paired = [row for row in rows if row['eligible']]
summary = {'sections': len(rows), 'eligible_pairs': len(paired),
    'near_available': sum(min(row['candidate_five_point_mm']) <= 1.5 for row in rows),
    'prior_near_wins': sum(row['prior_near_wins'] for row in paired)}
for name in ('zero', 'parent', 'adapted'):
    summary[name] = {'near_wins': sum(row[name + '_score'][0] > row[name + '_score'][1]
        for row in paired), 'null_near_wins': sum(row[name + '_null_score'][0] >
        row[name + '_null_score'][1] for row in paired),
        'median_near_minus_far': float(np.median([row[name + '_score'][0] -
            row[name + '_score'][1] for row in paired])) if paired else None,
        'minimum_support_retention': min(min(row[name + '_support_retention'])
            for row in paired) if paired else None}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': sha(source), 'parent_checkpoint_sha256': sha(parent_path),
    'adapted_map_sha256': sha(map_path), 'output_sha256': {name: sha(out / name)
        for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'trained': False, 'dev_cal_final_read': False}, indent=2))
print(json.dumps({'completed': summary}, allow_nan=False), flush=True)
