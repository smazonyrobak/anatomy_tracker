"""Frozen-pose, no-update structural fit on one TRAIN guard and one DEV animal."""

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


def mind(image):
    shifted = F.pad(image, (3, 3, 3, 3), mode='replicate')
    distance = torch.cat([F.avg_pool2d((image - shifted[:, :, 3 + dy:99 + dy,
        3 + dx:99 + dx]).square(), 3, 1, 1) for dy, dx in
        ((3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (3, -3), (-3, 3), (-3, -3))], 1)
    return torch.exp(-distance / distance.mean(1, keepdim=True).clamp_min(1e-4))


def scores(observed, atlas_image, mask):
    source = F.avg_pool2d(observed, 3, 1, 1)[0, 0]
    target = F.avg_pool2d(atlas_image, 3, 1, 1)[:, 0]
    count = mask.sum()
    u = (source - (source * mask).sum() / count) * mask
    v = (target - (target * mask).sum((-2, -1))[:, None, None] / count) * mask
    pearson = ((u[None] * v).sum((-2, -1)) / (u.square().sum() *
        v.square().sum((-2, -1))).sqrt().clamp_min(1e-8)).abs()
    structure = -((mind(observed) - mind(atlas_image)).square().mean(1) *
        mask).sum((-2, -1)) / count
    return pearson.tolist(), structure.tolist()


source = Path(__file__).resolve()
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
out = root / 'runs/lupori_structural_fit_204_guard_dev_001_retry1'
parents = {'AL1A': root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt',
           'CC1B': root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt'}
assert not out.exists() and split['development'] == ['CC1B'] and 'AL1A' in split['train']
for animal, path in parents.items():
    step = '800' if animal == 'AL1A' else '1200'
    assert sha(path) == json.loads((path.parent / 'completed.json').read_text())['checkpoint_sha256'][step]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
side, grid = 256, 96
axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
config = {'animals': ['AL1A', 'CC1B'], 'roles': ['TRAIN guard', 'DEV'],
    'pose_checkpoint_sha256': {a: sha(p) for a, p in parents.items()},
    'source_sha256': sha(source), 'split_sha256': sha(split_path),
    'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'near_far': 'blind top16, <=1.5 mm versus >=3 mm approximate human plane',
    'local': 'same captured near pose shifted ±250 um in AP; physically nearer wins',
    'mask': 'detached common rigid atlas support >0.5 eroded 7x7, >=400 pixels',
    'scores': ['absolute Pearson after 3x3 smoothing',
               'negative 8-neighbor 3-pixel MIND squared distance'],
    'null': 'cyclic one-third within-mask intensity permutation',
    'map_used': False, 'trained': False, 'calibrated': False,
    'calibration_final_public_read': False}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    for animal in ('AL1A', 'CC1B'):
        saved = torch.load(parents[animal], map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        del saved
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
            record = {'animal_id': animal, 'section_id': section,
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest()}
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
                record.update({'prior_selected_mm': float(error[0]),
                    'best16_mm': float(error.min()), 'eligible_pair': False})
                if len(near_choices) and len(far_choices):
                    near = int(near_choices[error[near_choices].argmin()])
                    far = int(far_choices[0])
                    pair = states[[near, far]]
                    pair_flags = flags[[near, far]]
                    plane = rigid_points_090(pair, pair_flags, chart).reshape(2, grid, grid, 3)
                    normal = full_frame_state_to_components(pair)[1][..., :, 2]
                    world = plane[:, None] + offsets[None, :, None, None, None] * normal[:, None, None, None, :]
                    rigid = render_finite_thickness_coordinate_grid(atlas, world,
                        (0., 0., 0.), (25., 25., 25.), weights.expand(2, -1))
                    support = rigid[:, 1:2].clamp(0, 1)
                    mask = (support[:, 0] > .5).all(0).float()
                    mask = (F.avg_pool2d(mask[None, None], 7, 1, 3)[0, 0] > .999).float()
                    if int(mask.sum()) >= 400:
                        observed = F.interpolate(inputs[:, :1], (grid, grid), mode='area')
                        rendered = rigid[:, :1] / support.clamp_min(1e-4)
                        pixels = mask.flatten().nonzero()[:, 0]
                        permuted = rendered.flatten(1).clone()
                        permuted[:, pixels] = permuted[:, pixels.roll(max(1, len(pixels) // 3))]
                        pearson, structural = scores(observed, rendered, mask)
                        null_pearson, null_structural = scores(observed,
                            permuted.reshape_as(rendered), mask)
                        record.update({'eligible_pair': True, 'near_slot': near, 'far_slot': far,
                            'near_mm': float(error[near]), 'far_mm': float(error[far]),
                            'common_sites': int(mask.sum()), 'pearson': pearson,
                            'mind': structural, 'null_pearson': null_pearson,
                            'null_mind': null_structural})
                        local = pair[:1].expand(2, -1).clone()
                        local[:, 0] += torch.tensor([-250., 250.], device='cuda')
                        local_flags = pair_flags[:1].expand(2)
                        local_error = (rigid_points_090(local, local_flags, five) - truth[None]
                            ).norm(dim=-1).mean(-1) / 1000
                        local_plane = rigid_points_090(local, local_flags, chart).reshape(2, grid, grid, 3)
                        local_normal = full_frame_state_to_components(local)[1][..., :, 2]
                        local_world = local_plane[:, None] + offsets[None, :, None, None, None] * local_normal[:, None, None, None, :]
                        local_render = render_finite_thickness_coordinate_grid(atlas,
                            local_world, (0., 0., 0.), (25., 25., 25.), weights.expand(2, -1))
                        local_support = local_render[:, 1:2].clamp(0, 1)
                        local_mask = (local_support[:, 0] > .5).all(0).float()
                        local_mask = (F.avg_pool2d(local_mask[None, None], 7, 1, 3)[0, 0] > .999).float()
                        if int(local_mask.sum()) >= 400:
                            local_image = local_render[:, :1] / local_support.clamp_min(1e-4)
                            local_p, local_m = scores(observed, local_image, local_mask)
                            record.update({'local_error_mm': local_error.tolist(),
                                'local_pearson': local_p, 'local_mind': local_m,
                                'local_sites': int(local_mask.sum())})
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'animal': animal, 'sections': sum(r['animal_id'] == animal for r in rows)}), flush=True)

summary = {}
for animal in ('AL1A', 'CC1B'):
    eligible = [r for r in rows if r['animal_id'] == animal and r['eligible_pair']]
    local = [r for r in eligible if 'local_mind' in r and
             abs(r['local_error_mm'][0] - r['local_error_mm'][1]) > .02]
    summary[animal] = {'sections': sum(r['animal_id'] == animal for r in rows),
        'eligible_pairs': len(eligible), 'prior_near_wins': sum(r['near_slot'] < r['far_slot'] for r in eligible),
        'local_eligible': len(local)}
    for metric in ('pearson', 'mind'):
        summary[animal][metric] = {'near_wins': sum(r[metric][0] > r[metric][1] for r in eligible),
            'null_near_wins': sum(r['null_' + metric][0] > r['null_' + metric][1] for r in eligible),
            'local_direction_wins': int(sum(np.argmax(r['local_' + metric]) ==
                np.argmin(r['local_error_mm']) for r in local))}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'source_sha256': sha(source),
    'pose_checkpoint_sha256': config['pose_checkpoint_sha256'],
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'trained': False, 'calibration_final_public_read': False}, indent=2))
print(json.dumps({'completed': summary}), flush=True)
