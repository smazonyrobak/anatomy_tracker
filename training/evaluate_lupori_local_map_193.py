"""Frozen AL1A acquired-field guard for real-adapted local mapper."""

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
from PIL import Image
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt'
train = root / 'runs/lupori_local_map_193_inner_AL1A'
checkpoint = train / 'map_step_00600.pt'
out = root / 'runs/lupori_local_map_193_guard_AL1A'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
done = json.loads((train / 'completed.json').read_text())
assert done['updates'] == 600 and not done['guard_animal_used_in_training']
assert sha(checkpoint) == done['checkpoint_sha256']['600']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 800
model.load_state_dict(saved['model'], strict=True)
del saved
import copy
adapted = copy.deepcopy(model).eval().requires_grad_(False)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 600 and json.loads(json.dumps(saved['config'])) == json.loads(
    (train / 'config.json').read_text())
for name, weights in saved['modules'].items():
    getattr(adapted, name).load_state_dict(weights, strict=True)
del saved

side, grid_side = 256, 96
axis = (np.arange(grid_side) + .5) / grid_side - .5 / side
y, x = np.meshgrid(axis, axis, indexing='ij')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
choice = torch.zeros((1, 1), device='cuda', dtype=torch.long)
animal = 'AL1A'
out.mkdir(parents=True, exist_ok=False)
config = {'source_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'checkpoint_sha256': sha(checkpoint), 'parent_checkpoint_sha256': sha(parent),
    'split_sha256': sha(split_path), 'source_doi': split['source_doi'],
    'animal_id': animal, 'label_role': 'inner TRAIN-animal human field, affine-free gauge',
    'map_side': grid_side, 'evaluated_step': 600,
    'public_benchmark_used': False, 'calibrated': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
    valid = {row.findtext('name') for row in info.findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
    for metadata in visualign['slices']:
        section = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        image_bytes = source_zip.read(f'{animal}/{metadata["filename"]}')
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        width, height = image.size
        rgb = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = torch.from_numpy(gray).cuda()
        q = torch.tensor(metadata['anchoring'], dtype=torch.float32).reshape(3, 3)
        ouv = torch.stack((quicknii_to_allen_points(q[0]),
            quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
        state = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).cuda()
        frame = full_frame_state_to_components(state)[1]
        ouv = ouv.cuda()
        dx_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldX.csv')
        dy_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldY.csv')
        mask_bytes = source_zip.read(f'{animal}/masks/{section}-mask.png')
        dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
        dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
        query = [y * height * .2 - 1, x * width * .2 - 1]
        sampled_x = map_coordinates(dx, query, order=1, mode='nearest')
        sampled_y = map_coordinates(dy, query, order=1, mode='nearest')
        displacement = (torch.from_numpy(sampled_x).cuda().float()[..., None] * ouv[1] / width
            + torch.from_numpy(sampled_y).cuda().float()[..., None] * ouv[2] / height)
        local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
        target = project_surface_affine_out(local[None])[0][0]
        mask = torch.from_numpy(np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
            (grid_side, grid_side), Image.Resampling.NEAREST)) > 127).cuda()
        with torch.inference_mode():
            prediction = model.predict(inputs)
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = state
            old = model.map(supplied, offsets, choice, choice, (grid_side, grid_side),
                atlas, weights, feature_side=grid_side, source_shape=(side, side))[
                'local_displacement_um'][0, 0]
            new = adapted.map(supplied, offsets, choice, choice, (grid_side, grid_side),
                atlas, weights, feature_side=grid_side, source_shape=(side, side))[
                'local_displacement_um'][0, 0]
            zero_mm = target[:, mask].square().sum(0).mean().sqrt() / 1000
            parent_mm = (old[:, mask] - target[:, mask]).square().sum(0).mean().sqrt() / 1000
            adapted_mm = (new[:, mask] - target[:, mask]).square().sum(0).mean().sqrt() / 1000
        row = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'section_id': section, 'source_thumbnail': f'{animal}/{metadata["filename"]}',
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'displacement_x_sha256': hashlib.sha256(dx_bytes).hexdigest(),
            'displacement_y_sha256': hashlib.sha256(dy_bytes).hexdigest(),
            'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
            'tissue_sites_96': int(mask.sum()),
            'zero_local_rms_mm': float(zero_mm),
            'parent_local_rms_mm': float(parent_mm),
            'adapted_local_rms_mm': float(adapted_mm)}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 13 == 0:
            stream.flush()
            print(json.dumps({'event': 'guard_milestone', 'sections': len(rows)}), flush=True)

assert len(rows) == 52
summary = {'sections': len(rows), 'rms_mm': {}}
for name in ('zero', 'parent', 'adapted'):
    values = np.asarray([row[f'{name}_local_rms_mm'] for row in rows])
    summary['rms_mm'][name] = {'mean': float(values.mean()),
        'median': float(np.median(values)), 'p90': float(np.quantile(values, .9))}
zero, parent_score, candidate = (summary['rms_mm'][name]['mean']
    for name in ('zero', 'parent', 'adapted'))
worse = sum(row['adapted_local_rms_mm'] - row['zero_local_rms_mm'] > .1 for row in rows)
summary['predeclared_gate'] = {'passed': bool(candidate <= .9 * zero and
    candidate <= .9 * parent_score and worse <= 4),
    'worse_than_zero_by_gt_0p1mm': worse}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'public_benchmark_used': False, 'calibrated': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
