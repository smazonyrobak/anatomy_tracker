"""TRAIN-only supplied-plane comparison of inherited and human local fields."""

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
protocol = source.parents[1] / 'docs/publication/LUPORI_ACQUIRED_LOCAL_MAP_192_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent = root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt'
previous = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'
out = root / 'runs/lupori_acquired_local_map_192_train_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert sha(parent) == json.loads((root /
    'runs/lupori_fluorescence_pose_188_all_train/completed.json').read_text())[
    'checkpoint_sha256']['1200']
prior_done = json.loads((previous / 'completed.json').read_text())
assert sha(previous / 'cases.jsonl') == prior_done['output_sha256']['cases.jsonl']
selected = [json.loads(line) for line in (previous / 'cases.jsonl').read_text().splitlines()]
assert len(selected) == 64
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 1200 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

side, grid_side = 256, 96
y, x = np.meshgrid((np.arange(grid_side) + .5) / grid_side - .5 / side,
                   (np.arange(grid_side) + .5) / grid_side - .5 / side, indexing='ij')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
choice = torch.zeros((1, 1), device='cuda', dtype=torch.long)
config = {'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'source_sha256': sha(source), 'protocol_sha256': sha(protocol),
    'split_sha256': sha(split_path), 'checkpoint_sha256': sha(parent),
    'previous_189_completion_sha256': sha(previous / 'completed.json'),
    'previous_189_cases_sha256': sha(previous / 'cases.jsonl'),
    'animals': split['train'], 'sections_per_animal': 16, 'grid_side': grid_side,
    'field_sign': 'reference minus observed original thumbnail pixels',
    'field_sampling': '96-grid normalized pixel centres times original dimensions and 0.2, minus 1 CSV index',
    'target': 'human local field after unweighted full-canvas affine projection',
    'trained': False, 'calibrated': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    for previous_row in selected:
        animal, section = previous_row['animal_id'], previous_row['section_id']
        image_bytes = source_zip.read(previous_row['source_thumbnail'])
        assert hashlib.sha256(image_bytes).hexdigest() == previous_row['source_thumbnail_sha256']
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        width, height = image.size
        rgb = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = torch.from_numpy(gray).cuda()
        metadata = next(row for row in json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))[
            'slices'] if f'{animal}/{row["filename"]}' == previous_row['source_thumbnail'])
        q = torch.tensor(metadata['anchoring'], device='cuda', dtype=torch.float32).reshape(3, 3)
        ouv = torch.stack((quicknii_to_allen_points(q[0]),
            quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
        state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
        frame = full_frame_state_to_components(state)[1]
        dx_name = f'{animal}/dispFields/{section}-dispFieldX.csv'
        dy_name = f'{animal}/dispFields/{section}-dispFieldY.csv'
        mask_name = f'{animal}/masks/{section}-mask.png'
        dx_bytes, dy_bytes = source_zip.read(dx_name), source_zip.read(dy_name)
        mask_bytes = source_zip.read(mask_name)
        dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
        dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
        query = [y * height * .2 - 1, x * width * .2 - 1]
        sampled_dx = map_coordinates(dx, query, order=1, mode='nearest')
        sampled_dy = map_coordinates(dy, query, order=1, mode='nearest')
        mask = np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
            (grid_side, grid_side), Image.Resampling.NEAREST)) > 127
        displacement = (torch.from_numpy(sampled_dx).cuda().float()[..., None] * ouv[1] / width
            + torch.from_numpy(sampled_dy).cuda().float()[..., None] * ouv[2] / height)
        local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
        residual, affine = project_surface_affine_out(local[None])
        with torch.inference_mode():
            prediction = model.predict(inputs)
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = state
            mapped = model.map(supplied, offsets, choice, choice,
                (grid_side, grid_side), atlas, weights, feature_side=grid_side,
                source_shape=(side, side))
            predicted = mapped['local_displacement_um'][0, 0]
            target = residual[0]
            valid = torch.from_numpy(mask).cuda()
            zero_mm = (target[:, valid].square().sum(0).mean().sqrt() / 1000)
            model_mm = ((predicted[:, valid] - target[:, valid]
                ).square().sum(0).mean().sqrt() / 1000)
            prediction_mm = predicted[:, valid].square().sum(0).mean().sqrt() / 1000
            target_p95_mm = target[:, valid].square().sum(0).sqrt().quantile(.95) / 1000
        row = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'section_id': section, 'source_thumbnail': previous_row['source_thumbnail'],
            'source_thumbnail_sha256': previous_row['source_thumbnail_sha256'],
            'displacement_x_sha256': hashlib.sha256(dx_bytes).hexdigest(),
            'displacement_y_sha256': hashlib.sha256(dy_bytes).hexdigest(),
            'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
            'width': width, 'height': height, 'tissue_sites_96': int(mask.sum()),
            'zero_local_rms_mm': float(zero_mm), 'model_local_rms_mm': float(model_mm),
            'predicted_local_rms_mm': float(prediction_mm),
            'target_local_p95_mm': float(target_p95_mm),
            'removed_affine_local_um': affine[0].tolist()}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 16 == 0:
            stream.flush()
            print(json.dumps({'event': 'animal_milestone', 'sections': len(rows),
                'animal_id': animal}), flush=True)

by_animal = {}
for animal in split['train']:
    group = [row for row in rows if row['animal_id'] == animal]
    by_animal[animal] = {'sections': len(group),
        'zero_mean_mm': float(np.mean([row['zero_local_rms_mm'] for row in group])),
        'model_mean_mm': float(np.mean([row['model_local_rms_mm'] for row in group]))}
zero = float(np.mean([by_animal[animal]['zero_mean_mm'] for animal in split['train']]))
model_error = float(np.mean([by_animal[animal]['model_mean_mm'] for animal in split['train']]))
better_animals = sum(by_animal[animal]['model_mean_mm'] < by_animal[animal]['zero_mean_mm']
    for animal in split['train'])
summary = {'sections': len(rows), 'animal': by_animal,
    'zero_equal_animal_mean_mm': zero, 'model_equal_animal_mean_mm': model_error,
    'predeclared_gate': {'passed': bool(model_error <= .9 * zero and better_animals >= 3),
        'animals_improved': better_animals}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': sha(source), 'checkpoint_sha256': sha(parent),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'trained': False, 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}, allow_nan=False), flush=True)
