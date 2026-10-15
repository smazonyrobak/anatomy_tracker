"""Conditional real-field readout of the frozen 309 and 312 direct models."""
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
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
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out

SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/ACQUIRED_MAP_314_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
OUT = ROOT / 'reports/acquired_map_314_AL1A'
CHECKPOINTS = {
    'parent_309': ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt',
    'detached_control_312': ROOT / 'runs/paired_map_to_pose_312_001/detached_control/pose_local_00256.pt',
    'joint_feedback_312': ROOT / 'runs/paired_map_to_pose_312_001/joint_feedback/pose_local_00256.pt',
}
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert ARCHIVE.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
models = {name: load_one_shot_checkpoint(path, 'cuda')[0].eval().requires_grad_(False)
          for name, path in CHECKPOINTS.items()}
side, grid_side, animal = 192, 96, 'AL1A'
axis = (np.arange(grid_side) + .5) / grid_side - .5 / side
y, x = np.meshgrid(axis, axis, indexing='ij')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
choice = torch.zeros((1, 1), device='cuda', dtype=torch.long)
OUT.mkdir(parents=True)
config = {'animal_id': animal, 'label_role': 'inner TRAIN animal, approximate human plane and field',
          'source_doi': split['source_doi'], 'archive_md5_from_split': split['archive_md5'],
          'archive_bytes': split['archive_bytes'],
          'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
          'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'checkpoint_sha256': {name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                                for name, path in CHECKPOINTS.items()},
          'input_side': side, 'map_side': grid_side, 'field_sign': 'reference minus observed',
          'csv_sampling': '(grid centre / 96 - 0.5 / 192) * original side * 0.2 - 1',
          'target': 'human field after unweighted full-canvas affine projection',
          'zero_field_baseline': True, 'calibrated': False, 'public_benchmark_used': False}
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

rows = []
with ZipFile(ARCHIVE) as source_zip, (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
    valid = {item.findtext('name') for item in info.findall('slices')
             if item.findtext('valid') == '1'}
    registration = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
    for metadata in registration['slices']:
        section = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        image_name = f'{animal}/{metadata["filename"]}'
        image_bytes = source_zip.read(image_name)
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
            rms = {'zero': float(target[:, mask].square().sum(0).mean().sqrt() / 1000)}
            for name, model in models.items():
                prediction = model.predict(inputs)
                supplied = {**prediction, 'state': prediction['state'].clone()}
                supplied['state'][0, 0] = state
                field = model.map(supplied, offsets, choice, choice, (grid_side, grid_side),
                    atlas, weights, feature_side=grid_side, source_shape=(side, side))[
                    'local_displacement_um'][0, 0]
                rms[name] = float((field[:, mask] - target[:, mask]).square().sum(0).mean().sqrt() / 1000)
        row = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
               'section_id': section, 'source_thumbnail': image_name,
               'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
               'displacement_x_sha256': hashlib.sha256(dx_bytes).hexdigest(),
               'displacement_y_sha256': hashlib.sha256(dy_bytes).hexdigest(),
               'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
               'tissue_sites_96': int(mask.sum()), 'rms_mm': rms}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 26 == 0:
            stream.flush()
            print(json.dumps({'event': 'animal_milestone', 'sections': len(rows)}), flush=True)

assert len(rows) == 52
summary = {'sections': len(rows), 'rms_mm': {}, 'predeclared_gate': {}}
for name in ('zero', *models):
    values = np.asarray([row['rms_mm'][name] for row in rows])
    summary['rms_mm'][name] = {'mean': float(values.mean()),
        'median': float(np.median(values)), 'p90': float(np.quantile(values, .9))}
for name in models:
    zero = summary['rms_mm']['zero']['mean']
    parent = summary['rms_mm']['parent_309']['mean']
    score = summary['rms_mm'][name]['mean']
    worse = sum(row['rms_mm'][name] - row['rms_mm']['zero'] > .1 for row in rows)
    summary['predeclared_gate'][name] = {'passed': bool(score <= .9 * zero
        and score <= .9 * parent and worse <= 4), 'worse_than_zero_by_gt_0p1mm': worse}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': config['script_sha256'], 'protocol_sha256': config['protocol_sha256'],
    'output_sha256': {name: hashlib.file_digest((OUT / name).open('rb'), 'sha256').hexdigest()
                      for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
