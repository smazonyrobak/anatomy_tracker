"""Real-field local mapper adaptation with independent full-angle rehearsal."""

import hashlib
import json
import os
import sys
import time
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
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/LUPORI_LOCAL_MAP_ADAPTATION_193_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
run = root / 'runs/lupori_local_map_193_inner_AL1A'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert sha(parent) == json.loads((parent_run / 'completed.json').read_text())[
    'checkpoint_sha256']['800']
assert archive.stat().st_size == split['archive_bytes']
seed, updates, side, grid_side = 20261011193, 600, 256, 96
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 800 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
model.requires_grad_(False)
trainable = ('warp', 'warp_condition', 'pair', 'atlas_encoder')
for name in trainable:
    getattr(model, name).requires_grad_(True)
optimizer = torch.optim.AdamW([
    {'params': model.warp.parameters(), 'lr': 3e-5},
    {'params': [parameter for name in ('warp_condition', 'pair', 'atlas_encoder')
        for parameter in getattr(model, name).parameters()], 'lr': 1e-5}],
    weight_decay=1e-4)
parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]

axis = (np.arange(grid_side) + .5) / grid_side - .5 / side
y, x = np.meshgrid(axis, axis, indexing='ij')
records, images, states, fields, masks = [], [], [], [], []
with ZipFile(archive) as source_zip:
    for animal in split['train']:
        if animal == 'AL1A':
            continue
        info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
        valid = {node.findtext('name') for node in info.findall('slices')
                 if node.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        for row in visualign['slices']:
            section = Path(row['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            image_bytes = source_zip.read(f'{animal}/{row["filename"]}')
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            width, height = image.size
            rgb = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            q = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
            ouv = torch.stack((quicknii_to_allen_points(q[0]),
                quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
            state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
            frame = full_frame_state_to_components(state)[1]
            dx_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldX.csv')
            dy_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldY.csv')
            mask_bytes = source_zip.read(f'{animal}/masks/{section}-mask.png')
            dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
            dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
            query = [y * height * .2 - 1, x * width * .2 - 1]
            sampled_x = map_coordinates(dx, query, order=1, mode='nearest')
            sampled_y = map_coordinates(dy, query, order=1, mode='nearest')
            displacement = (torch.from_numpy(sampled_x).float()[..., None] * ouv[1] / width
                + torch.from_numpy(sampled_y).float()[..., None] * ouv[2] / height)
            local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
            residual = project_surface_affine_out(local[None])[0][0]
            mask = np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
                (grid_side, grid_side), Image.Resampling.NEAREST)) > 127
            images.append(gray)
            states.append(state)
            fields.append(residual)
            masks.append(mask)
            records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'section_id': section,
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'displacement_x_sha256': hashlib.sha256(dx_bytes).hexdigest(),
                'displacement_y_sha256': hashlib.sha256(dy_bytes).hexdigest(),
                'mask_sha256': hashlib.sha256(mask_bytes).hexdigest()})
assert len(records) == 188 and len({(row['animal_id'], row['section_id']) for row in records}) == 188
images = torch.from_numpy(np.stack(images)).cuda()
states = torch.stack(states).cuda()
fields = torch.stack(fields).cuda()
masks = torch.from_numpy(np.stack(masks)).cuda()
choice = torch.zeros((1, 1), device='cuda', dtype=torch.long)
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
source_names = ('train_lupori_local_map_193.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_ribbon_v6.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'map_side': grid_side,
    'train_animals': ['CC2B', 'CC4B', 'CC4A'], 'guard_animal': 'AL1A',
    'train_sections': len(records), 'source_doi': split['source_doi'],
    'archive_md5': split['archive_md5'], 'split_sha256': sha(split_path),
    'protocol_sha256': sha(protocol), 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'source_sha256': {name: sha(source.parent / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'input': 'whole RGB 256, max(red,green)/255, image-wide p99, four channels zero',
    'real_target': 'human displacement projected out of unweighted full-canvas affine gauge',
    'synthetic_target': 'one independently sampled v4 physical plane and mapped tissue',
    'optimized': trainable, 'checkpoints': [0, 300, 600],
    'calibrated': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False, 'development_calibration_final_read': False}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    path = run / f'map_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'modules': {name: getattr(model, name).state_dict()
        for name in trainable}, 'optimizer': optimizer.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
        'calibrated': False}, temp)
    os.replace(temp, path)


save(0)
order = rng.permutation(len(records))
cursor, epoch, attempts = 0, 0, 0
started = time.perf_counter()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        if cursor == len(order):
            order = rng.permutation(len(records))
            cursor = 0
            epoch += 1
        index = int(order[cursor])
        cursor += 1
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = images[index]
        with torch.no_grad():
            prediction = model.predict(inputs)
        supplied = {**prediction, 'state': prediction['state'].clone()}
        supplied['state'][0, 0] = states[index]
        real = model.map(supplied, offsets, choice, choice, (grid_side, grid_side),
            atlas, weights, feature_side=grid_side, source_shape=(side, side))
        predicted = real['local_displacement_um'][0, 0]
        diff = (predicted - fields[index]) / 1000
        real_loss = diff[:, masks[index]].square().sum(0).add(1e-8).sqrt().mean()

        while True:
            virtual = int(rng.integers(len(context['subjects'])))
            draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            with torch.no_grad():
                sample = sample_one_shot_slide_artifacts_v4(
                    context, [virtual], draw_seed, side=side)
            attempts += 1
            if bool(sample['eligible'][0]):
                break
        with torch.no_grad():
            prediction_synthetic = model.predict(sample['inputs'])
        synthetic_state = {**prediction_synthetic,
            'state': prediction_synthetic['state'].clone()}
        synthetic_state['state'][0, 0] = sample['state'][0]
        synthetic = model.map(synthetic_state, sample['offsets'], choice,
            sample['reflection'][:, None].long(), (grid_side, grid_side),
            atlas, sample['weights'], feature_side=grid_side, source_shape=(side, side))
        target = F.interpolate(sample['centre'].permute(0, 3, 1, 2),
            (grid_side, grid_side), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        valid = F.interpolate(sample['valid_mask'][:, None].float(),
            (grid_side, grid_side), mode='nearest')[:, 0].bool()
        synthetic_diff = (synthetic['centre_surface_ccf_ap_dv_ml_um'][:, 0] - target) / 1000
        synthetic_loss = synthetic_diff[valid].square().sum(-1).add(1e-8).sqrt().mean()
        loss = real_loss + .5 * synthetic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        draws.write(json.dumps({'update': step, 'epoch': epoch, 'real': records[index],
            'synthetic': sample['provenance'][0], 'synthetic_draw_seed': draw_seed,
            'synthetic_draw_attempts_cumulative': attempts}, allow_nan=False) + '\n')
        log.write(json.dumps({'update': step, 'real_map_mm': float(real_loss.detach()),
            'synthetic_map_mm': float(synthetic_loss.detach()),
            'loss': float(loss.detach()), 'gradient_norm': float(gradient.detach())},
            allow_nan=False) + '\n')
        if step in (300, 600):
            draws.flush()
            log.flush()
            save(step)
            print(json.dumps({'event': 'checkpoint', 'step': step,
                'elapsed_seconds': round(time.perf_counter() - started, 1),
                'real_map_mm': float(real_loss.detach()),
                'synthetic_map_mm': float(synthetic_loss.detach())}), flush=True)

done = {'updates': updates, 'gradient_animals': ['CC2B', 'CC4B', 'CC4A'],
    'gradient_sections': len(records), 'guard_animal_used_in_training': False,
    'synthetic_presentations': updates, 'synthetic_draw_attempts': attempts,
    'source_sha256': config['source_sha256'], 'protocol_sha256': config['protocol_sha256'],
    'config_sha256': sha(run / 'config.json'), 'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'map_step_{step:05d}.pt')
        for step in (0, 300, 600)}, 'calibrated': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False}
(run / 'completed.json').write_text(json.dumps(done, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
    'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)
