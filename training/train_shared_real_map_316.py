"""Matched real-field map supervision with and without shared image-encoder updates."""
import hashlib
import json
import os
import sys
import time
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
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS

SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/SHARED_REAL_MAP_316_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PARENT = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
PARENT_DONE = PARENT.parent / 'completed.json'
PANEL = ROOT / 'runs/first_choice_exposure_309_001/panels.pt'
RUN = ROOT / 'runs/shared_real_map_316_001'
SEED, SIDE, GRID, UPDATES = 20261015316, 192, 96, 600
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert ARCHIVE.stat().st_size == split['archive_bytes']
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert sha(PARENT) == json.loads(PARENT_DONE.read_text(encoding='utf8'))[
    'output_sha256'][PARENT.name]
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model, _ = load_one_shot_checkpoint(PARENT, 'cuda')
assert model.modes == 80 and model.atlas_conditioning and model.spatial_feedback
initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
context = load_fov_balance_v10('cuda')
atlas = context['atlas']
axis = (np.arange(GRID) + .5) / GRID - .5 / SIDE
y, x = np.meshgrid(axis, axis, indexing='ij')
records, images, states, fields, masks = [], [], [], [], []
with ZipFile(ARCHIVE) as source_zip:
    for animal in ('CC2B', 'CC4B', 'CC4A'):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            source_zip.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        for row in registration['slices']:
            section = Path(row['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            image_bytes = source_zip.read(f'{animal}/{row["filename"]}')
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            width, height = image.size
            rgb = np.asarray(image.resize((SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.float32)
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
            images.append(gray)
            states.append(state)
            fields.append(project_surface_affine_out(local[None])[0][0])
            masks.append(np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
                (GRID, GRID), Image.Resampling.NEAREST)) > 127)
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
choice = torch.zeros((1, 1), dtype=torch.long, device='cuda')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
sources = ('train_shared_real_map_316.py', 'arbitrary_plane_fov_balance_v10.py',
    'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_inference.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_full_frame_primitives.py')
config = {'seed': SEED, 'batches_per_arm': UPDATES, 'input_side': SIDE, 'map_side': GRID,
    'gradient_animals': ['CC2B', 'CC4B', 'CC4A'], 'gradient_sections': len(records),
    'guard_animal': 'AL1A', 'synthetic_train_bases': [0, 55],
    'synthetic_guard_bases': [56, 63], 'real_loss': 'mean tissue-site vector distance mm',
    'synthetic_loss_weight': .25, 'real_condition': 'approximate human QuickNII pose',
    'synthetic_condition': 'postartifact gauge pose and reflection',
    'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'archive_bytes': split['archive_bytes'], 'split_sha256': sha(SPLIT),
    'parent_checkpoint_sha256': sha(PARENT), 'parent_completion_sha256': sha(PARENT_DONE),
    'panel_sha256': sha(PANEL), 'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE.with_name(name)) for name in sources},
    'calibrated': False, 'public_benchmark_used': False,
    'outside_pretrained_weights_features_pseudolabels_used': False}
RUN.mkdir(parents=True)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

signatures = None
for arm in ('map_only', 'shared_encoder'):
    model.load_state_dict(initial)
    model.eval().requires_grad_(False)
    map_modules = ('warp_shared', 'warp_condition', 'warp', 'pair', 'atlas_encoder')
    for name in map_modules:
        getattr(model, name).requires_grad_(True)
    if arm == 'shared_encoder':
        model.encoder.requires_grad_(True)
        model.lateral.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': model.warp_shared.parameters(), 'lr': 3e-5},
        {'params': [parameter for name in ('warp_condition', 'warp', 'pair', 'atlas_encoder')
                    for parameter in getattr(model, name).parameters()], 'lr': 1e-5},
        *([{'params': [parameter for name in ('encoder', 'lateral')
                        for parameter in getattr(model, name).parameters()], 'lr': 1e-6}]
          if arm == 'shared_encoder' else [])], weight_decay=1e-4)
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))
    cursor, epoch, attempts, draw_hashes = 0, 0, 0, []
    arm_path = RUN / arm
    arm_path.mkdir()

    def save(step):
        path = arm_path / f'map_step_{step:05d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'step': step, 'arm': arm,
                    'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'config': config, 'calibrated': False}, temp)
        os.replace(temp, path)

    save(0)
    started = time.perf_counter()
    with (arm_path / 'draws.jsonl').open('w', encoding='utf8') as draws, \
         (arm_path / 'training.jsonl').open('w', encoding='utf8') as log:
        for step in range(1, UPDATES + 1):
            if cursor == len(order):
                order = rng.permutation(len(records))
                cursor, epoch = 0, epoch + 1
            index = int(order[cursor])
            cursor += 1
            real_input = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
            real_input[0, 0] = images[index]
            prediction = model.predict(real_input)
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = states[index]
            real = model.map(supplied, offsets, choice, choice, (GRID, GRID),
                atlas, weights, feature_side=GRID, source_shape=(SIDE, SIDE))
            difference = (real['local_displacement_um'][0, 0] - fields[index]) / 1000
            real_loss = difference[:, masks[index]].square().sum(0).add(1e-8).sqrt().mean()

            while True:
                virtual = int(rng.integers(56)) * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = int(rng.integers(2**63 - 1))
                with torch.no_grad():
                    sample = sample_fov_balance_v10(context, [virtual], draw_seed, SIDE)
                attempts += 1
                if bool(sample['eligible'][0]):
                    break
            synthetic_hash = hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest()
            prediction = model.predict(sample['inputs'])
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = sample['state'][0]
            synthetic = model.map(supplied, sample['offsets'], choice,
                sample['reflection'][:, None].long(), (GRID, GRID), atlas,
                sample['weights'], feature_side=GRID, source_shape=(SIDE, SIDE))
            target = F.interpolate(sample['centre'].permute(0, 3, 1, 2),
                (GRID, GRID), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            valid = F.interpolate(sample['valid_mask'][:, None].float(),
                (GRID, GRID), mode='nearest')[:, 0].bool()
            difference = (synthetic['centre_surface_ccf_ap_dv_ml_um'][:, 0] - target) / 1000
            synthetic_loss = difference[valid].square().sum(-1).add(1e-8).sqrt().mean()
            loss = real_loss + .25 * synthetic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if arm == 'shared_encoder' and step == 1:
                assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                           for name in ('encoder', 'lateral')
                           for parameter in getattr(model, name).parameters())
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            identity = {'step': step, 'epoch': epoch, 'real': records[index],
                'synthetic': sample['provenance'][0], 'synthetic_virtual': virtual,
                'synthetic_seed': draw_seed, 'synthetic_input_sha256': synthetic_hash,
                'synthetic_attempts_cumulative': attempts}
            draw_hashes.append(hashlib.sha256(json.dumps(
                identity, sort_keys=True, allow_nan=False).encode()).hexdigest())
            draws.write(json.dumps(identity, allow_nan=False) + '\n')
            log.write(json.dumps({'step': step, 'real_map_mm': float(real_loss.detach()),
                'synthetic_map_mm': float(synthetic_loss.detach()),
                'loss': float(loss.detach()), 'gradient_norm': float(gradient.detach()),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if step in (300, 600):
                draws.flush()
                log.flush()
                save(step)
                print(json.dumps({'event': 'checkpoint', 'arm': arm, 'step': step,
                    'elapsed_seconds': round(time.perf_counter() - started, 1),
                    'real_map_mm': float(real_loss.detach()),
                    'synthetic_map_mm': float(synthetic_loss.detach())}), flush=True)
    if signatures is None:
        signatures = draw_hashes
    else:
        assert draw_hashes == signatures
    (arm_path / 'completed.json').write_text(json.dumps({'arm': arm,
        'steps': [0, 300, 600], 'gradient_sections': len(records),
        'synthetic_presentations': UPDATES, 'synthetic_attempts': attempts,
        'paired_draws_identical': arm == 'map_only' or draw_hashes == signatures,
        'config_sha256': sha(RUN / 'config.json'),
        'draws_sha256': sha(arm_path / 'draws.jsonl'),
        'training_sha256': sha(arm_path / 'training.jsonl'),
        'checkpoint_sha256': {str(step): sha(arm_path / f'map_step_{step:05d}.pt')
                              for step in (0, 300, 600)},
        'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
    print(json.dumps({'event': 'arm_completed', 'arm': arm,
        'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)

(RUN / 'completed.json').write_text(json.dumps({'arms': ['map_only', 'shared_encoder'],
    'paired_draws_identical': True, 'source_sha256': config['source_sha256'],
    'protocol_sha256': config['protocol_sha256'], 'config_sha256': sha(RUN / 'config.json'),
    'arm_completion_sha256': {arm: sha(RUN / arm / 'completed.json')
                              for arm in ('map_only', 'shared_encoder')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'batches_per_arm': UPDATES}), flush=True)
