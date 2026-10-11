"""Matched TRAIN-only rigid MIND-to-pose continuation from Lupori 187 step 800."""

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
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mind(image):
    side = image.shape[-1]
    padded = F.pad(image, (3, 3, 3, 3), mode='replicate')
    distances = torch.cat([F.avg_pool2d((image - padded[:, :, 3 + dy:3 + dy + side,
        3 + dx:3 + dx + side]).square(), 3, 1, 1) for dy, dx in
        ((3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (3, -3), (-3, 3), (-3, -3))], 1)
    return torch.exp(-distances / distances.mean(1, keepdim=True).clamp_min(1e-4))


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/LUPORI_LOCAL_FIT_TO_POSE_206_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
run = root / 'runs/lupori_local_fit_to_pose_206_matched_001'
seed, side, grid, updates, sites = 20261011206, 256, 96, 400, 128
milestones = (0, 200, 400)
train_animals = ('CC2B', 'CC4B', 'CC4A')
assert source.drive.upper() == root.drive.upper() == run.drive.upper() == 'I:'
assert not run.exists() and archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', *train_animals]
assert split['development'] == ['CC1B'] and split['calibration'] == ['CC3A']
assert split['final_test'] == ['CC1A']
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['800']
assert not any(parent_done.get(key, False) for key in
    ('calibrated', 'public_benchmark_used', 'external_pretrained_weights_used',
     'heldout_animal_used_in_training'))
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
parent_config = json.loads((parent_run / 'config.json').read_text())
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']

lupori_records, lupori_images, lupori_masks, lupori_ouv = [], [], [], []
with ZipFile(archive) as source_zip:
    for animal in train_animals:
        info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        for row in visualign['slices']:
            section_id = Path(row['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            image_bytes = source_zip.read(f'{animal}/{row["filename"]}')
            mask_bytes = source_zip.read(f'{animal}/masks/{section_id}-mask.png')
            rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
                (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            mask = np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
                (side, side), Image.Resampling.NEAREST)) > 127
            q = torch.tensor(row['anchoring'], dtype=torch.float64).reshape(3, 3)
            o = quicknii_to_allen_points(q[0]) * 25
            u = quicknii_to_allen_vectors(q[1]) * 25
            v = quicknii_to_allen_vectors(q[2]) * 25
            lupori_images.append(gray.astype(np.float16))
            lupori_masks.append(mask)
            lupori_ouv.append(torch.stack((o, u, v)))
            lupori_records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'section_id': section_id,
                'source_thumbnail': f'{animal}/{row["filename"]}',
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'source_mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
                'source_doi': split['source_doi']})
assert len(lupori_records) == 188
assert len({(r['animal_id'], r['section_id']) for r in lupori_records}) == 188
lupori_images = torch.from_numpy(np.stack(lupori_images)).to('cuda', dtype=torch.float32)
lupori_masks = torch.from_numpy(np.stack(lupori_masks)).to('cuda')
lupori_ouv = torch.stack(lupori_ouv).to('cuda', dtype=torch.float32)
lupori_normals = F.normalize(torch.cross(lupori_ouv[:, 1], lupori_ouv[:, 2], dim=-1), dim=-1)

source_files = ('train_lupori_local_fit_to_pose_206.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates_per_arm': updates, 'checkpoints': milestones,
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_optimizer': 'AdamW moments restored separately in both arms',
    'model': 'same final OneShotJointSliceModel, 16 base + 64 normal-anchor modes',
    'trainable_for_direct_loss': ['pose', 'anchor_pose', 'anchor_global', 'encoder', 'lateral'],
    'fit_gradient_target': 'final pose/anchor output geometry rows only; no shared hidden or probability rows',
    'frozen': ['atlas matcher', 'local map', 'uncertainty', 'pose refiner', 'other heads'],
    'synthetic_per_update': {'v7': 2, 'v3': 1},
    'synthetic_provenance': context['provenance'],
    'lupori_train_animals': train_animals, 'lupori_sections': len(lupori_records),
    'inner_heldout_animal': 'AL1A', 'split_sha256': sha(split_path),
    'lupori_archive_bytes': split['archive_bytes'], 'lupori_archive_md5': split['archive_md5'],
    'source_doi': split['source_doi'], 'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source.with_name(name)) for name in source_files},
    'fit': 'rigid 5-offset MIND at current prior top-1, reference error <=1.5 mm; '
           'detached 7x7-eroded predicted atlas support, at least 400 pixels',
    'fit_gradient_cap': 'at most 0.2 times same-step supervised pose-head gradient norm',
    'fit_into_global_branch_mass': False,
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_read': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
flags = torch.arange(2, device='cuda')[None, None]
state_scale = torch.tensor([1000., 1000., 1000., 1., 1., 1., 1., 1., 1.,
                            1., 1., 1.], device='cuda')
matched_draws = []

for arm in ('detached', 'fit'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    site_generator = torch.Generator(device='cuda').manual_seed(seed)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().train()
    attach_global_plane_matcher(model, enabled=True)
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    assert saved['step'] == 800 and saved['calibrated'] is False
    assert saved['config'] == parent_config
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    for module in (*pose_modules, *shared_modules):
        module.requires_grad_(True)
    groups = [
        {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 2e-5},
        {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 5e-6},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    pose_parameters = [p for module in pose_modules for p in module.parameters()]
    geometry_parameters = [p for module in pose_modules for p in
        (module[-1].weight, module[-1].bias)]
    geometry_masks = []
    for module, rows, width in ((model.pose, model.base_modes, 21),
                                (model.anchor_pose, model.normal_anchor_count, 18),
                                (model.anchor_global, model.normal_anchor_count, 18)):
        for parameter in (module[-1].weight, module[-1].bias):
            mask = torch.zeros_like(parameter).reshape(rows, width, *parameter.shape[1:])
            mask[:, :12 if width == 21 else 9] = 1
            geometry_masks.append(mask.reshape_as(parameter))
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert all(not p.requires_grad for name, p in model.named_parameters()
               if not name.startswith(('pose.', 'anchor_pose.', 'anchor_global.',
                                       'encoder.', 'lateral.')))
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * .4
    order = rng.permutation(len(lupori_records))
    cursor, epoch = 0, 0
    checkpoint_path = directory / 'joint_step_00000.pt'
    temporary_path = checkpoint_path.with_suffix('.tmp')
    torch.save({'step': 0, 'arm': arm, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'site_rng': site_generator.get_state(), 'order': order, 'cursor': cursor,
        'epoch': epoch, 'config': config, 'calibrated': False},
        temporary_path)
    os.replace(temporary_path, checkpoint_path)
    started = time.perf_counter()
    draw_count, fit_eligible_count = 0, 0
    with (directory / 'draws.jsonl').open('w') as draws, (directory / 'training.jsonl').open('w') as log:
        for step in range(1, updates + 1):
            samples, signature = [], []
            for slot, appearance in enumerate(('v7', 'v7', 'v3')):
                sampler = (sample_one_shot_slide_artifacts_v7 if appearance == 'v7'
                           else sample_one_shot_slide_artifacts_v3)
                while True:
                    virtual = int(rng.integers(len(context['subjects'])))
                    draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                    with torch.no_grad():
                        sample = sampler(context, [virtual], draw_seed, side=side)
                    eligible = bool(sample['eligible'][0])
                    identity = sample['provenance'][0]
                    draw_count += 1
                    signature.append((slot, appearance, virtual, draw_seed,
                                      identity['physical_section_id'], eligible))
                    draws.write(json.dumps({'kind': 'synthetic_train', 'step': step,
                        'slot': slot, 'appearance': appearance, 'virtual': virtual,
                        'draw_seed': draw_seed, 'draw_attempt': draw_count,
                        'used': eligible, **identity}, allow_nan=False) + '\n')
                    if eligible:
                        samples.append(sample)
                        break
            if cursor == len(order):
                order = rng.permutation(len(lupori_records))
                cursor, epoch = 0, epoch + 1
            index = int(order[cursor])
            cursor += 1
            signature.append(('lupori', index, epoch))
            if arm == 'detached':
                matched_draws.append(signature)
            else:
                assert signature == matched_draws[step - 1]
            real_input = torch.zeros(1, 5, side, side, device='cuda')
            real_input[0, 0] = lupori_images[index]
            draws.write(json.dumps({'kind': 'lupori_train', 'step': step, 'slot': 3,
                'epoch': epoch, 'used': True, **lupori_records[index]}, allow_nan=False) + '\n')
            inputs = torch.cat([sample['inputs'] for sample in samples] + [real_input])
            target_state = torch.cat([sample['state'] for sample in samples])
            target_reflection = torch.cat([sample['reflection'] for sample in samples])
            valid = torch.cat([sample['valid_mask'] for sample in samples])
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(states), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

            pixel = torch.multinomial(valid.flatten(1).float(), sites,
                                      replacement=True, generator=site_generator)
            tissue_chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            reference = rigid_points_090(target_state, target_reflection, tissue_chart)
            proposed = rigid_points_090(states[:3], reflections[:3], tissue_chart[:, None, None])
            reference_five = rigid_points_090(target_state, target_reflection, five)
            proposed_five = rigid_points_090(states[:3], reflections[:3], five)
            tissue_um = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
            predicted_normal = full_frame_state_to_components(prediction['state'][:3])[1][..., :, 2]
            true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
            normal_penalty = 4000 * (1 - (predicted_normal * true_normal[:, None]
                ).sum(-1).abs().clamp_max(1))
            physical = (.75 * tissue_um + .25 * five_um + normal_penalty[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(3, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            synthetic_loss = (-1.5 * torch.logsumexp(prior[:3] - physical / 1.5, -1)
                + .5 * F.kl_div(prior[:3], target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1)).mean()

            real_pixel = torch.multinomial(lupori_masks[index].flatten().float(), sites,
                replacement=True, generator=site_generator)
            real_chart = torch.stack((real_pixel.remainder(side),
                real_pixel.div(side, rounding_mode='floor')), -1).float() / side
            ouv = lupori_ouv[index]
            real_reference = ouv[0] + real_chart[:, :1] * ouv[1] + real_chart[:, 1:] * ouv[2]
            real_proposed = rigid_points_090(states[3:4], reflections[3:4], real_chart)
            real_reference_five = ouv[0] + five[:, :1] * ouv[1] + five[:, 1:] * ouv[2]
            real_proposed_five = rigid_points_090(states[3:4], reflections[3:4], five)
            real_tissue = (real_proposed[0] - real_reference).norm(dim=-1).mean(-1)
            real_five = (real_proposed_five[0] - real_reference_five).norm(dim=-1).mean(-1)
            real_normal = full_frame_state_to_components(prediction['state'][3:4])[1][..., :, 2][0]
            real_normal_penalty = 4000 * (1 - (real_normal * lupori_normals[index]
                ).sum(-1).abs().clamp_max(1))
            real_cost = (.75 * real_tissue + .25 * real_five + real_normal_penalty[:, None]
                ).flatten() / 1000
            real_mass = F.softmax(-real_cost.detach() / .7, -1)
            real_nearest = (lupori_normals[index] @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4)
            real_near_cost = real_cost.reshape(model.modes, 2).min(-1).values[
                model.base_modes + real_nearest.indices]
            real_neighbourhood = F.softmax(40 * (real_nearest.values - real_nearest.values[:1]), -1)
            real_loss = (-1.5 * torch.logsumexp(prior[3] - real_cost / 1.5, -1)
                + .5 * F.kl_div(prior[3], real_mass, reduction='sum')
                + .5 * real_cost[:2 * model.base_modes].min()
                + (real_neighbourhood * real_near_cost).sum())
            supervised = synthetic_loss + real_loss

            selected_branch = int(prior[3].detach().argmax())
            selected_mode, selected_reflection = divmod(selected_branch, 2)
            selected_error_mm = float(real_five.detach().flatten()[selected_branch] / 1000)
            captured = selected_error_mm <= 1.5
            fit = None
            support_sites = 0
            direction_cosine = None
            direction_dot = None
            fit_grads = None
            if captured:
                selected_state = prediction['state'][3, selected_mode][None]
                selected_flag = torch.tensor([selected_reflection], device='cuda')
                plane = rigid_points_090(selected_state, selected_flag, chart).reshape(1, grid, grid, 3)
                normal = full_frame_state_to_components(selected_state)[1][..., :, 2]
                world = plane[:, None] + offsets[None, :, None, None, None] * normal[:, None, None, None, :]
                rigid = render_finite_thickness_coordinate_grid(atlas, world,
                    (0., 0., 0.), (25., 25., 25.), weights[None])
                support = rigid[:, 1:2].detach().clamp(0, 1)
                mask = (support > .5).float()
                mask = (F.avg_pool2d(mask, 7, 1, 3) > .999).float().detach()
                support_sites = int(mask.sum())
                if support_sites >= 400:
                    observed = F.interpolate(real_input[:, :1], (grid, grid), mode='area')
                    rendered = rigid[:, :1] / support.clamp_min(1e-4)
                    fit = (((mind(observed) - mind(rendered)).square().mean(1)
                        * mask[:, 0]).sum() / mask.sum())
                    assert bool(torch.isfinite(fit))
                    fit_state_gradient = torch.autograd.grad(fit, prediction['state'],
                        retain_graph=True)[0][3, selected_mode] * state_scale
                    weak_state_gradient = torch.autograd.grad(real_cost[selected_branch],
                        prediction['state'], retain_graph=True)[0][3, selected_mode] * state_scale
                    direction_dot = float((fit_state_gradient * weak_state_gradient).sum().detach())
                    direction_cosine = float((fit_state_gradient * weak_state_gradient).sum().detach() /
                        (fit_state_gradient.norm() * weak_state_gradient.norm()).clamp_min(1e-12))
                    assert np.isfinite(direction_dot) and np.isfinite(direction_cosine)
                    assert torch.autograd.grad(fit,
                        (prediction['log_mass'], prediction['reflection_logit']),
                        retain_graph=True, allow_unused=True) == (None, None)
                    if arm == 'fit':
                        fit_grads = torch.autograd.grad(fit, geometry_parameters,
                            retain_graph=True, allow_unused=True)
                    fit_eligible_count += 1

            assert bool(torch.isfinite(supervised))
            supervised.backward()
            supervised_pose_norm = torch.stack([p.grad.square().sum() for p in pose_parameters
                if p.grad is not None]).sum().sqrt()
            supervised_geometry_norm = torch.stack([(p.grad * mask).square().sum()
                for p, mask in zip(geometry_parameters, geometry_masks)]).sum().sqrt()
            assert bool(torch.isfinite(supervised_pose_norm))
            assert bool(torch.isfinite(supervised_geometry_norm))
            fit_raw_norm, fit_scale, fit_applied_norm = None, 0., 0.
            fit_probability_row_gradient_max = None
            if fit_grads is not None:
                assert all(g is None or bool(torch.isfinite(g).all()) for g in fit_grads)
                fit_probability_row_gradient_max = max(float((g * (1 - mask)).abs().max())
                    for g, mask in zip(fit_grads, geometry_masks) if g is not None)
                assert fit_probability_row_gradient_max == 0.
                fit_raw_norm_tensor = torch.stack([(g * mask).square().sum()
                    for g, mask in zip(fit_grads, geometry_masks) if g is not None]).sum().sqrt()
                fit_scale = float(torch.minimum(torch.ones_like(fit_raw_norm_tensor),
                    .2 * supervised_geometry_norm / fit_raw_norm_tensor.clamp_min(1e-12)))
                fit_raw_norm = float(fit_raw_norm_tensor)
                fit_applied_norm = fit_scale * fit_raw_norm
                for parameter, gradient, mask in zip(geometry_parameters, fit_grads, geometry_masks):
                    if gradient is not None:
                        parameter.grad.add_(gradient * mask, alpha=fit_scale)
                assert fit_applied_norm <= .2 * float(supervised_geometry_norm) + 1e-5
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
            total_gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()

            row = {'arm': arm, 'step': step, 'lupori_epoch': epoch,
                'lupori_animal_id': lupori_records[index]['animal_id'],
                'lupori_section_id': lupori_records[index]['section_id'],
                'synthetic_section_ids': [sample['provenance'][0]['physical_section_id']
                    for sample in samples],
                'synthetic_selected_mm': physical.detach().gather(1,
                    prior[:3].detach().argmax(-1)[:, None])[:, 0].tolist(),
                'synthetic_oracle_all160_mm': physical.detach().min(-1).values.tolist(),
                'lupori_selected_branch': selected_branch,
                'lupori_selected_five_point_mm': selected_error_mm,
                'lupori_selected_composite_mm': float(real_cost.detach()[selected_branch]),
                'lupori_oracle_all160_composite_mm': float(real_cost.detach().min()),
                'fit_reference_gate_le_1p5mm': captured,
                'fit_support_sites': support_sites,
                'fit_eligible': fit is not None,
                'rigid_mind_loss': None if fit is None else float(fit.detach()),
                'fit_vs_weak_reference_state_gradient_dot': direction_dot,
                'fit_vs_weak_reference_state_gradient_cosine': direction_cosine,
                'fit_descent_predicted_to_reduce_weak_reference_cost':
                    None if direction_dot is None else direction_dot > 0,
                'synthetic_supervised_loss': float(synthetic_loss.detach()),
                'lupori_supervised_loss': float(real_loss.detach()),
                'supervised_pose_head_gradient_norm': float(supervised_pose_norm),
                'supervised_geometry_output_gradient_norm': float(supervised_geometry_norm),
                'fit_raw_geometry_output_gradient_norm': fit_raw_norm,
                'fit_probability_row_gradient_max': fit_probability_row_gradient_max,
                'fit_gradient_scale': fit_scale,
                'fit_applied_geometry_output_gradient_norm': fit_applied_norm,
                'total_gradient_norm_before_global_clip': float(total_gradient),
                'all_gradients_finite': True,
                'seconds_since_arm_start': time.perf_counter() - started}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if step == 1 or step % 50 == 0:
                log.flush()
                draws.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    'step': step, 'fit_eligible_so_far': fit_eligible_count,
                    'lupori_selected_five_point_mm': selected_error_mm}), flush=True)
            if step in milestones:
                checkpoint_path = directory / f'joint_step_{step:05d}.pt'
                temporary_path = checkpoint_path.with_suffix('.tmp')
                torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'site_rng': site_generator.get_state(), 'order': order, 'cursor': cursor,
                    'epoch': epoch, 'config': config,
                    'calibrated': False}, temporary_path)
                os.replace(temporary_path, checkpoint_path)
    (directory / 'completed.json').write_text(json.dumps({'arm': arm,
        'updates': updates, 'synthetic_presentations': 3 * updates,
        'synthetic_draw_attempts': draw_count, 'lupori_presentations': updates,
        'lupori_gradient_animals': train_animals,
        'fit_eligible_presentations': fit_eligible_count,
        'parent_checkpoint_sha256': sha(parent), 'config_sha256': sha(run / 'config.json'),
        'draws_sha256': sha(directory / 'draws.jsonl'),
        'training_sha256': sha(directory / 'training.jsonl'),
        'checkpoint_sha256': {str(step): sha(directory / f'joint_step_{step:05d}.pt')
            for step in milestones},
        'calibrated': False, 'public_benchmark_used': False,
        'development_calibration_final_read': False}, indent=2))
    del model, optimizer
print(json.dumps({'event': 'matched_local_fit_complete', 'updates_per_arm': updates}), flush=True)
