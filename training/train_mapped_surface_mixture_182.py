"""Bounded TRAIN-only direct mapped-surface mixture continuation of frozen 132."""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.joint_pose_map_121 import joint_fit_loss_121, joint_target_loss_121


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def section(base, draw_seed, appearance):
    with torch.no_grad():
        geometry = sample_panel_geometry_train_166(context, [base], draw_seed, 256)
        sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                   else sample_one_shot_slide_artifacts_v3)
        return sampler(context, [base], draw_seed, 256, source_sample=geometry)


def input_sha(sample):
    return hashlib.sha256(sample['inputs'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def blind_beam(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(len(beam), device=beam.device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(len(beam), 64, 2)[
            torch.arange(len(beam), device=beam.device), anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


def map_ids(model, prediction, sample, ids, atlas_volume):
    return model.map(prediction, sample['offsets'], ids // 2, ids % 2, (96, 96),
                     atlas_volume, sample['weights'], feature_side=48,
                     source_shape=(256, 256))


def evaluate(model, arm, step, null_atlas=None):
    model.eval()
    rows = []
    if null_atlas is None:
        null_atlas = atlas.clone()
        null_atlas[0].zero_()
    with (run / 'inner_rows.jsonl').open('a') as output, torch.inference_mode():
        for item in panel:
            sample = section(item['base_index'], item['draw_seed'], item['appearance'])
            assert sample['provenance'][0]['physical_section_id'] == item['physical_section_id']
            assert input_sha(sample) == item['model_inputs_sha256']
            valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
            pixel = valid[torch.linspace(0, len(valid) - 1, 128, device='cuda').long()][None]
            chart = torch.stack((pixel.remainder(256), pixel.div(256, rounding_mode='floor')), -1).float() / 256
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = model.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            top = prior.argmax(-1)[:, None]
            selected = map_ids(model, prediction, sample, top, atlas)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            _, beam = blind_beam(prediction)
            selected_mm = joint_target_loss_121(selected, sample['valid_mask'], sample['centre'],
                sample['state'], sample['reflection'], pixel)['map_loss'][0, 0]
            null = map_ids(model, prediction, sample, top, null_atlas)
            null_mm = joint_target_loss_121(null, sample['valid_mask'], sample['centre'],
                sample['state'], sample['reflection'], pixel)['map_loss'][0, 0]
            exact_prediction = {'state': sample['state'][:, None],
                'feature': prediction['feature'], 'global_feature': prediction['global_feature']}
            exact = model.map(exact_prediction, sample['offsets'],
                torch.zeros((1, 1), device='cuda', dtype=torch.long),
                sample['reflection'][:, None], (96, 96), atlas, sample['weights'],
                feature_side=48, source_shape=(256, 256))
            exact_mm = joint_target_loss_121(exact, sample['valid_mask'], sample['centre'],
                sample['state'], sample['reflection'], pixel)['map_loss'][0, 0]
            truth = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
            beam_state = prediction['state'][0, beam[0] // 2]
            rigid = rigid_points_090(beam_state, beam[0] % 2, chart[0])
            best16_mm = (rigid - truth).norm(dim=-1).mean(-1).min() / 1000
            top_state = prediction['state'][0, top[0] // 2]
            direct_rigid = rigid_points_090(top_state, top[0] % 2, chart[0])
            direct_rigid_mm = (direct_rigid - truth).norm(dim=-1).mean() / 1000
            coverage = joint_fit_loss_121(sample['inputs'], selected, atlas,
                sample['weights'], sample['valid_mask'], side=48)['atlas_coverage'][0, 0]
            record = {'arm': arm, 'step': step, 'appearance': item['appearance'],
                'base_index': item['base_index'], 'physical_section_id': item['physical_section_id'],
                'selected_mm': float(selected_mm), 'null_mm': float(null_mm),
                'exact_mm': float(exact_mm), 'direct_rigid_mm': float(direct_rigid_mm),
                'best16_rigid_mm': float(best16_mm),
                'atlas_coverage': float(coverage), 'direct_id': int(top[0, 0]),
                'direct_seconds': elapsed}
            output.write(json.dumps(record, allow_nan=False) + '\n')
            rows.append(record)
    del null_atlas
    torch.cuda.empty_cache()
    model.train()
    return rows


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/MAPPED_SURFACE_MIXTURE_182_TRAIN_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/mapped_surface_mixture_182_train_001'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
seed, updates, sites, temperature = 20261011182, 300, 128, .7
assert source.drive.upper() == protocol.drive.upper() == run.drive.upper() == 'I:'
assert torch.cuda.is_available() and not run.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert not any(parent_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['config'] == parent_config and parent['calibrated'] is False
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
assert len(context['bases']) == 64 and all(base['lineage']['split'] == 'train' for base in context['bases'])
atlas = context['atlas']
coronal = load_reserved_real_train()
assert coronal['bindings'] == parent_config['coronal_bindings']
sag_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'summary.json') == parent_config['sagittal_summary_sha256']
assert all(sha(sagittal_dir / name) == digest for name, digest in sag_summary['output_sha256'].items())
sag_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert all(row['split'] == 'train' for row in sag_records)
sag_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
sag_donors = {}
for index, record in enumerate(sag_records):
    sag_donors.setdefault(record['donor_id'], []).append(index)
sag_donor_ids = sorted(sag_donors)
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sag_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], 256 * affine[:, :, 0], 256 * affine[:, :, 1]), 1)
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(sag_states)[1][..., :, 2]
sag_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sag_reflection, 0] += 255 / 256 * ouv[sag_reflection, 1]
ouv[sag_reflection, 1] *= -1
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

source_files = ('train_mapped_surface_mixture_182.py', 'arbitrary_plane_one_shot_model.py',
    'joint_pose_map_121.py', 'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py', 'arbitrary_plane_recurrent_model.py',
    'arbitrary_plane_joint_model_v7.py', 'global_atlas_contrast_090.py',
    'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'checkpoints': [0, 150, 300],
    'parent_checkpoint_sha256': sha(parent_path), 'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol), 'source_sha256': {name: sha(source.parent / name) for name in source_files},
    'synthetic_provenance': context['provenance'], 'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sag_summary['output_sha256'],
    'train_bases': list(range(56)), 'inner_bases': list(range(56, 64)),
    'inner_sections_per_base_appearance': 8, 'train_appearance_cycle': ['v4', 'v4', 'v3'],
    'blind_beam': '181 8 base + 6 anchor + 2 normal-diverse anchor representations',
    'map_temperature_mm': temperature, 'map_anchor_weight': .2,
    'blind_warp_penalty_weight': .03, 'map_output_side': 96, 'atlas_feature_side': 48,
    'map_chunk': 2, 'trainable_mapper': ['warp_condition', 'atlas_encoder', 'pair', 'warp'],
    'frozen_mapper_stem': 'warp_shared', 'calibrated': False,
    'arms': ['full', 'detached', 'support_only'],
    'train_atlas': {'full': 'intact', 'detached': 'intact',
        'support_only': 'intensity zero, support unchanged'},
    'development_used': False, 'final_animals_used': False, 'public_benchmark_used': False}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
seen = set()
draws, panel = [], []
with (run / 'draws.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        appearance = ('v4', 'v4', 'v3')[(step - 1) % 3]
        while True:
            base = int(rng.integers(56))
            draw_seed = int(rng.integers(0, 2**63 - 1000))
            sample = section(base, draw_seed, appearance)
            provenance = sample['provenance'][0]
            physical_id = provenance['physical_section_id']
            assert physical_id not in seen and provenance['split'] == 'train'
            seen.add(physical_id)
            accepted = bool(sample['eligible'][0]) and bool(sample['valid_mask'][0].sum() >= sites)
            image_hash = input_sha(sample)
            log.write(json.dumps({'kind': 'gradient_attempt', 'step': step,
                'appearance': appearance, 'base_index': base, 'draw_seed': draw_seed,
                'used': accepted, 'valid_pixels': int(sample['valid_mask'][0].sum()),
                'model_inputs_sha256': image_hash, 'provenance': provenance}, allow_nan=False) + '\n')
            if accepted:
                break
        cor_donor = int(rng.integers(len(coronal['donors'])))
        cor_row = int(rng.integers(len(coronal['donors'][cor_donor]['identities'])))
        sag_donor = sag_donor_ids[int(rng.integers(len(sag_donor_ids)))]
        sag_row = int(rng.choice(sag_donors[sag_donor]))
        item = {'step': step, 'appearance': appearance, 'base_index': base,
            'draw_seed': draw_seed, 'physical_section_id': physical_id,
            'model_inputs_sha256': image_hash,
            'site_seed': int(rng.integers(0, 2**63 - 1000)),
            'coronal_donor': cor_donor, 'coronal_row': cor_row, 'sagittal_row': sag_row,
            'coronal_identity': coronal['donors'][cor_donor]['identities'][cor_row],
            'sagittal_identity': {key: sag_records[sag_row][key] for key in
                ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}}
        draws.append(item)
        log.write(json.dumps({'kind': 'paired_update', **item}, allow_nan=False) + '\n')
    for appearance in ('v3', 'v4'):
        for base in range(56, 64):
            for repeat in range(8):
                while True:
                    draw_seed = int(rng.integers(0, 2**63 - 1000))
                    sample = section(base, draw_seed, appearance)
                    provenance = sample['provenance'][0]
                    physical_id = provenance['physical_section_id']
                    assert physical_id not in seen and provenance['split'] == 'train'
                    seen.add(physical_id)
                    accepted = bool(sample['eligible'][0]) and bool(sample['valid_mask'][0].sum() >= sites)
                    image_hash = input_sha(sample)
                    log.write(json.dumps({'kind': 'inner_attempt', 'appearance': appearance,
                        'base_index': base, 'repeat': repeat, 'draw_seed': draw_seed,
                        'used': accepted, 'valid_pixels': int(sample['valid_mask'][0].sum()),
                        'model_inputs_sha256': image_hash,
                        'provenance': provenance}, allow_nan=False) + '\n')
                    if accepted:
                        break
                panel.append({'appearance': appearance, 'base_index': base,
                    'draw_seed': draw_seed, 'physical_section_id': physical_id,
                    'model_inputs_sha256': image_hash})
assert len(draws) == updates and len(panel) == 128
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / 256
flags = torch.arange(2, device='cuda')[None, None]
results = {}
with (run / 'training.jsonl').open('w') as log:
    for arm in ('full', 'detached', 'support_only'):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_atlas = atlas if arm != 'support_only' else atlas.clone()
        if arm == 'support_only':
            train_atlas[0].zero_()
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
            atlas_conditioning=True, fit_quality=True, vector_refinement=True,
            candidate_ranking=True, fitted_ranking=True).cuda().train()
        attach_global_plane_matcher(model, enabled=True)
        model.load_state_dict(parent['model'], strict=True)
        model.requires_grad_(False)
        pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
        shared_modules = (model.encoder, model.lateral)
        map_modules = (model.warp_condition, model.atlas_encoder, model.pair, model.warp)
        for module in (*pose_modules, *shared_modules, *map_modules):
            module.requires_grad_(True)
        groups = [
            {'params': [p for m in pose_modules for p in m.parameters()], 'base_lr': 2e-5},
            {'params': [p for m in shared_modules for p in m.parameters()], 'base_lr': 5e-6},
            {'params': [p for m in map_modules for p in m.parameters()], 'base_lr': 1e-5},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        parameters = [p for group in groups for p in group['params']]
        torch.save({'step': 0, 'arm': arm, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'config': config, 'calibrated': False},
            run / f'{arm}_step_00000.pt')
        if arm == 'full':
            results['parent'] = evaluate(model, 'parent', 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for item in draws:
            step = item['step']
            sample = section(item['base_index'], item['draw_seed'], item['appearance'])
            assert sample['provenance'][0]['physical_section_id'] == item['physical_section_id']
            assert input_sha(sample) == item['model_inputs_sha256']
            real_coronal = sample_reserved_real_train(coronal, item['coronal_donor'], [item['coronal_row']])
            coronal_inputs = F.interpolate(real_coronal['inputs'], (256, 256),
                mode='bilinear', align_corners=False)
            sagittal_inputs = torch.zeros(1, 5, 256, 256, device='cuda')
            sagittal_inputs[:, :1] = torch.from_numpy(np.asarray(
                sag_images[item['sagittal_row']:item['sagittal_row'] + 1]).copy()).to('cuda')
            inputs = torch.cat((sample['inputs'], coronal_inputs, sagittal_inputs))
            real_state = torch.cat((real_coronal['state'],
                sag_states[item['sagittal_row']:item['sagittal_row'] + 1].to('cuda')))
            real_reflection = torch.cat((real_coronal['reflection'],
                sag_reflection[item['sagittal_row']:item['sagittal_row'] + 1].long().to('cuda')))
            generator = torch.Generator(device='cuda').manual_seed(item['site_seed'])
            pixel = torch.multinomial(sample['valid_mask'].flatten(1).float(),
                sites, replacement=True, generator=generator)
            chart = torch.stack((pixel.remainder(256),
                pixel.div(256, rounding_mode='floor')), -1).float() / 256
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                pilot = model.predict(inputs)
                pilot_synthetic = {key: value[:1] for key, value in pilot.items()
                    if isinstance(value, torch.Tensor)}
                pilot_state = pilot_synthetic['state']
                pilot_feature = pilot_synthetic['feature']
                pilot_global_feature = pilot_synthetic['global_feature']
                pilot_prior, beam = blind_beam(pilot_synthetic)
                errors = []
                for start in range(0, 16, 2):
                    mapped = map_ids(model, pilot_synthetic, sample, beam[:, start:start + 2], train_atlas)
                    errors.append(joint_target_loss_121(mapped, sample['valid_mask'],
                        sample['centre'], sample['state'], sample['reflection'], pixel)['map_loss'])
                errors = torch.cat(errors, 1)
                chosen_prior = pilot_prior.gather(1, beam)
                responsibility = F.softmax(chosen_prior - errors / temperature, -1)
                mapped_nll = -temperature * torch.logsumexp(
                    chosen_prior - errors / temperature, -1).mean()
                order = responsibility.argsort(-1, descending=True)
                beam = beam.gather(1, order)
                responsibility = responsibility.gather(1, order)
                pilot_errors = errors.gather(1, order)
            del pilot, pilot_synthetic, mapped, errors, chosen_prior, order

            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(states), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            assert bool((prediction['state'][:1].detach() - pilot_state).abs().max() < 1e-2)
            assert bool((prior[:1].detach() - pilot_prior).abs().max() < 1e-4)
            assert bool((prediction['feature'][:1].detach() - pilot_feature).abs().max() < 1e-4)
            assert bool((prediction['global_feature'][:1].detach() - pilot_global_feature).abs().max() < 1e-4)
            reference = rigid_points_090(sample['state'], sample['reflection'], chart)
            proposed = rigid_points_090(states[:1], reflections[:1], chart[:, None, None])
            reference_five = rigid_points_090(sample['state'], sample['reflection'], corners)
            proposed_five = rigid_points_090(states[:1], reflections[:1], corners)
            tissue_mm = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
            five_mm = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
            predicted_normal = full_frame_state_to_components(prediction['state'][:1])[1][..., :, 2]
            true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
            normal_penalty = 4000 * (1 - (predicted_normal * true_normal[:, None]
                ).sum(-1).abs().clamp_max(1))
            physical = (.75 * tissue_mm + .25 * five_mm + normal_penalty[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            synthetic_loss = (-1.5 * torch.logsumexp(prior[:1] - physical / 1.5, -1)
                + .5 * F.kl_div(prior[:1], target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1)).mean()
            real_five = rigid_points_090(real_state, real_reflection, corners)
            real_proposed = rigid_points_090(states[1:], reflections[1:], corners)
            real_cost = (real_proposed - real_five[:, None, None]).norm(dim=-1
                ).mean(-1).flatten(1) / 1000
            real_mass = F.softmax(-real_cost.detach() / .7, -1)
            real_loss = (-1.5 * torch.logsumexp(prior[1:] - real_cost / 1.5, -1)
                + .5 * F.kl_div(prior[1:], real_mass, reduction='none').sum(-1)
                + .5 * real_cost.min(-1).values).mean()
            mapped_prediction = {'state': prediction['state'][:1],
                'feature': prediction['feature'][:1], 'global_feature': prediction['global_feature'][:1]}
            if arm == 'detached':
                mapped_prediction = {key: value.detach() for key, value in mapped_prediction.items()}
            decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * decay
            optimizer.zero_grad(set_to_none=True)
            warp_value = 0.
            probe = None
            for start in range(0, 16, 2):
                ids = beam[:, start:start + 2]
                mapped = map_ids(model, mapped_prediction, sample, ids, train_atlas)
                map_cost = joint_target_loss_121(mapped, sample['valid_mask'], sample['centre'],
                    sample['state'], sample['reflection'], pixel)['map_loss']
                assert bool((map_cost.detach() - pilot_errors[:, start:start + 2]).abs().max() < 1e-3)
                local = mapped['local_displacement_um'] / 1000
                warp = (local.square().mean()
                    + .3 * (local[..., 1:] - local[..., :-1]).square().mean()
                    + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean())
                chunk_loss = (responsibility[:, start:start + 2] * map_cost).sum() + .03 / 8 * warp
                if step == 1 and start == 0:
                    test = torch.autograd.grad(chunk_loss,
                        (model.pose[-1].weight, model.anchor_pose[-1].weight,
                         model.anchor_global[-1].weight, model.encoder[0][0].weight,
                         model.warp[-1].weight),
                        retain_graph=True, allow_unused=True)
                    probe = {'mapped_to_pose_grad': sum(float(value.norm()) for value in test[:3]
                        if value is not None),
                        'mapped_to_shared_grad': 0. if test[3] is None else float(test[3].norm()),
                        'mapped_to_mapper_grad': 0. if test[4] is None else float(test[4].norm())}
                    assert probe['mapped_to_mapper_grad'] > 0
                    assert (probe['mapped_to_pose_grad'] > 0) == (arm != 'detached')
                    assert (probe['mapped_to_shared_grad'] > 0) == (arm != 'detached')
                chunk_loss.backward(retain_graph=True)
                warp_value += float(warp.detach()) / 8
                del mapped, map_cost, local, warp, chunk_loss
            exact_prediction = {**mapped_prediction, 'state': sample['state'][:, None]}
            exact = model.map(exact_prediction, sample['offsets'],
                torch.zeros((1, 1), device='cuda', dtype=torch.long),
                sample['reflection'][:, None], (96, 96), train_atlas, sample['weights'],
                feature_side=48, source_shape=(256, 256))
            anchor = .2 * joint_target_loss_121(exact, sample['valid_mask'], sample['centre'],
                sample['state'], sample['reflection'], pixel)['map_loss'].mean()
            anchor.backward(retain_graph=True)
            mass_term = (-temperature * (responsibility * prior[:1].gather(1, beam)).sum()
                if arm != 'detached' else prior.new_zeros(()))
            (synthetic_loss + .5 * real_loss + mass_term).backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            record = {'arm': arm, 'update': step,
                'physical_section_id': item['physical_section_id'], 'appearance': item['appearance'],
                'synthetic_loss': float(synthetic_loss.detach()),
                'weak_real_loss': float(real_loss.detach()), 'mapped_nll': float(mapped_nll),
                'exact_anchor_mm': float(anchor.detach() / .2), 'warp_penalty': warp_value,
                'best160_rigid_mix_cost': float(physical.detach().min()),
                'direct_top1_rigid_mix_cost': float(physical.detach()[0, prior[0].detach().argmax()]),
                'gradient_norm': float(gradient), 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                'gradient_probe': probe}
            log.write(json.dumps(record, allow_nan=False) + '\n')
            if step == 1 or step % 50 == 0:
                log.flush()
                print(json.dumps({'event': 'train', **{key: record[key] for key in
                    ('arm', 'update', 'synthetic_loss', 'mapped_nll', 'peak_cuda_bytes', 'gradient_probe')}}),
                    flush=True)
            if step in (150, 300):
                torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config, 'calibrated': False},
                    run / f'{arm}_step_{step:05d}.pt')
            del sample, real_coronal, coronal_inputs, sagittal_inputs, inputs, real_state, real_reflection
            del generator, pixel, chart, pilot_state, pilot_feature, pilot_global_feature
            del pilot_prior, pilot_errors, responsibility, beam, mapped_nll
            del prediction, states, reflections, prior, reference, proposed, reference_five, proposed_five
            del tissue_mm, five_mm, predicted_normal, true_normal, normal_penalty, physical, target_mass
            del nearest, near_mode, near_cost, neighbourhood, synthetic_loss, real_five, real_proposed
            del real_cost, real_mass, real_loss, mapped_prediction, exact_prediction, exact, anchor, mass_term
            if step == 1:
                del test
        model.zero_grad(set_to_none=True)
        del optimizer, groups, parameters
        torch.cuda.empty_cache()
        results[arm] = evaluate(model, arm, updates,
            train_atlas if arm == 'support_only' else None)
        del model, pose_modules, shared_modules, map_modules, train_atlas
        torch.cuda.empty_cache()

summary = {}
for arm, rows in results.items():
    summary[arm] = {}
    for appearance in ('v3', 'v4'):
        part = [row for row in rows if row['appearance'] == appearance]
        summary[arm][appearance] = {
            'n': len(part),
            'selected_mm': float(np.mean([row['selected_mm'] for row in part])),
            'selected_le_1p5': float(np.mean([row['selected_mm'] <= 1.5 for row in part])),
            'direct_rigid_mm': float(np.mean([row['direct_rigid_mm'] for row in part])),
            'direct_rigid_le_1p5': float(np.mean([row['direct_rigid_mm'] <= 1.5 for row in part])),
            'best16_rigid_le_1p5': float(np.mean([row['best16_rigid_mm'] <= 1.5 for row in part])),
            'exact_mm': float(np.mean([row['exact_mm'] for row in part])),
            'atlas_coverage': float(np.mean([row['atlas_coverage'] for row in part])),
            'null_mm': float(np.mean([row['null_mm'] for row in part])),
            'null_le_1p5': float(np.mean([row['null_mm'] <= 1.5 for row in part])),
            'p95_direct_seconds': float(np.percentile([row['direct_seconds'] for row in part], 95)),
            'base_mean_mm': {str(base): float(np.mean([row['selected_mm'] for row in part
                if row['base_index'] == base])) for base in range(56, 64)}}
gate = {}
for appearance in ('v3', 'v4'):
    full = summary['full'][appearance]
    detached = summary['detached'][appearance]
    support_only = summary['support_only'][appearance]
    parent_metrics = summary['parent'][appearance]
    gate[appearance] = {
        'mean_gain_vs_detached_0p25': full['selected_mm'] <= detached['selected_mm'] - .25,
        'mean_gain_vs_parent_0p25': full['selected_mm'] <= parent_metrics['selected_mm'] - .25,
        'capture_gain_vs_detached_0p05': full['selected_le_1p5'] >= detached['selected_le_1p5'] + .05,
        'capture_noninferior_parent': full['selected_le_1p5'] >= parent_metrics['selected_le_1p5'],
        'direct_rigid_mean_noninferior_detached_parent_0p10': full['direct_rigid_mm'] <=
            min(detached['direct_rigid_mm'], parent_metrics['direct_rigid_mm']) + .10,
        'direct_rigid_capture_noninferior_detached_parent_0p05': full['direct_rigid_le_1p5'] >=
            max(detached['direct_rigid_le_1p5'], parent_metrics['direct_rigid_le_1p5']) - .05,
        'direct_rigid_mean_gain_vs_support_only_0p15': full['direct_rigid_mm'] <=
            support_only['direct_rigid_mm'] - .15,
        'direct_rigid_capture_gain_vs_support_only_0p05': full['direct_rigid_le_1p5'] >=
            support_only['direct_rigid_le_1p5'] + .05,
        'mapped_gain_vs_support_only_intact_0p15': full['selected_mm'] <=
            support_only['selected_mm'] - .15,
        'mapped_gain_vs_support_only_matched_0p15': full['selected_mm'] <=
            support_only['null_mm'] - .15,
        'six_of_eight_bases_nonworse': sum(full['base_mean_mm'][str(base)] <=
            detached['base_mean_mm'][str(base)] for base in range(56, 64)) >= 6,
        'best16_capture_retained': full['best16_rigid_le_1p5'] >= max(
            detached['best16_rigid_le_1p5'], parent_metrics['best16_rigid_le_1p5']) - .05,
        'exact_map_retained': full['exact_mm'] <= min(parent_metrics['exact_mm'] + .05, .25),
        'coverage_retained': full['atlas_coverage'] >= parent_metrics['atlas_coverage'] - .02,
        'atlas_intensity_advantage_0p05': full['selected_mm'] <= full['null_mm'] - .05,
        'p95_direct_latency_le_2s': full['p95_direct_seconds'] <= 2.}
summary['gate'] = gate
summary['passed'] = all(all(checks.values()) for checks in gate.values())
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
outputs = ('config.json', 'draws.jsonl', 'training.jsonl', 'inner_rows.jsonl', 'summary.json',
    'full_step_00000.pt', 'full_step_00150.pt', 'full_step_00300.pt',
    'detached_step_00000.pt', 'detached_step_00150.pt', 'detached_step_00300.pt',
    'support_only_step_00000.pt', 'support_only_step_00150.pt', 'support_only_step_00300.pt')
(run / 'completed.json').write_text(json.dumps({'source_sha256': config['source_sha256'],
    'protocol_sha256': config['protocol_sha256'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'output_sha256': {name: sha(run / name) for name in outputs},
    'paired_updates': updates, 'inner_physical_sections': len(panel),
    'passed': summary['passed'], 'calibrated': False,
    'development_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2) + '\n')
print(json.dumps({'event': 'completed', 'passed': summary['passed'], 'gate': gate}), flush=True)
