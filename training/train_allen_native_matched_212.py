"""Frozen matched 400-update TRAIN-native versus synthetic continuation."""

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
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.sample_allen_stpt_native_211 import (
    load_allen_stpt_native_211, sample_allen_stpt_native_211)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ALLEN_NATIVE_STPT_212_MATCHED_PROTOCOL_20261011.md'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
run = root / 'runs/allen_native_matched_212_001'
native_acquired = root / 'data/allen_stpt_native_208_train_001'
native_maps = root / 'data/allen_stpt_native_maps_210_train_001'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
seed, side, grid, sites, updates = 20261011212, 256, 96, 128, 400
milestones = (0, 200, 400)
gradient_donors = (14383, 14451, 14453, 14543, 14544, 14598)
inner_held = (14599, 14600)
train_animals = ('CC2B', 'CC4B', 'CC4A')
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert split['train'] == ['AL1A', *train_animals]
assert archive.stat().st_size == split['archive_bytes']
parent_done = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['800']
assert not any(parent_done.get(key, False) for key in
    ('calibrated', 'public_benchmark_used', 'external_pretrained_weights_used',
     'heldout_animal_used_in_training'))

# The 210 artifact is read only after its completion receipt exists.
assert (native_maps / 'completed.json').is_file()
maps_done = json.loads((native_maps / 'completed.json').read_text())
assert maps_done['donors'] == 8
assert set(maps_done['donor_ids']) == set(gradient_donors + inner_held)
assert maps_done['all_split'] == 'train'
assert not maps_done['development_calibration_final_used']
bindings = {}
for donor in gradient_donors:
    binding_path = native_maps / f'donor_{donor}/binding.json'
    assert sha(binding_path) == maps_done['binding_sha256'][f'donor_{donor}/binding.json']
    binding = json.loads(binding_path.read_text())
    assert binding['donor_id'] == donor and binding['split'] == 'train'
    assert binding['weak_alignment_not_anatomical_truth']
    assert sha(binding['map_path']) == binding['map_sha256']
    assert sha(binding['native_stack_path']) == binding['native_stack_sha256']
    assert sha(binding['native_geometry_path']) == binding['native_geometry_sha256']
    bindings[donor] = binding

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
old_coronal = load_reserved_real_train()
assert old_coronal['bindings'] == parent_config['coronal_bindings']
sagittal_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'summary.json') == parent_config['sagittal_summary_sha256']
assert all(sha(sagittal_dir / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
sagittal_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
sagittal_donors = {}
for index, row in enumerate(sagittal_records):
    sagittal_donors.setdefault(row['donor_id'], []).append(index)
sagittal_donor_ids = sorted(sagittal_donors)
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sagittal_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0], side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
sagittal_normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = sagittal_normal.gather(
    1, sagittal_normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

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
            rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
                (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            mask = np.asarray(Image.open(BytesIO(source_zip.read(
                f'{animal}/masks/{section_id}-mask.png'))).convert('L').resize(
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
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'source_doi': split['source_doi']})
assert len(lupori_records) == 188
lupori_images = torch.from_numpy(np.stack(lupori_images)).to('cuda', dtype=torch.float32)
lupori_masks = torch.from_numpy(np.stack(lupori_masks)).to('cuda')
lupori_ouv = torch.stack(lupori_ouv).to('cuda', dtype=torch.float32)
lupori_normals = F.normalize(torch.cross(lupori_ouv[:, 1], lupori_ouv[:, 2], dim=-1), dim=-1)

source_files = ('train_allen_native_matched_212.py', 'sample_allen_stpt_native_211.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
    'global_plane_matcher_120.py')
config = {'seed': seed, 'updates_per_arm': updates, 'checkpoints': milestones,
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'protocol_sha256': sha(protocol), 'split_sha256': sha(split_path),
    'source_sha256': {name: sha(source.with_name(name)) for name in source_files},
    'native_map_completion_sha256': sha(native_maps / 'completed.json'),
    'native_gradient_donors': gradient_donors, 'native_inner_held_donors': inner_held,
    'native_gradient_bindings_sha256': {str(donor): sha(
        native_maps / f'donor_{donor}/binding.json') for donor in gradient_donors},
    'synthetic_per_update_control': {'v4': 2, 'v3': 1},
    'synthetic_per_update_native': {'v4': 1, 'v3': 1, 'native_211': 1},
    'native_artifact_wrapper': 'v3 applied after each independent 211 source draw',
    'synthetic_provenance': context['provenance'],
    'lupori_gradient_animals': train_animals, 'lupori_sections': len(lupori_records),
    'lupori_inner_held_animal': 'AL1A',
    'old_coronal_bindings': old_coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'parent_optimizer': '187 AdamW pose/shared moments restored separately in both arms',
    'pose_lr': {'heads': 8e-6, 'encoder_lateral': 2e-6},
    'map_lr': {'warp_and_warp_shared': 3e-5,
               'warp_condition_pair_atlas_encoder': 1e-5},
    'map_trainable': ['warp_shared', 'warp', 'warp_condition', 'pair', 'atlas_encoder'],
    'pose_loss': '187 probabilistic mode/reflection tissue/five-point/normal composite',
    'native_information_weight': 'min(1, visible_tissue_fraction/0.1), applied to native pose and map only',
    'map_loss': 'teacher-forced correct state, valid-tissue mean Euclidean CCF error at 96 px; coefficient 0.5',
    'map_to_pose_gradient': False, 'MIND_to_pose_gradient': False,
    'map_feature_gradient': 'detached before warp_shared; mapper updates warp_shared but not encoder/lateral',
    'old_real_rehearsal_weight': .3, 'map_side': grid, 'sites': sites,
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_read': False,
    'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))

# Render independently seeded native virtual cuts donor by donor so only one
# acquired 3-D stack and authored map occupy GPU memory at a time.
native_samples = [None] * updates
native_plan = np.random.default_rng(seed + 211).permutation(
    np.resize(np.asarray(gradient_donors), updates))
native_attempts = 0
with (run / 'native_draws.jsonl').open('w') as native_log:
    for donor in gradient_donors:
        native_context = load_allen_stpt_native_211(
            native_acquired / f'donor_{donor}', bindings[donor]['map_path'],
            device='cuda', inner_role='train_gradient')
        assert native_context['geometry']['animal_id'] == donor
        for step in np.flatnonzero(native_plan == donor):
            attempt = 0
            while True:
                attempt += 1
                draw_seed = int(np.random.default_rng(np.random.SeedSequence(
                    [seed, int(step) + 1, donor, attempt])).integers(0, 2**63 - 1))
                source_sample = sample_allen_stpt_native_211(
                    native_context, [donor], draw_seed, side=side)
                sample = sample_one_shot_slide_artifacts_v3(
                    {}, [donor], draw_seed, side=side, source_sample=source_sample)
                native_attempts += 1
                accepted = bool(sample['eligible'][0])
                fraction = float(sample['valid_mask'][0].float().mean())
                source_fraction = float(source_sample['visible'][0].mean())
                native_log.write(json.dumps({'step': int(step) + 1, 'attempt': attempt,
                    'used': accepted, 'source_211_eligible': bool(source_sample['eligible'][0]),
                    'source_211_visible_tissue_fraction': source_fraction,
                    'wrapped_valid_tissue_fraction': fraction,
                    **sample['provenance'][0]}, allow_nan=False) + '\n')
                if accepted:
                    native_samples[int(step)] = {key: sample[key].detach().cpu()
                        for key in ('inputs', 'state', 'reflection', 'offsets',
                                    'weights', 'centre', 'valid_mask')}
                    native_samples[int(step)]['provenance'] = sample['provenance']
                    native_samples[int(step)]['visible_tissue_fraction'] = fraction
                    native_samples[int(step)]['source_211_visible_tissue_fraction'] = source_fraction
                    break
        del native_context, source_sample, sample
        torch.cuda.empty_cache()
assert all(sample is not None for sample in native_samples)

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
flags = torch.arange(2, device='cuda')[None, None]
choice = torch.zeros(3, 1, device='cuda', dtype=torch.long)
matched_common = []

for arm in ('control', 'native'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    extra_rng = np.random.default_rng(seed + 1)
    site_generators = [torch.Generator(device='cuda').manual_seed(seed + 1000 + i)
                       for i in range(4)]
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().train()
    attach_global_plane_matcher(model, enabled=True)
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    assert saved['step'] == 800 and saved['config'] == parent_config
    assert saved['calibrated'] is False
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    for module in (*pose_modules, *shared_modules, model.warp_shared, model.warp,
                   model.warp_condition, model.pair, model.atlas_encoder):
        module.requires_grad_(True)
    groups = [
        {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 2e-5},
        {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 5e-6},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer'])
    optimizer.add_param_group({'params': [p for module in (model.warp_shared, model.warp)
        for p in module.parameters()], 'lr': 3e-5})
    optimizer.add_param_group({'params': [p for module in (
        model.warp_condition, model.pair, model.atlas_encoder)
        for p in module.parameters()], 'lr': 1e-5})
    for group in optimizer.param_groups[:2]:
        group['lr'] = group['base_lr'] * .4
    del saved
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    order = rng.permutation(len(lupori_records))
    cursor, epoch, synthetic_attempts = 0, 0, 0
    checkpoint = directory / 'joint_step_00000.pt'
    temporary = checkpoint.with_suffix('.tmp')
    torch.save({'step': 0, 'arm': arm, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
        'extra_numpy_rng': extra_rng.bit_generator.state,
        'site_rng': [g.get_state() for g in site_generators],
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'order': order, 'cursor': cursor, 'epoch': epoch,
        'config': config, 'calibrated': False}, temporary)
    os.replace(temporary, checkpoint)
    started = time.perf_counter()
    with (directory / 'draws.jsonl').open('w') as draws, (
            directory / 'training.jsonl').open('w') as log:
        for step in range(1, updates + 1):
            samples, signature = [], []
            for slot, appearance in ((0, 'v4'), (1, 'v3'), (2, 'v4' if arm == 'control' else 'native')):
                if appearance == 'native':
                    sample = {key: (value.to('cuda') if torch.is_tensor(value) else value)
                              for key, value in native_samples[step - 1].items()}
                    draws.write(json.dumps({'kind': 'native_train', 'step': step,
                        'slot': slot, 'used': True,
                        'visible_tissue_fraction': sample['visible_tissue_fraction'],
                        'source_211_visible_tissue_fraction':
                            sample['source_211_visible_tissue_fraction'],
                        'effective_weight': min(1., sample['visible_tissue_fraction'] / .1),
                        **sample['provenance'][0]}, allow_nan=False) + '\n')
                else:
                    draw_rng = rng if slot < 2 else extra_rng
                    sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                               else sample_one_shot_slide_artifacts_v3)
                    while True:
                        virtual = int(draw_rng.integers(len(context['subjects'])))
                        draw_seed = int(draw_rng.integers(0, 2**63 - 1))
                        with torch.no_grad():
                            sample = sampler(context, [virtual], draw_seed, side=side)
                        synthetic_attempts += 1
                        accepted = bool(sample['eligible'][0])
                        draws.write(json.dumps({'kind': 'synthetic_train', 'step': step,
                            'slot': slot, 'appearance': appearance, 'virtual': virtual,
                            'draw_seed': draw_seed, 'draw_attempt': synthetic_attempts,
                            'used': accepted, 'valid_tissue_fraction':
                            float(sample['valid_mask'][0].float().mean()),
                            **sample['provenance'][0]}, allow_nan=False) + '\n')
                        if accepted:
                            break
                    if slot < 2:
                        signature.append((slot, appearance, virtual, draw_seed,
                                          sample['provenance'][0]['physical_section_id']))
                samples.append(sample)
            if cursor == len(order):
                order = rng.permutation(len(lupori_records))
                cursor, epoch = 0, epoch + 1
            index = int(order[cursor])
            cursor += 1
            signature.append(('lupori', index, epoch))
            lupori_input = torch.zeros(1, 5, side, side, device='cuda')
            lupori_input[0, 0] = lupori_images[index]
            draws.write(json.dumps({'kind': 'lupori_train', 'step': step,
                'slot': 3, 'epoch': epoch, 'used': True,
                **lupori_records[index]}, allow_nan=False) + '\n')
            old_kind = 'coronal' if step % 2 else 'sagittal'
            if old_kind == 'coronal':
                donor_index = int(rng.integers(len(old_coronal['donors'])))
                section_index = int(rng.integers(len(
                    old_coronal['donors'][donor_index]['identities'])))
                old = sample_reserved_real_train(old_coronal, donor_index, [section_index])
                old_input = F.interpolate(old['inputs'], (side, side),
                    mode='bilinear', align_corners=False)
                old_state, old_reflection = old['state'], old['reflection']
                old_identity = old['identities'][0]
            else:
                donor_id = sagittal_donor_ids[int(rng.integers(len(sagittal_donor_ids)))]
                rows = sagittal_donors[donor_id]
                old_index = rows[int(rng.integers(len(rows)))]
                old_input = torch.zeros(1, 5, side, side, device='cuda')
                old_input[:, :1] = torch.from_numpy(np.asarray(
                    sagittal_images[old_index:old_index + 1]).copy()).to('cuda')
                old_state = sagittal_states[old_index:old_index + 1].to('cuda')
                old_reflection = sagittal_reflection[
                    old_index:old_index + 1].long().to('cuda')
                old_identity = {key: sagittal_records[old_index][key] for key in
                    ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}
            signature.append((old_kind, old_identity))
            if arm == 'control':
                matched_common.append(signature)
            else:
                assert signature == matched_common[step - 1]
            draws.write(json.dumps({'kind': f'old_{old_kind}_weak_train',
                'step': step, 'slot': 4, 'used': True,
                **old_identity}, allow_nan=False) + '\n')

            inputs = torch.cat([s['inputs'] for s in samples] + [lupori_input, old_input])
            target_state = torch.cat([s['state'] for s in samples])
            target_reflection = torch.cat([s['reflection'] for s in samples])
            valid = torch.cat([s['valid_mask'].float() for s in samples])
            pre_warp = []
            hook = model.warp_shared.register_forward_pre_hook(
                lambda module, args: pre_warp.append(args[0].detach()))
            prediction = model.predict(inputs)
            hook.remove()
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(states), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            pixel = torch.stack([torch.multinomial(
                valid[i].flatten(), sites, replacement=True, generator=site_generators[i])
                for i in range(3)])
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            reference = rigid_points_090(target_state, target_reflection, chart)
            proposed = rigid_points_090(states[:3], reflections[:3], chart[:, None, None])
            reference_five = rigid_points_090(target_state, target_reflection, five)
            proposed_five = rigid_points_090(states[:3], reflections[:3], five)
            tissue_um = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
            predicted_normal = full_frame_state_to_components(
                prediction['state'][:3])[1][..., :, 2]
            true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
            normal_penalty = 4000 * (1 - (predicted_normal * true_normal[:, None]
                ).sum(-1).abs().clamp_max(1))
            physical = (.75 * tissue_um + .25 * five_um +
                        normal_penalty[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(3, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            pose_each = (-1.5 * torch.logsumexp(prior[:3] - physical / 1.5, -1)
                + .5 * F.kl_div(prior[:3], target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1))
            native_weight = (min(1., samples[2]['visible_tissue_fraction'] / .1)
                             if arm == 'native' else 1.)
            pose_loss = (pose_each[0] + pose_each[1] + native_weight * pose_each[2]) / 3

            lpixel = torch.multinomial(lupori_masks[index].flatten().float(),
                sites, replacement=True, generator=site_generators[3])
            lchart = torch.stack((lpixel.remainder(side),
                lpixel.div(side, rounding_mode='floor')), -1).float() / side
            louv = lupori_ouv[index]
            lreference = louv[0] + lchart[:, :1] * louv[1] + lchart[:, 1:] * louv[2]
            lproposed = rigid_points_090(states[3:4], reflections[3:4], lchart)
            lreference_five = louv[0] + five[:, :1] * louv[1] + five[:, 1:] * louv[2]
            lproposed_five = rigid_points_090(states[3:4], reflections[3:4], five)
            ltissue = (lproposed[0] - lreference).norm(dim=-1).mean(-1)
            lfive = (lproposed_five[0] - lreference_five).norm(dim=-1).mean(-1)
            lnormal = full_frame_state_to_components(prediction['state'][3:4])[1][..., :, 2][0]
            lnormal_penalty = 4000 * (1 - (lnormal * lupori_normals[index]
                ).sum(-1).abs().clamp_max(1))
            lcost = (.75 * ltissue + .25 * lfive +
                     lnormal_penalty[:, None]).flatten() / 1000
            lmass = F.softmax(-lcost.detach() / .7, -1)
            lnearest = (lupori_normals[index] @ model.normal_anchor_frames[:, :, 2].T
                ).abs().topk(4)
            lnear_cost = lcost.reshape(model.modes, 2).min(-1).values[
                model.base_modes + lnearest.indices]
            lneighbourhood = F.softmax(40 * (lnearest.values - lnearest.values[:1]), -1)
            lupori_loss = (-1.5 * torch.logsumexp(prior[3] - lcost / 1.5, -1)
                + .5 * F.kl_div(prior[3], lmass, reduction='sum')
                + .5 * lcost[:2 * model.base_modes].min()
                + (lneighbourhood * lnear_cost).sum())
            old_reference = rigid_points_090(old_state, old_reflection, five)
            old_proposed = rigid_points_090(states[4:5], reflections[4:5], five)
            old_cost = (old_proposed - old_reference[:, None, None]).norm(dim=-1
                ).mean(-1).flatten(1) / 1000
            old_mass = F.softmax(-old_cost.detach() / .7, -1)
            old_loss = (-1.5 * torch.logsumexp(prior[4:] - old_cost / 1.5, -1)
                + .5 * F.kl_div(prior[4:], old_mass, reduction='none').sum(-1)
                + .5 * old_cost.min(-1).values).mean()

            # The true affine gauge and detached pre-warp feature are supplied
            # only to the mapper; its loss updates warp_shared, not pose.
            supplied = {'state': target_state[:, None],
                'feature': model.warp_shared(pre_warp[0][:3]),
                'global_feature': prediction['global_feature'][:3].detach()}
            mapped = model.map(supplied,
                torch.cat([s['offsets'] for s in samples]),
                choice, target_reflection[:, None], (grid, grid),
                atlas, torch.cat([s['weights'] for s in samples]),
                feature_side=grid, source_shape=(side, side))
            valid_grid = F.interpolate(valid[:, None], (grid, grid), mode='area')[:, 0]
            centre = torch.cat([s['centre'] for s in samples]).permute(0, 3, 1, 2)
            target_grid = F.interpolate(centre * valid[:, None], (grid, grid),
                mode='area') / valid_grid[:, None].clamp_min(1e-6)
            map_error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] -
                         target_grid.permute(0, 2, 3, 1)).norm(dim=-1) / 1000
            map_each = (map_error * valid_grid).sum((1, 2)) / valid_grid.sum((1, 2))
            map_loss = (map_each[0] + map_each[1] + native_weight * map_each[2]) / 3
            loss = pose_loss + lupori_loss + .3 * old_loss + .5 * map_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            row = {'arm': arm, 'step': step, 'lupori_epoch': epoch,
                'lupori_animal_id': lupori_records[index]['animal_id'],
                'lupori_section_id': lupori_records[index]['section_id'],
                'slot_sources': ['v4', 'v3', 'v4' if arm == 'control' else 'native_211'],
                'slot_tissue_fraction': [float(s['valid_mask'][0].float().mean())
                    for s in samples],
                'third_slot_information_weight': native_weight,
                'slot_pose_loss': pose_each.detach().tolist(),
                'slot_map_loss_mm': map_each.detach().tolist(),
                'slot_selected_pose_cost_mm': physical.detach().gather(
                    1, prior[:3].detach().argmax(-1)[:, None])[:, 0].tolist(),
                'lupori_selected_pose_cost_mm': float(
                    lcost.detach()[prior[3].detach().argmax()]),
                'old_real_selected_five_point_mm': float(
                    old_cost.detach()[0, prior[4].detach().argmax()]),
                'pose_loss': float(pose_loss.detach()),
                'map_loss_mm': float(map_loss.detach()),
                'lupori_loss': float(lupori_loss.detach()),
                'old_real_loss': float(old_loss.detach()),
                'loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'seconds_since_arm_start': time.perf_counter() - started}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if step == 1 or step % 50 == 0:
                log.flush()
                draws.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    'step': step, 'native_weight': native_weight,
                    'slot_pose_loss': row['slot_pose_loss'],
                    'slot_map_loss_mm': row['slot_map_loss_mm']}), flush=True)
            if step in milestones:
                checkpoint = directory / f'joint_step_{step:05d}.pt'
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'numpy_rng': rng.bit_generator.state,
                    'extra_numpy_rng': extra_rng.bit_generator.state,
                    'site_rng': [g.get_state() for g in site_generators],
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all(),
                    'order': order, 'cursor': cursor, 'epoch': epoch,
                    'config': config, 'calibrated': False}, temporary)
                os.replace(temporary, checkpoint)
    (directory / 'completed.json').write_text(json.dumps({'arm': arm,
        'updates': updates, 'synthetic_presentations': (3 if arm == 'control' else 2) * updates,
        'native_presentations': 0 if arm == 'control' else updates,
        'synthetic_draw_attempts': synthetic_attempts,
        'lupori_presentations': updates, 'old_coronal_presentations': updates // 2,
        'old_sagittal_presentations': updates // 2,
        'parent_checkpoint_sha256': sha(parent),
        'config_sha256': sha(run / 'config.json'),
        'draws_sha256': sha(directory / 'draws.jsonl'),
        'training_sha256': sha(directory / 'training.jsonl'),
        'checkpoint_sha256': {str(step): sha(
            directory / f'joint_step_{step:05d}.pt') for step in milestones},
        'calibrated': False, 'public_benchmark_used': False,
        'development_calibration_final_read': False,
        'inner_held_donors_used_in_training': False}, indent=2))
    del model, optimizer
    torch.cuda.empty_cache()

(run / 'completed.json').write_text(json.dumps({'updates_per_arm': updates,
    'native_gradient_donors': gradient_donors, 'native_inner_held_donors': inner_held,
    'native_draw_attempts': native_attempts,
    'native_draws_sha256': sha(run / 'native_draws.jsonl'),
    'control_completion_sha256': sha(run / 'control/completed.json'),
    'native_completion_sha256': sha(run / 'native/completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_read': False}, indent=2))
print(json.dumps({'event': 'matched_native_pilot_complete',
    'updates_per_arm': updates}), flush=True)
