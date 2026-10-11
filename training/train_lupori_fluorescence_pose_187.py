"""Inner-animal fluorescence pose adaptation with full-sphere rehearsal."""

import hashlib
import json
import math
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


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
full_training = os.environ.get('ANATOMY_LUPORI_FULL_TRAIN') == '1'
protocol_name = ('LUPORI_FLUORESCENCE_POSE_ALL_TRAIN_188_PROTOCOL_20261011.md'
    if full_training else 'LUPORI_FLUORESCENCE_POSE_ADAPTATION_187_PROTOCOL_20261011.md')
protocol = source.parent / 'docs/publication' / protocol_name
split_path = source / 'lupori_animal_split_185.json'
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent = parent_dir / 'joint_step_02000.pt'
run = root / 'runs' / ('lupori_fluorescence_pose_188_all_train'
    if full_training else 'lupori_fluorescence_pose_187_inner_AL1A')
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
guard_animal = None if full_training else 'AL1A'
seed, side, updates, sites = (20261011188 if full_training else 20261011187), 256, 1200, 128
checkpoints = (0, 400, 800, 1200)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == root.drive.upper() == run.drive.upper() == 'I:'
assert not run.exists() and archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert guard_animal is None or guard_animal in split['train']
parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert not any(parent_done.get(key, False) for key in
    ('calibrated', 'public_benchmark_used', 'expert_real_truth_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
old_coronal = load_reserved_real_train()
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
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
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0],
                   side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
sagittal_normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = sagittal_normal.gather(
    1, sagittal_normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

lupori_records, lupori_images, lupori_masks, lupori_ouv = [], [], [], []
with ZipFile(archive) as source_zip:
    for animal in split['train']:
        if animal == guard_animal:
            continue
        info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        assert visualign['target-resolution'] == [456, 528, 320]
        for row in visualign['slices']:
            section_id = Path(row['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            thumbnail = f'{animal}/{row["filename"]}'
            mask_name = f'{animal}/masks/{section_id}-mask.png'
            image_bytes = source_zip.read(thumbnail)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            mask = Image.open(BytesIO(source_zip.read(mask_name))).convert('L')
            assert image.size == mask.size == (row['width'], row['height'])
            gray = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(gray[..., 0], gray[..., 1]) / 255
            tissue = np.asarray(mask.resize((side, side), Image.Resampling.NEAREST)) > 127
            scale = np.percentile(gray, 99)
            gray = np.clip(gray / max(scale, .01), 0, 1)
            source_ouv = torch.tensor(row['anchoring'], dtype=torch.float64).reshape(3, 3)
            o = quicknii_to_allen_points(source_ouv[0]) * 25
            u = quicknii_to_allen_vectors(source_ouv[1]) * 25
            v = quicknii_to_allen_vectors(source_ouv[2]) * 25
            lupori_images.append(gray.astype(np.float16))
            lupori_masks.append(tissue)
            lupori_ouv.append(torch.stack((o, u, v)))
            lupori_records.append({'animal_id': animal, 'source_section_id': section_id,
                'source_thumbnail': thumbnail, 'source_thumbnail_sha256':
                hashlib.sha256(image_bytes).hexdigest(), 'source_doi': split['source_doi'],
                'specimen_id': animal, 'experiment_id': None})
expected_sections = 240 if full_training else 188
assert len(lupori_records) == expected_sections and len({
    (r['animal_id'], r['source_section_id']) for r in lupori_records}) == expected_sections
lupori_images = torch.from_numpy(np.stack(lupori_images)).to('cuda', dtype=torch.float32)
lupori_masks = torch.from_numpy(np.stack(lupori_masks)).to('cuda')
lupori_ouv = torch.stack(lupori_ouv).to('cuda', dtype=torch.float32)
lupori_normals = F.normalize(torch.cross(lupori_ouv[:, 1], lupori_ouv[:, 2], dim=-1), dim=-1)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
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
parameters = [p for group in groups for p in group['params']]
source_files = ('train_lupori_fluorescence_pose_187.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'sites': sites,
    'training_phase': 'all_four_train_animals' if full_training else 'inner_animal_pilot',
    'lupori_archive_bytes': split['archive_bytes'],
    'lupori_archive_md5': split['archive_md5'],
    'synthetic_per_update': {'v4': 2, 'v3': 1},
    'lupori_per_update': 1, 'old_weak_per_update': 1,
    'inner_guard_animal': guard_animal, 'lupori_gradient_animals':
    [animal for animal in split['train'] if animal != guard_animal],
    'lupori_gradient_sections': len(lupori_records), 'checkpoints': checkpoints,
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol), 'split_sha256': sha(split_path),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'old_coronal_bindings': old_coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sagittal_summary['output_sha256'],
    'input': 'max(red,green)/255, image-wide p99 normalized, other four channels zero',
    'loss': '132 synthetic mixture + 1.0 Lupori tissue/five-point/normal mixture + 0.3 alternating old-real mixture',
    'optimized': 'shared image encoder/lateral and probabilistic direct pose heads only',
    'calibrated': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False, 'heldout_animal_used_in_training': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    path = run / f'joint_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
        'calibrated': False}, temp)
    os.replace(temp, path)


save(0)
flags = torch.arange(2, device='cuda')[None, None]
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
order = rng.permutation(len(lupori_records))
cursor = 0
epoch = 0
started = time.perf_counter()
draw_count = 0
accepted_ids = set()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        samples = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                eligible = bool(sample['eligible'][0])
                record = sample['provenance'][0]
                draw_count += 1
                draws.write(json.dumps({'kind': 'synthetic_train', 'update': step,
                    'slot': slot, 'appearance_version': appearance,
                    'draw_attempt': draw_count, 'used': eligible, **record},
                    allow_nan=False) + '\n')
                if eligible:
                    accepted_ids.add(record['physical_section_id'])
                    samples.append(sample)
                    break

        if cursor == len(order):
            order = rng.permutation(len(lupori_records))
            cursor = 0
            epoch += 1
        lupori_index = int(order[cursor])
        cursor += 1
        lupori_input = torch.zeros(1, 5, side, side, device='cuda')
        lupori_input[0, 0] = lupori_images[lupori_index]
        draws.write(json.dumps({'kind': 'lupori_train', 'update': step, 'slot': 3,
            'epoch': epoch, 'used': True, **lupori_records[lupori_index]},
            allow_nan=False) + '\n')

        old_kind = 'coronal' if step % 2 else 'sagittal'
        if old_kind == 'coronal':
            donor_index = int(rng.integers(len(old_coronal['donors'])))
            section_index = int(rng.integers(len(old_coronal['donors'][donor_index]['identities'])))
            old = sample_reserved_real_train(old_coronal, donor_index, [section_index])
            old_input = F.interpolate(old['inputs'], (side, side),
                mode='bilinear', align_corners=False)
            old_state, old_reflection = old['state'], old['reflection']
            old_identity = old['identities'][0]
        else:
            donor_id = sagittal_donor_ids[int(rng.integers(len(sagittal_donor_ids)))]
            rows = sagittal_donors[donor_id]
            index = rows[int(rng.integers(len(rows)))]
            old_input = torch.zeros(1, 5, side, side, device='cuda')
            old_input[:, :1] = torch.from_numpy(np.asarray(
                sagittal_images[index:index + 1]).copy()).to('cuda')
            old_state = sagittal_states[index:index + 1].to('cuda')
            old_reflection = sagittal_reflection[index:index + 1].long().to('cuda')
            old_identity = {key: sagittal_records[index][key] for key in
                ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}
        draws.write(json.dumps({'kind': f'old_{old_kind}_weak_train', 'update': step,
            'slot': 4, 'used': True, **old_identity}, allow_nan=False) + '\n')

        inputs = torch.cat([sample['inputs'] for sample in samples] + [lupori_input, old_input])
        target_state = torch.cat([sample['state'] for sample in samples])
        target_reflection = torch.cat([sample['reflection'] for sample in samples])
        valid = torch.cat([sample['valid_mask'] for sample in samples])
        prediction = model.predict(inputs)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags.expand(len(states), model.modes, 2)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

        pixel = torch.multinomial(valid.flatten(1).float(), sites, replacement=True)
        chart = torch.stack((pixel.remainder(side),
            pixel.div(side, rounding_mode='floor')), -1).float() / side
        reference = rigid_points_090(target_state, target_reflection, chart)
        proposed = rigid_points_090(states[:3], reflections[:3], chart[:, None, None])
        reference_five = rigid_points_090(target_state, target_reflection, corners)
        proposed_five = rigid_points_090(states[:3], reflections[:3], corners)
        tissue_mm = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
        five_mm = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
        predicted_normal = full_frame_state_to_components(prediction['state'][:3])[1][..., :, 2]
        true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
        normal_penalty = 4000 * (1 - (predicted_normal * true_normal[:, None]
            ).sum(-1).abs().clamp_max(1))
        physical = (.75 * tissue_mm + .25 * five_mm + normal_penalty[..., None]).flatten(1) / 1000
        target_mass = F.softmax(-physical.detach() / .7, -1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        near_mode = model.base_modes + nearest.indices
        near_cost = physical.reshape(3, model.modes, 2).min(-1).values.gather(1, near_mode)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        synthetic_loss = (-1.5 * torch.logsumexp(prior[:3] - physical / 1.5, -1)
            + .5 * F.kl_div(prior[:3], target_mass, reduction='none').sum(-1)
            + .5 * physical[:, :2 * model.base_modes].min(-1).values
            + (neighbourhood * near_cost).sum(-1)).mean()

        lpixel = torch.multinomial(lupori_masks[lupori_index].flatten().float(),
                                   sites, replacement=True)
        lchart = torch.stack((lpixel.remainder(side),
            lpixel.div(side, rounding_mode='floor')), -1).float() / side
        louv = lupori_ouv[lupori_index]
        lreference = louv[0] + lchart[:, :1] * louv[1] + lchart[:, 1:] * louv[2]
        lproposed = rigid_points_090(states[3:4], reflections[3:4], lchart)
        lreference_five = louv[0] + corners[:, :1] * louv[1] + corners[:, 1:] * louv[2]
        lproposed_five = rigid_points_090(states[3:4], reflections[3:4], corners)
        ltissue = (lproposed[0] - lreference).norm(dim=-1).mean(-1)
        lfive = (lproposed_five[0] - lreference_five).norm(dim=-1).mean(-1)
        lnormal = full_frame_state_to_components(prediction['state'][3:4])[1][..., :, 2][0]
        lnormal_penalty = 4000 * (1 - (lnormal * lupori_normals[lupori_index]).sum(-1
            ).abs().clamp_max(1))
        lcost = (.75 * ltissue + .25 * lfive + lnormal_penalty[:, None]).flatten() / 1000
        lmass = F.softmax(-lcost.detach() / .7, -1)
        lnearest = (lupori_normals[lupori_index] @ model.normal_anchor_frames[:, :, 2].T
            ).abs().topk(4)
        lnear_cost = lcost.reshape(model.modes, 2).min(-1).values[
            model.base_modes + lnearest.indices]
        lneighbourhood = F.softmax(40 * (lnearest.values - lnearest.values[:1]), -1)
        lupori_loss = (-1.5 * torch.logsumexp(prior[3] - lcost / 1.5, -1)
            + .5 * F.kl_div(prior[3], lmass, reduction='sum')
            + .5 * lcost[:2 * model.base_modes].min()
            + (lneighbourhood * lnear_cost).sum())

        old_reference = rigid_points_090(old_state, old_reflection, corners)
        old_proposed = rigid_points_090(states[4:5], reflections[4:5], corners)
        old_cost = (old_proposed - old_reference[:, None, None]).norm(dim=-1
            ).mean(-1).flatten(1) / 1000
        old_mass = F.softmax(-old_cost.detach() / .7, -1)
        old_loss = (-1.5 * torch.logsumexp(prior[4:] - old_cost / 1.5, -1)
            + .5 * F.kl_div(prior[4:], old_mass, reduction='none').sum(-1)
            + .5 * old_cost.min(-1).values).mean()
        loss = synthetic_loss + lupori_loss + .3 * old_loss
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'synthetic_presentations': 3 * step,
            'lupori_presentations': step, 'lupori_epoch': epoch,
            'lupori_animal_id': lupori_records[lupori_index]['animal_id'],
            'lupori_section_id': lupori_records[lupori_index]['source_section_id'],
            'synthetic_section_ids': [sample['provenance'][0]['physical_section_id']
                for sample in samples],
            'synthetic_selected_mm': float(physical.detach().gather(
                1, prior[:3].detach().argmax(-1)[:, None]).mean()),
            'lupori_selected_mm': float(lcost.detach()[prior[3].detach().argmax()]),
            'lupori_oracle_all160_mm': float(lcost.detach().min()),
            'old_weak_selected_mm': float(old_cost.detach()[0, prior[4].detach().argmax()]),
            'synthetic_loss': float(synthetic_loss.detach()),
            'lupori_loss': float(lupori_loss.detach()),
            'old_weak_loss': float(old_loss.detach()),
            'loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 400 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({'event': 'train_milestone', 'update': step,
                'lupori_epoch': epoch, 'lupori_selected_mm': row['lupori_selected_mm'],
                'synthetic_selected_mm': row['synthetic_selected_mm'],
                'loss': row['loss']}), flush=True)
        if step in checkpoints:
            save(step)

(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates,
    'synthetic_draw_attempts': draw_count,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'lupori_presentations': updates,
    'lupori_gradient_sections': len(lupori_records),
    'lupori_inner_guard_animal': guard_animal,
    'old_coronal_weak_presentations': (updates + 1) // 2,
    'old_sagittal_weak_presentations': updates // 2,
    'parent_checkpoint_sha256': sha(parent),
    'protocol_sha256': sha(protocol), 'split_sha256': sha(split_path),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False,
    'heldout_animal_used_in_training': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
