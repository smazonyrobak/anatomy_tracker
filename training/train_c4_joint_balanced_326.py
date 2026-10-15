"""1,000-batch from-scratch C4 acquired-balance transfer pilot; TRAIN data only."""
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

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261015326, 192, 1000
STAGES = (0, UPDATES)
LUPORI_TRAIN = ('CC2B', 'CC4B', 'CC4A')
RUN = ROOT / 'runs/c4_joint_balanced_326_001'
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
LUPORI_SPLIT = Path(__file__).with_name('lupori_animal_split_185.json')
CORONAL_SPLIT = ROOT / 'data/coronal_train_split_308_001/split.json'
CORONAL_TRAIN = ROOT / 'data/joint_v7_reserved_train_images_192_001'
CORONAL_INNER = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
SAGITTAL = ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
REFERENCE = ROOT / 'runs/c4_joint_321_002'
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/C4_BALANCED_TRANSFER_326_PROTOCOL_20261015.md'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
assert json.loads(LUPORI_SPLIT.read_text(encoding='utf8'))['train'] == ['AL1A', *LUPORI_TRAIN]
split = json.loads(CORONAL_SPLIT.read_text(encoding='utf8'))
coronal_gradient_donors = split['gradient_donors']
assert len(coronal_gradient_donors) == 1727
coronal_rows = [json.loads(line) for line in (CORONAL_INNER / 'records.jsonl').open(encoding='utf8')
                if '"training_split": "train"' in line]
assert len(coronal_rows) == 1280 and all(row['training_split'] == 'train' for row in coronal_rows)
coronal_by_donor = {}
for row in coronal_rows:
    coronal_by_donor.setdefault(row['animal_partition_key'], []).append(row)
assert len(coronal_by_donor) == 58
assert set(coronal_by_donor) == set(split['prior_pool_excluded_from_308_gradients'])
assert not set(coronal_by_donor) & set(coronal_gradient_donors)
sagittal_rows = [json.loads(line) for line in (SAGITTAL / 'geometry.jsonl').open(encoding='utf8')]
assert len(sagittal_rows) == 653 and all(row['split'] == 'train' for row in sagittal_rows)
sagittal_by_donor = {}
for row in sagittal_rows:
    sagittal_by_donor.setdefault(row['donor_id'], []).append(row)
assert len(sagittal_by_donor) == 40

# Freeze donor keys and lower-median sections before model initialization or training.
coronal_inner_donors = sorted(coronal_by_donor, key=lambda key:
    hashlib.sha256(f'326-coronal-inner:{key}'.encode()).hexdigest())[:10]
sagittal_inner_donors = sorted(sagittal_by_donor, key=lambda key:
    hashlib.sha256(f'326-sagittal-inner:{key}'.encode()).hexdigest())[:8]
sagittal_gradient_donors = sorted(set(sagittal_by_donor) - set(sagittal_inner_donors))
assert len(sagittal_gradient_donors) == 32
assert not set(sagittal_inner_donors) & set(sagittal_gradient_donors)
real_inner = []
for domain, donors, groups in (('coronal', coronal_inner_donors, coronal_by_donor),
                               ('sagittal', sagittal_inner_donors, sagittal_by_donor)):
    for donor in donors:
        group = sorted(groups[donor], key=lambda row: (row['section_number'], row['section_id']))
        real_inner.append((domain, group[(len(group) - 1) // 2]))

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_fov_balance_v10('cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 64 * VARIANTS
reference_completion_sha = sha(REFERENCE / 'completed.json')
assert reference_completion_sha == 'cdc5261a2c07e47da721048b10add7388fc48bbe41c29496189a7481b38b89ec'
reference_receipt = json.loads((REFERENCE / 'completed.json').read_text(encoding='utf8'))
reference_panels_sha = sha(REFERENCE / 'panels.pt')
reference_readouts_sha = sha(REFERENCE / 'readouts.jsonl')
assert reference_panels_sha == reference_receipt['output_sha256']['panels.pt']
assert reference_readouts_sha == reference_receipt['output_sha256']['readouts.jsonl']
panels = torch.load(REFERENCE / 'panels.pt', map_location='cpu', weights_only=True)
reference = {row['batch']: row['metrics'] for row in map(json.loads,
    (REFERENCE / 'readouts.jsonl').read_text(encoding='utf8').splitlines())}
assert set(panels) == {'inner', 'steep', 'sparse'} and 1000 in reference
assert [len(panels[key]) for key in ('inner', 'steep', 'sparse')] == [128, 32, 32]
assert all(case['meta']['base'] >= 56 for cases in panels.values() for case in cases)

lupori_records, lupori_images, lupori_ouvs = [], [], []
with ZipFile(ARCHIVE) as archive:
    for animal in LUPORI_TRAIN:
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        for item in registration['slices']:
            section = Path(item['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            member = f'{animal}/{item["filename"]}'
            raw = archive.read(member)
            rgb = np.asarray(Image.open(BytesIO(raw)).convert('RGB').resize(
                (SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            anchoring = torch.as_tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            lupori_records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'source_section_id': section,
                'archive_member': member, 'source_image_sha256': hashlib.sha256(raw).hexdigest()})
            lupori_images.append(gray)
            lupori_ouvs.append(torch.stack((quicknii_to_allen_points(anchoring[0]) * 25,
                quicknii_to_allen_vectors(anchoring[1]) * 25,
                quicknii_to_allen_vectors(anchoring[2]) * 25)))
assert len(lupori_records) == 188
lupori_images = torch.from_numpy(np.stack(lupori_images)).cuda()
lupori_ouvs = torch.stack(lupori_ouvs).cuda()
lupori_by_animal = {animal: [i for i, row in enumerate(lupori_records)
    if row['animal_id'] == animal] for animal in LUPORI_TRAIN}
coronal_inner_images = np.load(CORONAL_INNER / 'images.npy', mmap_mode='r')
sagittal_images = np.load(SAGITTAL / 'model_input.npy', mmap_mode='r')
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE


def weak_ouv(row, domain):
    affine = np.asarray(row['model_pixel_to_ccf_ref9_ap_dv_ml_um' if domain == 'sagittal'
                            else 'model_pixel_to_ap_dv_ml_um'], dtype=np.float32)
    offset, scale = (1 / 6, 256) if domain == 'sagittal' else (0., SIDE)
    origin = affine[:, 2] + offset * (affine[:, 0] + affine[:, 1])
    return torch.as_tensor(np.stack((origin, scale * affine[:, 0],
                                      scale * affine[:, 1])), device='cuda')


def acquired_input(raw, domain):
    if domain == 'lupori':
        image = raw[None, None]
    else:
        image = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None, None].cuda()
        if domain == 'sagittal':
            image = F.interpolate(image, (SIDE, SIDE), mode='area')
    return torch.cat((image, torch.zeros(1, 4, SIDE, SIDE, device='cuda')), 1)


def rigid_candidates(model, prediction, chart):
    batch, modes = prediction['state'].shape[:2]
    branch = torch.arange(modes, device=chart.device)[None, :, None].expand(batch, -1, 2)
    reflection = torch.arange(2, device=chart.device)[None, None].expand(batch, modes, -1)
    local_chart = model.canonical_chart(
        chart[:, None, None].expand(-1, modes, 2, -1, -1),
        branch // model.normal_modes, SIDE)
    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
    return rigid_points_090(states, reflection, local_chart, (SIDE, SIDE))


def branch_scores(prediction):
    reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                              F.logsigmoid(prediction['reflection_logit'])), -1)
    return (prediction['log_mass'][..., None] + reflection).flatten(1)


def evaluate(model):
    rows = []
    with torch.inference_mode():
        for cohort, cases in panels.items():
            for start in range(0, len(cases), 4):
                group = cases[start:start + 4]
                image = torch.stack([case['input'] for case in group]).cuda()
                chart = torch.stack([case['chart'] for case in group]).cuda()
                truth = torch.stack([case['truth'] for case in group]).cuda()
                prediction = model.predict(image)
                scores = branch_scores(prediction)
                first = scores.argmax(-1)
                errors = (rigid_candidates(model, prediction, chart) -
                          truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                mapped = model.map_sites(prediction, chart,
                    (first // 2)[:, None], (first % 2)[:, None])
                map_gap = (mapped['coordinates'][:, 0] - truth).norm(dim=-1).mean(-1) / 1000
                oracle = errors.gather(1, scores.topk(16, -1).indices).min(-1).values
                for i, case in enumerate(group):
                    rows.append({'cohort': cohort, **case['meta'],
                        'first_rigid_gap_mm': float(errors[i, first[i]]),
                        'first_mapped_gap_mm': float(map_gap[i]),
                        'top16_rigid_oracle_mm': float(oracle[i])})
        for domain, row in real_inner:
            if domain == 'coronal':
                raw = coronal_inner_images[row['array_row_index'], 0]
            else:
                raw = sagittal_images[row['array_row_index'], 0]
            prediction = model.predict(acquired_input(raw, domain))
            scores = branch_scores(prediction)[0]
            first = int(scores.argmax())
            normals = full_frame_state_to_components(prediction['state'][0])[1][:, :, 2]
            reference = weak_ouv(row, domain)
            weak_normal = F.normalize(torch.cross(reference[1], reference[2], dim=-1), dim=-1)
            weak_axis = int(weak_normal.abs().argmax())
            first_normal = normals[first // 2]
            rows.append({'cohort': domain, 'donor_id': row.get('donor_id', row.get('animal_partition_key')),
                'animal_id': row.get('animal_id', row.get('donor_id')),
                'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                'section_id': row['section_id'], 'array_row_index': row['array_row_index'],
                'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256')),
                'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                'first_axis_correct': int(first_normal.abs().argmax()) == weak_axis,
                'top16_contains_weak_axis': any(
                    int(normals[int(branch) // 2].abs().argmax()) == weak_axis
                    for branch in scores.topk(16).indices),
                'first_weak_normal_angle_deg': float(torch.rad2deg(torch.acos(
                    (first_normal * weak_normal).sum().abs().clamp(0, 1))))})
    metrics = {}
    for cohort in panels:
        group = [row for row in rows if row['cohort'] == cohort]
        metrics[cohort] = {'n': len(group),
            'mean_first_rigid_gap_mm': float(np.mean([r['first_rigid_gap_mm'] for r in group])),
            'mean_first_mapped_gap_mm': float(np.mean([r['first_mapped_gap_mm'] for r in group])),
            'first_mapped_over_5mm': sum(r['first_mapped_gap_mm'] > 5 for r in group),
            'top16_rigid_capture_1p5mm': sum(r['top16_rigid_oracle_mm'] <= 1.5 for r in group)}
    for domain in ('coronal', 'sagittal'):
        group = [row for row in rows if row['cohort'] == domain]
        metrics[domain] = {'n': len(group),
            'first_axis_correct': sum(r['first_axis_correct'] for r in group),
            'top16_contains_weak_axis': sum(r['top16_contains_weak_axis'] for r in group),
            'median_weak_normal_angle_deg': float(np.median([
                r['first_weak_normal_angle_deg'] for r in group]))}
    assert [metrics[key]['n'] for key in ('inner', 'steep', 'sparse', 'coronal', 'sagittal')] == [128, 32, 32, 10, 8]
    return metrics, rows


config = {'candidate': 326, 'purpose': 'acquired-domain balance transfer feasibility, not promotion',
    'seed': SEED, 'optimizer_batches': UPDATES, 'readout_batches': STAGES,
    'model_initialization': 'random C4; no prior weights, features, or pseudo-labels',
    'model_comparison_limit': 'C4 ~6.23M parameters versus 309 ~22M; not an architecture comparison',
    'synthetic_per_batch': 4, 'synthetic_gradient_bases': list(range(56)),
    'synthetic_readout_bases': list(range(56, 64)),
    'synthetic_sampler': 'one independent random v10 physical cut per slot; no pairing or deliberate duplicate',
    'pose_and_map_losses': 'unchanged 321 synthetic pose, true-state/near/first map, support, and warp terms',
    'acquired_schedule': 'one weak real image per batch, cyclic equal Lupori/coronal/sagittal presentations',
    'weak_real_loss': '321 five-site OUV gap with 1mm tolerance and unsigned normal with 100um tolerance, weight 0.3',
    'coronal_gradient_donors': len(coronal_gradient_donors),
    'coronal_inner_donors': coronal_inner_donors,
    'sagittal_gradient_donors': sagittal_gradient_donors,
    'sagittal_inner_donors': sagittal_inner_donors,
    'lupori_gradient_animals': LUPORI_TRAIN,
    'real_inner_selection': 'SHA256-ranked donor keys, lower-median section; frozen before training',
    'real_inner_role': 'TRAIN-only weak-affine family transfer readout, not physical pose truth',
    'gate': 'at batch 1000 only: 321 batch-1000 synthetic nonregression, coronal retention, sagittal family gain',
    'model_source_sha256': sha(SOURCE.with_name('arbitrary_plane_c4_joint_model.py')),
    'runner_source_sha256': sha(SOURCE),
    'protocol_sha256': sha(PROTOCOL),
    'v10_source_sha256': sha(SOURCE.with_name('arbitrary_plane_fov_balance_v10.py')),
    'sampler_input_sha256': context['provenance']['input_sha256'],
    'reference_completion_sha256': reference_completion_sha,
    'reference_panels_sha256': reference_panels_sha,
    'reference_readouts_sha256': reference_readouts_sha,
    'lupori_split_sha256': sha(LUPORI_SPLIT), 'lupori_archive_sha256': sha(ARCHIVE),
    'coronal_split_sha256': sha(CORONAL_SPLIT),
    'coronal_inner_records_sha256': sha(CORONAL_INNER / 'records.jsonl'),
    'coronal_inner_images_sha256': sha(CORONAL_INNER / 'images.npy'),
    'sagittal_records_sha256': sha(SAGITTAL / 'geometry.jsonl'),
    'sagittal_images_sha256': sha(SAGITTAL / 'model_input.npy'),
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'real_inner.jsonl').write_text(''.join(json.dumps({'domain': domain,
    'donor_id': row.get('donor_id', row.get('animal_partition_key')),
    'animal_id': row.get('animal_id', row.get('donor_id')),
    'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
    'section_id': row['section_id'], 'section_number': row['section_number'],
    'array_row_index': row['array_row_index'],
    'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256'))}) + '\n'
    for domain, row in real_inner), encoding='utf8')

torch.manual_seed(SEED + 3)
torch.cuda.manual_seed_all(SEED + 3)
model = C4JointSliceModel(side=SIDE).cuda()
config['model_parameters'] = sum(parameter.numel() for parameter in model.parameters())
(RUN / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
train_rng = np.random.default_rng(SEED + 4)
site_rng = torch.Generator(device='cuda').manual_seed(SEED + 5)
donor_pools = {'lupori': LUPORI_TRAIN, 'coronal': coronal_gradient_donors,
               'sagittal': sagittal_gradient_donors}
donor_orders = {domain: train_rng.permutation(pool).tolist() for domain, pool in donor_pools.items()}
donor_cursors = {domain: 0 for domain in donor_pools}
donor_cache = {}
started, readouts, stage_gates = time.perf_counter(), {}, {}

with (RUN / 'train_draws.jsonl').open('w', encoding='utf8') as draw_file, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training_file, \
        (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
    for step in range(UPDATES + 1):
        if step:
            samples = []
            with torch.no_grad():
                for slot in range(4):
                    while True:
                        base = int(train_rng.integers(56))
                        variant = int(train_rng.integers(VARIANTS))
                        seed = int(train_rng.integers(2**63 - 1, dtype=np.int64))
                        sample = sample_fov_balance_v10(context,
                            [base * VARIANTS + variant], seed, SIDE)
                        provenance = sample['provenance'][0]
                        assert provenance['split'] == 'train'
                        accepted = bool(sample['eligible'][0])
                        draw_file.write(json.dumps({'step': step, 'slot': slot, 'kind': 'synthetic',
                            'base': base, 'variant': variant, 'seed': seed, 'accepted': accepted,
                            'provenance': provenance}, allow_nan=False) + '\n')
                        if accepted:
                            samples.append(sample)
                            break
            image = torch.cat([sample['inputs'] for sample in samples])
            target_state = torch.cat([sample['state'] for sample in samples])
            target_reflection = torch.cat([sample['reflection'] for sample in samples])
            valid = torch.cat([sample['valid_mask'] for sample in samples])
            centre = torch.cat([sample['centre'] for sample in samples])
            pixel = torch.multinomial(valid.flatten(1).float(), 64,
                                      replacement=True, generator=site_rng)
            chart = torch.stack((pixel.remainder(SIDE),
                pixel.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = centre.flatten(1, 2).gather(1, pixel[..., None].expand(-1, -1, 3))
            prediction = model.predict(image)
            scores = branch_scores(prediction)
            predicted = rigid_candidates(model, prediction, chart)
            site_um = (predicted - truth[:, None, None]).norm(dim=-1).mean(-1)
            five = rigid_candidates(model, prediction, corners[None].expand(4, -1, -1))
            true_five = rigid_points_090(target_state, target_reflection,
                                         corners, (SIDE, SIDE))
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            target_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * target_normal[:, None]).sum(-1).abs().clamp_max(1))
            cost = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-cost.detach() / .7, -1)
            nearest_anchor = (target_normal @ model.normal_frames[:, :, 2].T).abs().topk(4, -1).indices
            near_cost = cost.reshape(4, 4, model.normal_modes, 2).min(-1).values.gather(
                2, nearest_anchor[:, None].expand(-1, 4, -1))
            pose_loss = (-1.5 * torch.logsumexp(scores - cost / 1.5, -1)
                + .5 * F.kl_div(scores, target_mass, reduction='none').sum(-1)
                + .5 * near_cost.mean((1, 2))).mean()

            near = cost.detach().argmin(-1)
            first = scores.argmax(-1)
            mode = torch.stack((near // 2, first // 2, torch.zeros_like(near)), -1)
            reflection = torch.stack((near % 2, first % 2, target_reflection), -1)
            rows = torch.arange(4, device='cuda')
            selected_state = torch.stack((prediction['state'][rows, near // 2],
                prediction['state'][rows, first // 2], target_state), 1)
            mapped = model.map_sites(prediction, chart, mode, reflection, state=selected_state)
            map_gap = (mapped['coordinates'] - truth[:, None]).square().sum(-1).add(250**2).sqrt().sub(250)
            map_gap = map_gap.mean(-1) / 1000
            map_loss = (map_gap[:, 2] + .5 * map_gap[:, 0] + .2 * map_gap[:, 1]).mean()
            support = valid.float()
            pos_weight = ((1 - support).sum() / support.sum()).clamp(1, 10)
            support_loss = F.binary_cross_entropy_with_logits(
                mapped['correspondence_logit'][:, 2, 0], support, pos_weight=pos_weight)
            warp_penalty = mapped['local_displacement_um'][:, 2].square().mean().sqrt() / 1000
            loss = pose_loss + map_loss + .1 * support_loss + .02 * warp_penalty

            domain = ('lupori', 'coronal', 'sagittal')[(step - 1) % 3]
            if donor_cursors[domain] == len(donor_orders[domain]):
                donor_orders[domain] = train_rng.permutation(donor_pools[domain]).tolist()
                donor_cursors[domain] = 0
            donor = donor_orders[domain][donor_cursors[domain]]
            donor_cursors[domain] += 1
            if domain == 'lupori':
                index = int(train_rng.choice(lupori_by_animal[donor]))
                acquired = acquired_input(lupori_images[index], domain)
                reference_ouv = lupori_ouvs[index]
                draw_record = lupori_records[index]
            elif domain == 'coronal':
                if donor not in donor_cache:
                    shard = CORONAL_TRAIN / f'donor_{split["donor_shards"][donor]["animal_id"]}'
                    receipt = json.loads((shard / 'completed.json').read_text(encoding='utf8'))
                    assert sha(shard / 'completed.json') == split['donor_shards'][donor]['receipt_sha256']
                    assert sha(shard / 'records.jsonl') == receipt['output_sha256']['records.jsonl']
                    assert sha(shard / 'images.npy') == receipt['output_sha256']['images.npy']
                    cohort = [json.loads(line) for line in (shard / 'records.jsonl').open(encoding='utf8')]
                    assert all(row['training_split'] == 'train' and row['animal_partition_key'] == donor
                               for row in cohort)
                    donor_cache[donor] = (shard, cohort)
                shard, cohort = donor_cache[donor]
                row = cohort[int(train_rng.integers(len(cohort)))]
                raw = np.load(shard / 'images.npy', mmap_mode='r')[row['donor_array_row'], 0]
                acquired = acquired_input(raw, domain)
                reference_ouv = weak_ouv(row, domain)
                draw_record = {'animal_partition_key': donor, 'animal_id': row['animal_id'],
                    'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                    'section_id': row['section_id'], 'array_row_index': row['donor_array_row'],
                    'source_image_sha256': row['source_image_sha256']}
            else:
                cohort = sagittal_by_donor[donor]
                row = cohort[int(train_rng.integers(len(cohort)))]
                acquired = acquired_input(sagittal_images[row['array_row_index'], 0], domain)
                reference_ouv = weak_ouv(row, domain)
                draw_record = {'donor_id': donor, 'animal_id': donor, 'specimen_id': row['specimen_id'],
                    'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
                    'array_row_index': row['array_row_index'], 'source_image_sha256': row['image_sha256']}
            draw_file.write(json.dumps({'step': step, 'slot': 4, 'kind': 'acquired_weak',
                'domain': domain, **draw_record}, allow_nan=False) + '\n')
            acquired_prediction = model.predict(acquired)
            acquired_scores = branch_scores(acquired_prediction)[0]
            acquired_five = rigid_candidates(model, acquired_prediction, corners[None])[0]
            reference_five = reference_ouv[0] + corners[:, :1] * reference_ouv[1] \
                + corners[:, 1:] * reference_ouv[2]
            five_um = (acquired_five - reference_five).norm(dim=-1).mean(-1)
            weak_normal = F.normalize(torch.cross(reference_ouv[1], reference_ouv[2], dim=-1), dim=-1)
            estimated_normal = full_frame_state_to_components(acquired_prediction['state'])[1][0, :, :, 2]
            normal_um = 4000 * (1 - (estimated_normal * weak_normal).sum(-1).abs().clamp_max(1))
            weak_cost = ((five_um - 1000).clamp_min(0) +
                (normal_um[:, None] - 100).clamp_min(0)).flatten() / 1000
            weak_target = F.softmax(-weak_cost.detach() / .7, -1)
            nearest_anchor = (weak_normal @ model.normal_frames[:, :, 2].T).abs().topk(4).indices
            near_weak = weak_cost.reshape(4, model.normal_modes, 2).min(-1).values[:, nearest_anchor]
            acquired_loss = (-1.5 * torch.logsumexp(acquired_scores - weak_cost / 1.5, 0)
                + .5 * F.kl_div(acquired_scores, weak_target, reduction='sum')
                + .5 * near_weak.mean())
            loss = loss + .3 * acquired_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            training_file.write(json.dumps({'batch': step, 'synthetic_sections': 4,
                'pose_loss': float(pose_loss.detach()), 'map_loss': float(map_loss.detach()),
                'true_state_mapped_gap_mm': float(map_gap[:, 2].mean().detach()),
                'near_predicted_mapped_gap_mm': float(map_gap[:, 0].mean().detach()),
                'first_predicted_mapped_gap_mm': float(map_gap[:, 1].mean().detach()),
                'support_bce': float(support_loss.detach()),
                'warp_rms_mm': float(warp_penalty.detach()),
                'acquired_domain': domain, 'acquired_weak_loss': float(acquired_loss.detach()),
                'gradient_norm': float(gradient), 'loss': float(loss.detach()),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')

        if step in STAGES:
            model.eval()
            metrics, rows = evaluate(model)
            readouts[step] = metrics
            readout_file.write(json.dumps({'batch': step, 'metrics': metrics,
                'rows': rows}, allow_nan=False) + '\n')
            readout_file.flush()
            training_file.flush()
            draw_file.flush()
            torch.save({'batch': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config,
                'calibrated': False, 'model_promoted': False}, RUN / f'joint_{step:05d}.pt')
            if step == UPDATES:
                previous = readouts[0]
                stage_gates[str(step)] = {
                    'inner_mapped_nonregression_321_1k_le0p20mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        3.920,
                    'inner_tail_nonregression_321_1k_plus_le2': metrics['inner']['first_mapped_over_5mm'] <=
                        21,
                    'steep_mapped_nonregression_321_1k_le0p20mm': metrics['steep']['mean_first_mapped_gap_mm'] <=
                        4.004,
                    'sparse_mapped_nonregression_321_1k_le0p20mm': metrics['sparse']['mean_first_mapped_gap_mm'] <=
                        5.756,
                    'coronal_first_family_retained_ge8of10': metrics['coronal']['first_axis_correct'] >= 8,
                    'coronal_top16_family_retained_ge9of10': metrics['coronal']['top16_contains_weak_axis'] >= 9,
                    'sagittal_first_family_gain_ge4of8': metrics['sagittal']['first_axis_correct'] >=
                        previous['sagittal']['first_axis_correct'] + 4,
                    'sagittal_first_family_ge6of8': metrics['sagittal']['first_axis_correct'] >= 6,
                    'sagittal_top16_family_ge6of8': metrics['sagittal']['top16_contains_weak_axis'] >= 6}
            print(json.dumps({'event': 'readout', 'batch': step, 'metrics': metrics,
                'gate': stage_gates.get(str(step))}, allow_nan=False), flush=True)
            model.train()
            if str(step) in stage_gates:
                print(json.dumps({'event': 'stage_gate', 'batch': step,
                    'passed': all(stage_gates[str(step)].values()),
                    'clauses': stage_gates[str(step)]}), flush=True)

(RUN / 'completed.json').write_text(json.dumps({'candidate': 326,
    'planned_optimizer_batches': UPDATES, 'terminal_batch': step,
    'stage_gates': stage_gates,
    'transfer_feasibility_gate_passed': all(stage_gates[str(UPDATES)].values()),
    'output_sha256': {path.name: sha(path) for path in RUN.iterdir()
                      if path.name != 'completed.json'},
    'calibrated': False, 'model_promoted': False}, indent=2) + '\n', encoding='utf8')
