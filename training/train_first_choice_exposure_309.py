"""Frozen TRAIN-only 309 first-choice exposure study."""
import copy
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

from training.arbitrary_plane_fov_balance_v10 import (
    CODE_SHA256 as V10_SHA, FOV_SCALE, sample_fov_balance_v10)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261014309, 192, 20000
STEPS = (0, 5000, 10000, 20000)
TRAIN, GUARD = ('CC2B', 'CC4B', 'CC4A'), 'AL1A'
SAGITTAL_HELDOUT = {10164, 10177, 10180, 10184, 10236, 10237, 10239, 10253,
                    10255, 10301, 10304, 10308, 10336, 10349, 10363, 10371,
                    10407, 10412, 10413, 10417}
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/FIRST_CHOICE_EXPOSURE_309_PROTOCOL_20261014.md'
SPLIT = ROOT / 'data/coronal_train_split_308_001/split.json'
CORPUS = ROOT / 'data/joint_v7_reserved_train_images_192_001'
SAGITTAL = ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
LUPORI_SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PARENT = ROOT / 'runs/real_corpus_diversity_308_001/control/pose_local_04000.pt'
RUN = ROOT / 'runs/first_choice_exposure_309_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == PARENT.drive.upper() == 'I:'
assert not RUN.exists() and FOV_SCALE == .72
assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == 'dd0451ec74565022240988bee80089924ca3d406aebae97870a3e91c22855489'
assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == 'b0e5ce4060c6192c2fc166494459c5134beda3caa55488d01a6581c374821bbf'
with PARENT.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert parent_sha == '33d7440e6fa2a75567dcc65675a466314af0fb2a35b981654fa0b4b3b1428f11'
parent_arm_completion = PARENT.parent / 'completed.json'
parent_run_completion = PARENT.parents[1] / 'completed.json'
parent_arm_receipt = json.loads(parent_arm_completion.read_text(encoding='utf8'))
parent_run_receipt = json.loads(parent_run_completion.read_text(encoding='utf8'))
assert parent_arm_receipt['output_sha256'][PARENT.name] == parent_sha
assert parent_run_receipt['arm_completion_sha256']['control'] == hashlib.sha256(
    parent_arm_completion.read_bytes()).hexdigest()
assert hashlib.sha256(SOURCE.with_name('arbitrary_plane_fov_balance_v10.py').read_bytes()).hexdigest() == V10_SHA
split = json.loads(SPLIT.read_text(encoding='utf8'))
lupori_split = json.loads(LUPORI_SPLIT.read_text(encoding='utf8'))
assert tuple(lupori_split['train']) == (GUARD, *TRAIN)
assert ARCHIVE.stat().st_size == lupori_split['archive_bytes']
assert split['counts'] == {'source_donors': 1885, 'old_excluded': 58,
                           'new_inner': 100, 'gradient_donors': 1727}
gradient_donors = split['gradient_donors']
inner = split['new_inner_readout']
assert len(gradient_donors) == 1727 and len(inner) == 100
assert not (set(gradient_donors) & {r['animal_partition_key'] for r in inner})
assert not (set(gradient_donors) & set(split['prior_pool_excluded_from_308_gradients']))

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
v9 = load_streaming_synthetic_v7_64('cuda')
v10 = {**v9, 'subjects': copy.deepcopy(v9['subjects']),
       'nuisance': v9['nuisance'].copy(), 'provenance': copy.deepcopy(v9['provenance'])}
v10['provenance']['virtual_subjects'] = v10['subjects']
for subject in v10['subjects']:
    bounds = np.asarray(subject['bounds_um'])
    middle = bounds.mean(0)
    subject['bounds_um'] = (middle + FOV_SCALE * (bounds - middle)).tolist()
v10['nuisance'][:, 2:4] *= FOV_SCALE
v10['provenance']['fov_balance_v10'] = {'code_sha256': V10_SHA, 'fov_scale': FOV_SCALE}

records, images, masks, ouvs = [], [], [], []
with ZipFile(ARCHIVE) as archive:
    for animal in (GUARD, *TRAIN):
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
            mask = np.asarray(Image.open(BytesIO(archive.read(
                f'{animal}/masks/{section}-mask.png'))).convert('L').resize(
                    (SIDE, SIDE), Image.Resampling.NEAREST)) > 127
            source_ouv = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            o = quicknii_to_allen_points(source_ouv[0]) * 25
            u = quicknii_to_allen_vectors(source_ouv[1]) * 25
            v = quicknii_to_allen_vectors(source_ouv[2]) * 25
            records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'source_section_id': section,
                'archive_member': member, 'source_image_sha256': hashlib.sha256(raw).hexdigest()})
            images.append(gray)
            masks.append(mask)
            ouvs.append(torch.stack((o, u, v)))
assert len(records) == 240
guard_indices = [i for i, row in enumerate(records) if row['animal_id'] == GUARD]
train_indices = [i for i, row in enumerate(records) if row['animal_id'] in TRAIN]
assert len(guard_indices) == 52 and len(train_indices) == 188
images = torch.from_numpy(np.stack(images)).cuda()
masks = torch.from_numpy(np.stack(masks)).cuda()
ouvs = torch.stack(ouvs).cuda()
normal_targets = F.normalize(torch.cross(ouvs[:, 1], ouvs[:, 2], dim=-1), dim=-1)

sag_rows = [json.loads(line) for line in (SAGITTAL / 'geometry.jsonl').open(encoding='utf8')]
assert len(sag_rows) == 653 and all(row['split'] == 'train' for row in sag_rows)
sag_by_donor = {}
for row in sag_rows:
    sag_by_donor.setdefault(row['donor_id'], []).append(row)
assert len(sag_by_donor) == 40 and SAGITTAL_HELDOUT <= set(sag_by_donor)
sag_readout = [sorted(sag_by_donor[donor],
    key=lambda row: (row['section_number'], row['section_id']))[
    (len(sag_by_donor[donor]) - 1) // 2] for donor in sorted(SAGITTAL_HELDOUT)]
selected_sag = {donor: sorted(cohort,
    key=lambda row: (row['section_number'], row['section_id']))[
    (len(cohort) - 1) // 2]['section_id'] for donor, cohort in sag_by_donor.items()}
sag_train = {donor: [row for row in cohort if row['section_id'] != selected_sag[donor]]
             for donor, cohort in sag_by_donor.items() if donor not in SAGITTAL_HELDOUT}
assert len(sag_train) == 20 and all(sag_train.values())
sag_images = np.load(SAGITTAL / 'model_input.npy', mmap_mode='r')

def weak_ouv(row, sagittal=False):
    affine = np.asarray(row['model_pixel_to_ccf_ref9_ap_dv_ml_um' if sagittal
                            else 'model_pixel_to_ap_dv_ml_um'], dtype=np.float32)
    offset, scale = (1 / 6, 256) if sagittal else (0., SIDE)
    origin = affine[:, 2] + offset * (affine[:, 0] + affine[:, 1])
    return torch.as_tensor(np.stack((origin, scale * affine[:, 0],
                                      scale * affine[:, 1])), device='cuda')

donor_cache = {}

def donor_rows(key):
    if key in donor_cache:
        return donor_cache[key]
    donor = split['donor_shards'][key]
    shard = CORPUS / f'donor_{donor["animal_id"]}'
    assert hashlib.sha256((shard / 'completed.json').read_bytes()).hexdigest() == donor['receipt_sha256']
    receipt = json.loads((shard / 'completed.json').read_text(encoding='utf8'))
    with (shard / 'images.npy').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['output_sha256']['images.npy']
    raw = (shard / 'records.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['output_sha256']['records.jsonl']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == donor['image_count'] and all(
        row['animal_partition_key'] == key and row['training_split'] == 'train' for row in rows)
    donor_cache[key] = (shard, rows)
    return donor_cache[key]

RUN.mkdir(parents=True, exist_ok=False)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
flags = torch.arange(2, device='cuda')[None, None]
panels = {}
panel_rng = np.random.default_rng(SEED + 1)
panel_seeds, panel_ids = set(), set()
with (RUN / 'panel_draws.jsonl').open('w', encoding='utf8') as file, torch.inference_mode():
    for version, count in (('v9', 256), ('v10', 256), ('v10_sparse', 64),
                           ('v10_steep', 64)):
        panel = []
        context = v9 if version == 'v9' else v10
        for slot in range(count):
            base = 56 + slot % 8
            while True:
                if version in ('v10_sparse', 'v10_steep'):
                    base = int(panel_rng.integers(56, 64))
                variant = int(panel_rng.integers(VARIANTS))
                seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
                assert seed not in panel_seeds
                panel_seeds.add(seed)
                sample = (sample_postartifact_gauge_v1(context,
                    [base * VARIANTS + variant], seed, SIDE) if version == 'v9'
                    else sample_fov_balance_v10(context,
                    [base * VARIANTS + variant], seed, SIDE))
                valid = sample['valid_mask'][0]
                visible = float(((sample['visible'][0] > .05) &
                    ((sample['inputs'][0, 2] < .5) | sample['brush_mask'][0])).float().mean())
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                obliquity = math.degrees(math.acos(float(normal.abs().max().clamp(0, 1))))
                accepted = bool(sample['eligible'][0]) and (
                    version != 'v10_sparse' or visible < .15) and (
                    version != 'v10_steep' or obliquity >= 45)
                provenance = sample['provenance'][0]
                file.write(json.dumps({'cohort': version, 'slot': slot, 'base': base,
                    'variant': variant, 'seed': seed, 'accepted': accepted,
                    'postartifact_visible_fraction': visible, 'obliquity_deg': obliquity,
                    'provenance': provenance}, allow_nan=False) + '\n')
                if accepted:
                    break
            physical_id = provenance['physical_section_id']
            assert provenance['split'] == 'train' and physical_id not in panel_ids
            panel_ids.add(physical_id)
            sites = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                generator=torch.Generator(device='cuda').manual_seed(seed))
            chart = torch.stack((sites.remainder(SIDE),
                sites.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            panel.append({'inputs': sample['inputs'][0].cpu(), 'chart': chart.cpu(),
                'truth': sample['centre'][0].reshape(-1, 3)[sites].cpu(),
                'state': sample['state'][0].cpu(), 'meta': {
                'physical_section_id': physical_id, 'base': base, 'variant': variant,
                'seed': seed, 'postartifact_visible_fraction': visible,
                'obliquity_deg': obliquity,
                'framing': provenance['framing']['kind'],
                'sampling_branch': provenance['sampling_branch']}})
        panels[version] = panel
torch.save(panels, RUN / 'panels.pt')

shared_rng = np.random.default_rng(SEED + 2)
lupori_by_animal = {animal: [index for index in train_indices
    if records[index]['animal_id'] == animal] for animal in TRAIN}
assert all(lupori_by_animal.values())
lupori_order, lupori_cursor = shared_rng.permutation(TRAIN), 0
sag_order, sag_cursor = shared_rng.permutation(sorted(sag_train)), 0
coronal_order, coronal_cursor = shared_rng.permutation(gradient_donors), 0
shared = []
for step in range(1, UPDATES + 1):
    domain = ('lupori', 'coronal', 'sagittal')[(step - 1) % 3]
    acquired = None
    if domain == 'lupori':
        if lupori_cursor == len(lupori_order):
            lupori_order, lupori_cursor = shared_rng.permutation(TRAIN), 0
        animal = str(lupori_order[lupori_cursor])
        lupori_cursor += 1
        cohort = lupori_by_animal[animal]
        acquired = cohort[int(shared_rng.integers(len(cohort)))]
    elif domain == 'coronal':
        if coronal_cursor == len(coronal_order):
            coronal_order, coronal_cursor = shared_rng.permutation(gradient_donors), 0
        key = str(coronal_order[coronal_cursor])
        coronal_cursor += 1
        _, cohort = donor_rows(key)
        acquired = cohort[int(shared_rng.integers(len(cohort)))]
    if domain == 'sagittal':
        if sag_cursor == len(sag_order):
            sag_order, sag_cursor = shared_rng.permutation(sorted(sag_train)), 0
        donor = int(sag_order[sag_cursor])
        sag_cursor += 1
        cohort = sag_train[donor]
        acquired = cohort[int(shared_rng.integers(len(cohort)))]
    bases = [int(base) for base in shared_rng.integers(56, size=3)]
    shared.append((domain, acquired, bases))
assert len({row['animal_partition_key'] for domain, row, _ in shared
    if domain == 'coronal'}) == 1727
with (RUN / 'schedule.jsonl').open('w', encoding='utf8') as file:
    for step, (domain, acquired, bases) in enumerate(shared, 1):
        item = {'step': step, 'synthetic_bases': bases, 'acquired_domain': domain}
        if domain == 'lupori':
            item['lupori'] = records[acquired]
        elif domain == 'sagittal':
            item['sagittal'] = {key: acquired[key] for key in ('donor_id', 'animal_id',
                'specimen_id', 'experiment_id', 'section_id', 'array_row_index',
                'image_sha256') if key in acquired}
        else:
            item['coronal'] = {key: acquired[key] for key in (
                'animal_partition_key', 'animal_id', 'specimen_id', 'experiment_id',
                'section_id', 'donor_array_row', 'source_image_sha256')}
        file.write(json.dumps(item, allow_nan=False) + '\n')

config = {'candidate': 309, 'seed': SEED, 'optimizer_batches': UPDATES,
    'readout_steps': STEPS, 'gradient_donors': len(gradient_donors),
    'batch_slots': ['v9', 'v10', 'v10', 'cyclic_lupori_coronal_sagittal'],
    'acquired_presentations': {domain: sum(row_domain == domain for row_domain, _, _ in shared)
        for domain in ('lupori', 'coronal', 'sagittal')},
    'synthetic_train_bases': list(range(56)), 'synthetic_inner_bases': list(range(56, 64)),
    'lupori_gradient_animals': TRAIN, 'lupori_guard_animal': GUARD,
    'sagittal_heldout_donors': sorted(SAGITTAL_HELDOUT),
    'weak_allen_cost_um': 'max(five-site gap - 1000, 0) + max(unsigned-normal penalty - 100, 0)',
    'weak_allen_loss_weight': .3,
    'optimized_modules': ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global'],
    'optimizer': 'restored 308 control AdamW moments; constant LR pose 4e-6, shared 1e-6',
    'sparse_visible_rule': 'visible > 0.05 AND (brush absent OR brush mask), fraction < 0.15',
    'steep_rule': 'postartifact unsigned plane normal >=45 degrees from nearest cardinal axis',
    'parent_sha256': parent_sha,
    'parent_arm_completion_sha256': hashlib.sha256(parent_arm_completion.read_bytes()).hexdigest(),
    'parent_run_completion_sha256': hashlib.sha256(parent_run_completion.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    'corpus_source_manifest_sha256': split['source_manifest_sha256'],
    'lupori_split_sha256': hashlib.sha256(LUPORI_SPLIT.read_bytes()).hexdigest(),
    'lupori_archive_bytes': lupori_split['archive_bytes'],
    'lupori_archive_md5_from_split': lupori_split['archive_md5'],
    'sagittal_manifest_sha256': hashlib.sha256((SAGITTAL / 'geometry.jsonl').read_bytes()).hexdigest(),
    'model_source_sha256': hashlib.sha256(SOURCE.with_name(
        'arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest(),
    'inference_source_sha256': hashlib.sha256(SOURCE.with_name(
        'arbitrary_plane_one_shot_inference.py').read_bytes()).hexdigest(),
    'postartifact_gauge_source_sha256': hashlib.sha256(SOURCE.with_name(
        'arbitrary_plane_postartifact_gauge_v1.py').read_bytes()).hexdigest(),
    'v10_generator_sha256': V10_SHA,
    'v9_generator_input_sha256': v9['provenance']['input_sha256'],
    'panels_sha256': hashlib.file_digest((RUN / 'panels.pt').open('rb'), 'sha256').hexdigest(),
    'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.jsonl').read_bytes()).hexdigest(),
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

def evaluate(model):
    rows = []
    with torch.inference_mode():
        for cohort, panel in panels.items():
            for start in range(0, len(panel), 4):
                group = panel[start:start + 4]
                batch = torch.stack([case['inputs'] for case in group]).cuda()
                chart = torch.stack([case['chart'] for case in group]).cuda()
                truth = torch.stack([case['truth'] for case in group]).cuda()
                p = model.predict(batch)
                states = p['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(len(states), model.modes, 2)
                mass = (p['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-p['reflection_logit']),
                    F.logsigmoid(p['reflection_logit'])), -1)).flatten(1)
                mapped = rigid_points_090(states, reflections,
                                          chart[:, None, None], (SIDE, SIDE))
                errors = (mapped - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                first = mass.argmax(-1)
                top16 = mass.topk(16, -1).indices
                for j, case in enumerate(group):
                    rows.append({'cohort': cohort, **case['meta'],
                        'first_rigid_site_gap_mm': float(errors[j, first[j]]),
                        'top16_oracle_gap_mm': float(errors[j, top16[j]].min())})

        for start in range(0, len(guard_indices), 4):
            ids = guard_indices[start:start + 4]
            batch = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
            batch[:, 0] = images[ids]
            p = model.predict(batch)
            states = p['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(states), model.modes, 2)
            mass = (p['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-p['reflection_logit']),
                F.logsigmoid(p['reflection_logit'])), -1)).flatten(1)
            reference = ouvs[ids, 0, None] + corners[None, :, :1] * ouvs[ids, 1, None] \
                + corners[None, :, 1:] * ouvs[ids, 2, None]
            mapped = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            errors = (mapped - reference[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            top16 = mass.topk(16, -1).indices
            centres, frames, _ = full_frame_state_to_components(p['state'])
            for j, index in enumerate(ids):
                mode = int(first[j]) // 2
                weak_centre = ouvs[index, 0] + .5 * (ouvs[index, 1] + ouvs[index, 2])
                weak_axis = int(normal_targets[index].abs().argmax())
                rows.append({'cohort': 'lupori', **records[index],
                    'first_five_site_gap_mm': float(errors[j, first[j]]),
                    'top16_oracle_gap_mm': float(errors[j, top16[j]].min()),
                    'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                    'first_axis_correct': int(frames[j, mode, :, 2].abs().argmax()) == weak_axis,
                    'first_weak_centre_gap_mm': float((centres[j, mode] - weak_centre).norm() / 1000)})

        for cohort, cases in (('coronal_new', inner), ('sagittal', sag_readout)):
            for row in cases:
                if cohort == 'coronal_new':
                    shard, source_rows = donor_rows(row['animal_partition_key'])
                    source = source_rows[row['donor_array_row']]
                    assert source['section_id'] == row['section_id']
                    assert source['source_image_sha256'] == row['source_image_sha256']
                    raw = np.load(shard / 'images.npy', mmap_mode='r')[row['donor_array_row'], 0]
                else:
                    raw = sag_images[row['array_row_index'], 0]
                image = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None, None].cuda()
                if cohort == 'sagittal':
                    image = F.interpolate(image, (SIDE, SIDE), mode='area')
                batch = torch.cat((image, torch.zeros(1, 4, SIDE, SIDE, device='cuda')), 1)
                p = model.predict(batch)
                mass = (p['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-p['reflection_logit']),
                    F.logsigmoid(p['reflection_logit'])), -1)).flatten()
                first = int(mass.argmax())
                top16 = mass.topk(16).indices
                centres, frames, _ = full_frame_state_to_components(p['state'][0])
                normals = frames[:, :, 2]
                reference = weak_ouv(row, cohort == 'sagittal')
                weak_normal = F.normalize(torch.cross(reference[1], reference[2], dim=-1), dim=-1)
                weak_axis = int(weak_normal.abs().argmax())
                weak_centre = reference[0] + .5 * (reference[1] + reference[2])
                mode = first // 2
                rows.append({'cohort': cohort,
                    'animal_partition_key': row.get('animal_partition_key'),
                    'animal_id': row.get('animal_id', row.get('donor_id')),
                    'donor_id': row.get('donor_id'),
                    'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                    'section_id': row['section_id'],
                    'array_row_index': row.get('donor_array_row', row.get('array_row_index')),
                    'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256')),
                    'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                    'first_axis_correct': int(normals[mode].abs().argmax()) == weak_axis,
                    'top16_contains_weak_axis': any(
                        int(normals[int(branch // 2)].abs().argmax()) == weak_axis
                        for branch in top16),
                    'first_weak_centre_gap_mm': float((centres[mode] - weak_centre).norm() / 1000),
                    'first_weak_normal_angle_deg': float(torch.rad2deg(torch.acos(
                        (normals[mode] * weak_normal).sum().abs().clamp(0, 1))))})

    metrics = {}
    for cohort in ('v9', 'v10', 'v10_sparse', 'v10_steep'):
        group = [row for row in rows if row['cohort'] == cohort]
        metrics[cohort] = {'n': len(group),
            'mean_first_gap_mm': float(np.mean([row['first_rigid_site_gap_mm'] for row in group])),
            'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5 for row in group)}
    group = [row for row in rows if row['cohort'] == 'lupori']
    metrics['lupori'] = {'n': len(group),
        'mean_first_gap_mm': float(np.mean([row['first_five_site_gap_mm'] for row in group])),
        'median_weak_centre_gap_mm': float(np.median([
            row['first_weak_centre_gap_mm'] for row in group])),
        'first_axis_correct': sum(row['first_axis_correct'] for row in group)}
    for cohort in ('coronal_new', 'sagittal'):
        group = [row for row in rows if row['cohort'] == cohort]
        metrics[cohort] = {'n': len(group),
            'median_weak_centre_gap_mm': float(np.median([
                row['first_weak_centre_gap_mm'] for row in group])),
            'first_axis_correct': sum(row['first_axis_correct'] for row in group)}
    assert [metrics[cohort]['n'] for cohort in (
        'v9', 'v10', 'v10_sparse', 'v10_steep', 'lupori', 'coronal_new', 'sagittal')
        ] == [256, 256, 64, 64, 52, 100, 20]
    return metrics, rows

all_readouts = {}
for arm in ('single',):
    path = RUN / arm
    path.mkdir()
    model, parent_config = load_one_shot_checkpoint(PARENT, device='cuda')
    assert parent_config['candidate'] == 308 and model.modes == 80
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
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(PARENT, map_location='cpu', weights_only=True)
    assert saved['step'] == 5000 and saved['local_step'] == 4000 and saved['arm'] == 'control'
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    assert [group['base_lr'] for group in optimizer.param_groups] == [2e-5, 5e-6]
    for group in optimizer.param_groups:
        group['lr'] = .2 * group['base_lr']
    parameters = [p for group in groups for p in group['params']]
    torch.manual_seed(SEED + 10)
    torch.cuda.manual_seed_all(SEED + 10)
    synthetic_rng = np.random.default_rng(SEED + 11)
    synthetic_sites = torch.Generator(device='cuda').manual_seed(SEED + 12)
    used_seeds, used_ids = panel_seeds.copy(), panel_ids.copy()
    started, readouts = time.perf_counter(), {}
    model.train()
    with (path / 'draws.jsonl').open('w', encoding='utf8') as draws, \
            (path / 'training.jsonl').open('w', encoding='utf8') as training, \
            (path / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
        for step in range(UPDATES + 1):
            if step:
                domain, acquired, bases = shared[step - 1]
                samples = []
                for slot, (version, context, base) in enumerate((
                        ('v9', v9, bases[0]), ('v10', v10, bases[1]),
                        ('v10', v10, bases[2]))):
                    while True:
                        variant = int(synthetic_rng.integers(VARIANTS))
                        seed = int(synthetic_rng.integers(2**63 - 1, dtype=np.int64))
                        assert seed not in used_seeds
                        used_seeds.add(seed)
                        with torch.no_grad():
                            sample = (sample_postartifact_gauge_v1(context,
                                [base * VARIANTS + variant], seed, SIDE) if version == 'v9'
                                else sample_fov_balance_v10(context,
                                [base * VARIANTS + variant], seed, SIDE))
                        provenance = sample['provenance'][0]
                        assert provenance['split'] == 'train'
                        assert provenance['physical_section_id'] not in used_ids
                        used_ids.add(provenance['physical_section_id'])
                        accepted = bool(sample['eligible'][0])
                        draws.write(json.dumps({'kind': 'synthetic', 'step': step,
                            'slot': slot, 'version': version, 'base': base,
                            'variant': variant, 'seed': seed, 'accepted': accepted,
                            'provenance': provenance}, allow_nan=False) + '\n')
                        if accepted:
                            samples.append(sample)
                            break
                if domain == 'lupori':
                    draws.write(json.dumps({'kind': 'lupori', 'step': step,
                        'slot': 3, **records[acquired]}, allow_nan=False) + '\n')
                else:
                    draws.write(json.dumps({'kind': 'allen', 'step': step, 'slot': 3,
                        'domain': domain,
                        'animal_partition_key': acquired.get('animal_partition_key'),
                        'animal_id': acquired.get('animal_id', acquired.get('donor_id')),
                        'donor_id': acquired.get('donor_id'),
                        'specimen_id': acquired['specimen_id'],
                        'experiment_id': acquired['experiment_id'],
                        'section_id': acquired['section_id'],
                        'array_row_index': acquired.get('donor_array_row',
                            acquired.get('array_row_index')),
                        'source_image_sha256': acquired.get('source_image_sha256',
                            acquired.get('image_sha256'))}, allow_nan=False) + '\n')

                synthetic = torch.cat([sample['inputs'] for sample in samples])
                target_state = torch.cat([sample['state'] for sample in samples])
                target_reflection = torch.cat([sample['reflection'] for sample in samples])
                valid = torch.cat([sample['valid_mask'] for sample in samples])
                centres = torch.cat([sample['centre'] for sample in samples])
                acquired_input = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
                if domain == 'lupori':
                    acquired_input[0, 0] = images[acquired]
                else:
                    if domain == 'coronal':
                        shard = CORPUS / f'donor_{acquired["animal_id"]}'
                        raw = np.load(shard / 'images.npy', mmap_mode='r')[acquired['donor_array_row'], 0]
                    else:
                        raw = sag_images[acquired['array_row_index'], 0]
                    image = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None, None].cuda()
                    if domain == 'sagittal':
                        image = F.interpolate(image, (SIDE, SIDE), mode='area')
                    acquired_input[:, :1] = image
                prediction = model.predict(torch.cat((synthetic, acquired_input)))
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(len(states), model.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

                pixels = torch.multinomial(valid.flatten(1).float(), 128,
                    replacement=True, generator=synthetic_sites)
                chart = torch.stack((pixels.remainder(SIDE),
                    pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                truth = centres.flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
                proposed = rigid_points_090(states[:3], reflections[:3],
                                            chart[:, None, None], (SIDE, SIDE))
                five = rigid_points_090(states[:3], reflections[:3], corners, (SIDE, SIDE))
                true_five = rigid_points_090(target_state, target_reflection,
                                             corners, (SIDE, SIDE))
                tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
                normal = full_frame_state_to_components(prediction['state'][:3])[1][..., :, 2]
                true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
                normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
                physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
                target_mass = F.softmax(-physical.detach() / .7, -1)
                nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
                near_mode = model.base_modes + nearest.indices
                near_cost = physical.reshape(3, model.modes, 2).min(-1).values.gather(1, near_mode)
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
                synthetic_loss = (-1.5 * torch.logsumexp(prior[:3] - physical / 1.5, -1)
                    + .5 * F.kl_div(prior[:3], target_mass, reduction='none').sum(-1)
                    + .5 * physical[:, :2 * model.base_modes].min(-1).values
                    + (neighbourhood * near_cost).sum(-1)).mean()

                site_record = {'kind': 'site_indices', 'step': step,
                    'synthetic_sha256': hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest()}
                acquired_normal = full_frame_state_to_components(prediction['state'][3:4]
                    )[1][..., :, 2][0]
                if domain == 'lupori':
                    rpixels = torch.multinomial(masks[acquired].flatten().float(), 128,
                        replacement=True, generator=torch.Generator(device='cuda').manual_seed(
                            SEED + 1000 + step))
                    site_record['lupori_sha256'] = hashlib.sha256(
                        rpixels.cpu().numpy().tobytes()).hexdigest()
                    rchart = torch.stack((rpixels.remainder(SIDE),
                        rpixels.div(SIDE, rounding_mode='floor')), -1).float()[None] / SIDE
                    reference = ouvs[acquired, 0] + rchart[..., :1] * ouvs[acquired, 1] + \
                        rchart[..., 1:] * ouvs[acquired, 2]
                    rproposed = rigid_points_090(states[3:4], reflections[3:4],
                        rchart[:, None, None], (SIDE, SIDE))
                    reference_five = ouvs[acquired, 0] + corners[:, :1] * ouvs[acquired, 1] + \
                        corners[:, 1:] * ouvs[acquired, 2]
                    rfive = rigid_points_090(states[3:4], reflections[3:4],
                        corners, (SIDE, SIDE))
                    tissue_um = (rproposed - reference[:, None, None]).norm(dim=-1).mean(-1)
                    five_um = (rfive - reference_five).norm(dim=-1).mean(-1)
                    acquired_true_normal = normal_targets[acquired]
                    normal_um = 4000 * (1 - (acquired_normal * acquired_true_normal).sum(-1
                        ).abs().clamp_max(1))
                    acquired_cost = (.75 * tissue_um + .25 * five_um +
                        normal_um[None, :, None]).flatten(1) / 1000
                    acquired_weight = 1.
                else:
                    acquired_ouv = weak_ouv(acquired, domain == 'sagittal')
                    reference_five = acquired_ouv[0] + corners[:, :1] * acquired_ouv[1] + \
                        corners[:, 1:] * acquired_ouv[2]
                    five = rigid_points_090(states[3:4], reflections[3:4],
                        corners, (SIDE, SIDE))
                    five_um = (five - reference_five).norm(dim=-1).mean(-1)
                    acquired_true_normal = F.normalize(torch.cross(
                        acquired_ouv[1], acquired_ouv[2], dim=-1), dim=-1)
                    normal_um = 4000 * (1 - (acquired_normal * acquired_true_normal).sum(-1
                        ).abs().clamp_max(1))
                    acquired_cost = ((five_um - 1000).clamp_min(0) +
                        (normal_um[None, :, None] - 100).clamp_min(0)).flatten(1) / 1000
                    acquired_weight = .3
                draws.write(json.dumps(site_record) + '\n')
                acquired_mass = F.softmax(-acquired_cost.detach() / .7, -1)
                nearest = (acquired_true_normal @ model.normal_anchor_frames[:, :, 2].T
                    ).abs().topk(4)
                near_mode = model.base_modes + nearest.indices
                near_cost = acquired_cost.reshape(model.modes, 2).min(-1).values[near_mode]
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:1]), -1)
                acquired_loss = (-1.5 * torch.logsumexp(prior[3] - acquired_cost[0] / 1.5, -1)
                    + .5 * F.kl_div(prior[3], acquired_mass[0], reduction='sum')
                    + .5 * acquired_cost[0, :2 * model.base_modes].min()
                    + (neighbourhood * near_cost).sum())
                loss = synthetic_loss + acquired_weight * acquired_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                training.write(json.dumps({'step': step, 'synthetic_loss': float(synthetic_loss.detach()),
                    'acquired_weak_loss': float(acquired_loss.detach()),
                    'acquired_loss_weight': acquired_weight,
                    'acquired_domain': domain, 'loss': float(loss.detach()),
                    'gradient_norm': float(gradient),
                    'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')

            if step in STEPS:
                model.eval()
                metrics, rows = evaluate(model)
                readouts[str(step)] = metrics
                readout_file.write(json.dumps({'step': step, 'metrics': metrics,
                    'rows': rows}, allow_nan=False) + '\n')
                readout_file.flush()
                training.flush()
                draws.flush()
                torch.save({'step': 5000 + step, 'local_step': step,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': config, 'arm': arm, 'calibrated': False},
                    path / f'pose_local_{step:05d}.pt')
                print(json.dumps({'event': 'readout', 'arm': arm,
                    'step': step, 'metrics': metrics}), flush=True)
                model.train()

    all_readouts[arm] = readouts
    (path / 'completed.json').write_text(json.dumps({'arm': arm, 'updates': UPDATES,
        'readouts': readouts, 'output_sha256': {file.name: hashlib.file_digest(
            file.open('rb'), 'sha256').hexdigest() for file in path.iterdir()
            if file.name != 'completed.json'}, 'calibrated': False}, indent=2), encoding='utf8')
    del model, optimizer
    torch.cuda.empty_cache()

baseline = all_readouts['single']['0']
clauses_by_step = {}
for step in STEPS:
    result = all_readouts['single'][str(step)]
    clauses_by_step[str(step)] = {
        'v10_first_mean_improves_ge0p35mm':
            result['v10']['mean_first_gap_mm'] <= baseline['v10']['mean_first_gap_mm'] - .35,
        'v9_first_mean_regression_le0p15mm':
            result['v9']['mean_first_gap_mm'] <= baseline['v9']['mean_first_gap_mm'] + .15,
        'v10_top16_capture_loss_le4':
            result['v10']['top16_capture_1p5mm'] >= baseline['v10']['top16_capture_1p5mm'] - 4,
        'v10_sparse_first_mean_regression_le0p20mm':
            result['v10_sparse']['mean_first_gap_mm'] <=
            baseline['v10_sparse']['mean_first_gap_mm'] + .20,
        'v10_steep_first_mean_regression_le0p20mm':
            result['v10_steep']['mean_first_gap_mm'] <=
            baseline['v10_steep']['mean_first_gap_mm'] + .20,
        'lupori_first_mean_regression_le0p20mm':
            result['lupori']['mean_first_gap_mm'] <= baseline['lupori']['mean_first_gap_mm'] + .20,
        'coronal_median_weak_centre_regression_le0p20mm':
            result['coronal_new']['median_weak_centre_gap_mm'] <=
            baseline['coronal_new']['median_weak_centre_gap_mm'] + .20,
        'sagittal_first_axis_loss_le1':
            result['sagittal']['first_axis_correct'] >=
            baseline['sagittal']['first_axis_correct'] - 1,
    }
guarded = [step for step in STEPS if all(value for key, value in
    clauses_by_step[str(step)].items() if key != 'v10_first_mean_improves_ge0p35mm')]
selected = min(guarded, key=lambda step: (
    all_readouts['single'][str(step)]['v10']['mean_first_gap_mm'], step))
summary = {'candidate': 309, 'gate_pass': all(clauses_by_step[str(UPDATES)].values()),
    'clauses_by_step': clauses_by_step, 'selected_step': selected,
    'selected_checkpoint': f'single/pose_local_{selected:05d}.pt',
    'readouts': all_readouts['single'],
    'interpretation': 'TRAIN-inner first-choice exposure gate only; not physical-oblique validation or deployment',
    'output_sha256': {file.name: hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()
        for file in RUN.iterdir() if file.is_file() and file.name != 'completed.json'},
    'training_completion_sha256': hashlib.sha256((RUN / 'single/completed.json').read_bytes()
        ).hexdigest(),
    'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'gate_pass': summary['gate_pass'],
                  'selected_step': selected, 'clauses': clauses_by_step[str(UPDATES)]}), flush=True)
