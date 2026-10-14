"""Matched TRAIN-only 305 pose continuation with one v10 framing slot."""
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

SEED, SIDE, UPDATES = 20261014312, 192, 400
STEPS = (0, 200, 400)
TRAIN, GUARD = ('CC2B', 'CC4B', 'CC4A'), 'AL1A'
HELDOUT = {
    'coronal': {14543, 14599, 14601, 14642, 14643, 14644, 14711, 14745, 15060,
                15111, 15112, 15158, 15162, 15304, 15305, 15346, 15347, 15396,
                15653, 15979},
    'sagittal': {10164, 10177, 10180, 10184, 10236, 10237, 10239, 10253, 10255,
                 10301, 10304, 10308, 10336, 10349, 10363, 10371, 10407, 10412,
                 10413, 10417},
}
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/FOV_BALANCE_V10_SHORT_POSE_PROTOCOL_20261014.md'
GENERATOR = SOURCE.with_name('arbitrary_plane_fov_balance_v10.py')
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
REHEARSAL = ROOT / 'runs/joint_spatial_305_acquired_rehearsal_001'
PARENT = REHEARSAL / 'intervention/pose_local_00400.pt'
PRIOR = ROOT / 'runs/joint_spatial_feedback_305_acquired_train_001'
PREFLIGHT = ROOT / 'runs/fov_balance_v10_preflight_002'
RUN = ROOT / 'runs/fov_balance_v10_short_pose_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == PARENT.drive.upper() == 'I:'
assert not RUN.exists() and FOV_SCALE == .72
assert hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == V10_SHA
with PARENT.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert parent_sha == '56d10febb1a6e6592adf18b9b4c7fe1fe0beae2a9179d9ddaffb78eefb82e692'
rehearsal = json.loads((REHEARSAL / 'completed.json').read_text(encoding='utf8'))
assert rehearsal['summary']['step_400_gate_pass']
assert rehearsal['output_sha256']['intervention/completed.json'] == hashlib.sha256(
    (REHEARSAL / 'intervention/completed.json').read_bytes()).hexdigest()
arm_receipt = json.loads((REHEARSAL / 'intervention/completed.json').read_text(encoding='utf8'))
assert arm_receipt['output_sha256'][PARENT.name] == parent_sha
preflight = json.loads((PREFLIGHT / 'completed.json').read_text(encoding='utf8'))
assert preflight['summary']['framing_feasibility_gate_pass']
assert preflight['output_sha256']['cases.jsonl'] == hashlib.sha256(
    (PREFLIGHT / 'cases.jsonl').read_bytes()).hexdigest()
assert preflight['output_sha256']['draws.jsonl'] == hashlib.sha256(
    (PREFLIGHT / 'draws.jsonl').read_bytes()).hexdigest()
used_ids = {json.loads(line)['physical_section_id'] for line in
    (PREFLIGHT / 'cases.jsonl').open(encoding='utf8')}
assert len(used_ids) == 64
used_seeds = {json.loads(line)['seed'] for line in
    (PREFLIGHT / 'draws.jsonl').open(encoding='utf8')}
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert tuple(split['train']) == (GUARD, *TRAIN)
assert ARCHIVE.stat().st_size == split['archive_bytes']
prior_completion = json.loads((PRIOR / 'completed.json').read_text(encoding='utf8'))
assert prior_completion['cases_sha256'] == hashlib.sha256(
    (PRIOR / 'cases.jsonl').read_bytes()).hexdigest()
prior_cases = [json.loads(line) for line in (PRIOR / 'cases.jsonl').open(encoding='utf8')]
assert len(prior_cases) == 98

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
v9 = load_streaming_synthetic_v7_64('cuda')
# Share only immutable atlas/maps; v10 owns independent mutable subjects, spans and provenance.
v10 = {**v9, 'subjects': copy.deepcopy(v9['subjects']),
    'nuisance': v9['nuisance'].copy(), 'provenance': copy.deepcopy(v9['provenance'])}
v10['provenance']['virtual_subjects'] = v10['subjects']
for subject in v10['subjects']:
    bounds = np.asarray(subject['bounds_um'])
    middle = bounds.mean(0)
    subject['bounds_um'] = (middle + FOV_SCALE * (bounds - middle)).tolist()
v10['nuisance'][:, 2:4] *= FOV_SCALE
v10['provenance']['fov_balance_v10'] = {
    'code_sha256': V10_SHA, 'fov_scale': FOV_SCALE,
    'geometry': 'scale conservative subject box and TRAIN acquisition span lengths about their original centres',
    'unchanged': 'independent full-sphere normals, through-support-point and box-offset branches, roll, PSF, virtual deformation, appearance, damage and optional brush',
    'scope': 'TRAIN-only appearance/framing experiment; not a measured physical-cut prevalence'}
assert v9['subjects'] is not v10['subjects'] and v9['nuisance'] is not v10['nuisance']
assert 'fov_balance_v10' not in v9['provenance']

records, images, masks, ouvs = [], [], [], []
with ZipFile(ARCHIVE) as archive:
    for animal in (GUARD, *TRAIN):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        assert registration['target-resolution'] == [456, 528, 320]
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
images = torch.from_numpy(np.stack(images)).to('cuda')
masks = torch.from_numpy(np.stack(masks)).to('cuda')
ouvs = torch.stack(ouvs).to('cuda')
normal_targets = F.normalize(torch.cross(ouvs[:, 1], ouvs[:, 2], dim=-1), dim=-1)

datasets = (
    ('coronal', ROOT / 'data/joint_v7_allen_fullcanvas_192_001',
     'records.jsonl', 'images.npy', 'animal_id', 'model_pixel_to_ap_dv_ml_um'),
    ('sagittal', ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'geometry.jsonl', 'model_input.npy', 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
)
old_images, old_by_donor, old_donors, old_readout, dataset_config = {}, {}, {}, [], {}
for domain, directory, manifest, array, donor_key, affine_key in datasets:
    manifest_path = directory / manifest
    if domain == 'coronal':
        with manifest_path.open('rb', buffering=0) as stream:
            lines = [stream.readline() for _ in range(1280)]
    else:
        lines = manifest_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if domain == 'coronal':
        assert len(rows) == 1280 and all(row['training_split'] == 'train'
            and row['array_row_index'] < 1280 for row in rows)
    else:
        assert len(rows) == 653 and all(row['split'] == 'train' for row in rows)
    old_images[domain] = np.load(directory / array, mmap_mode='r')
    donors = sorted({row[donor_key] for row in rows})
    assert len(donors) == (58 if domain == 'coronal' else 40)
    assert HELDOUT[domain] <= set(donors) and len(HELDOUT[domain]) == 20
    selected = {}
    for row in rows:
        affine = np.asarray(row[affine_key], dtype=np.float64)
        offset, scale = (1 / 6, 256) if domain == 'sagittal' else (0., 192)
        origin = affine[:, 2] + offset * (affine[:, 0] + affine[:, 1])
        edges = np.stack((scale * affine[:, 0], scale * affine[:, 1]), -1)
        row['_ouv'] = np.stack((origin, edges[:, 0], edges[:, 1])).astype(np.float32)
        row['_normal'] = np.cross(edges[:, 0], edges[:, 1])
        row['_normal'] /= np.linalg.norm(row['_normal'])
        row['_axis'] = int(np.abs(row['_normal']).argmax())
    for donor in donors:
        cohort = sorted((row for row in rows if row[donor_key] == donor),
                        key=lambda row: (row['section_number'], row['section_id']))
        row = cohort[(len(cohort) - 1) // 2]
        prior = prior_cases[len(old_readout)]
        source_sha = row.get('source_image_sha256', row.get('image_sha256'))
        assert (domain, donor, row['section_id'], row['array_row_index'], source_sha) == (
            prior['domain'], prior['donor_id'], prior['section_id'],
            prior['array_row_index'], prior['source_image_sha256'])
        assert ('AP', 'DV', 'ML')[row['_axis']] == prior['weak_axis']
        row['_heldout'] = donor in HELDOUT[domain]
        selected[donor] = row['section_id']
        old_readout.append((domain, row))
    old_donors[domain] = [donor for donor in donors if donor not in HELDOUT[domain]]
    old_by_donor[domain] = {donor: [row for row in rows if row[donor_key] == donor
        and row['section_id'] != selected[donor]] for donor in old_donors[domain]}
    assert all(old_by_donor[domain][donor] for donor in old_donors[domain])
    dataset_config[domain] = {'train_manifest_sha256': hashlib.sha256(b''.join(lines)).hexdigest(),
        'train_rows': len(rows), 'readout_donors': len(donors),
        'fully_heldout_donors': sorted(HELDOUT[domain]),
        'gradient_donors': len(old_donors[domain]),
        'gradient_rows': sum(map(len, old_by_donor[domain].values()))}
assert len(old_readout) == 98

RUN.mkdir(parents=True, exist_ok=False)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
flags = torch.arange(2, device='cuda')[None, None]
panels = {}
panel_rng = np.random.default_rng(SEED + 1)
with (RUN / 'panel_draws.jsonl').open('w', encoding='utf8') as panel_draws, torch.inference_mode():
    for version, context in (('v9', v9), ('v10', v10)):
        panel = []
        for base in range(56, 64):
            accepted = attempt = 0
            while accepted < 8:
                attempt += 1
                variant = int(panel_rng.integers(VARIANTS))
                seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
                assert seed not in used_seeds
                used_seeds.add(seed)
                sample = (sample_postartifact_gauge_v1(context,
                    [base * VARIANTS + variant], seed, SIDE) if version == 'v9'
                    else sample_fov_balance_v10(context,
                    [base * VARIANTS + variant], seed, SIDE))
                record = sample['provenance'][0]
                physical_id = record['physical_section_id']
                assert physical_id not in used_ids and record['split'] == 'train'
                used_ids.add(physical_id)
                eligible = bool(sample['eligible'][0])
                panel_draws.write(json.dumps({'version': version, 'base': base,
                    'attempt': attempt, 'accepted_slot': accepted if eligible else None,
                    'variant': variant, 'seed': seed, 'accepted': eligible,
                    'provenance': record}, allow_nan=False) + '\n')
                if not eligible:
                    continue
                accepted += 1
                valid = sample['valid_mask'][0]
                sites = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                    generator=torch.Generator(device='cuda').manual_seed(seed))
                chart = torch.stack((sites.remainder(SIDE),
                    sites.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                meta = {'physical_section_id': physical_id, 'version': version,
                    'base': base, 'variant': variant, 'seed': seed,
                    'visible_fraction': float(valid.float().mean()),
                    'obliquity_deg': math.degrees(math.acos(float(normal.abs().max().clamp(0, 1)))),
                    'framing': record['framing']['kind'],
                    'sampling_branch': record['sampling_branch'],
                    'exterior_style': record['one_shot_slide_artifacts_v9']['exterior_style']}
                panel.append({'inputs': sample['inputs'][0].cpu(),
                    'chart': chart.cpu(),
                    'truth': sample['centre'][0].reshape(-1, 3)[sites].cpu(),
                    'state': sample['state'][0].cpu(), 'meta': meta})
        assert len(panel) == 64 and {case['meta']['base'] for case in panel} == set(range(56, 64))
        panels[version] = panel
torch.save(panels, RUN / 'panels.pt')
panel_counts = {version: {
    'sparse_lt15pct': sum(case['meta']['visible_fraction'] < .15 for case in panel),
    'oblique_ge45deg': sum(case['meta']['obliquity_deg'] >= 45 for case in panel),
    'framing_box': sum(case['meta']['framing'] == 'conservative_virtual_subject_box'
                       for case in panel),
    'framing_tuple': sum(case['meta']['framing'] == 'TRAIN_acquisition_tuple'
                         for case in panel),
    'placement_support': sum(case['meta']['sampling_branch'] == 'through_support_point'
                             for case in panel),
    'placement_box_offset': sum(case['meta']['sampling_branch'] == 'uniform_box_offset'
                                for case in panel)} for version, panel in panels.items()}
assert all(counts['framing_box'] + counts['framing_tuple'] == 64
    and counts['placement_support'] + counts['placement_box_offset'] == 64
    for counts in panel_counts.values())
if (panel_counts['v10']['sparse_lt15pct'] < 4
        or panel_counts['v10']['oblique_ge45deg'] < 4):
    decision = {'decision': 'inconclusive_insufficient_v10_tail',
        'panel_counts': panel_counts,
        'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
        'panel_sha256': hashlib.sha256((RUN / 'panels.pt').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    (RUN / 'inconclusive.json').write_text(json.dumps(decision, indent=2), encoding='utf8')
    print(json.dumps(decision), flush=True)
    raise SystemExit(0)

# An acquired schedule and base schedule shared by both arms; synthetic seeds are not shared.
schedule_rng = np.random.default_rng(SEED + 2)
lupori_order = schedule_rng.permutation(train_indices)
lupori_cursor = 0
old_order = {domain: schedule_rng.permutation(donors) for domain, donors in old_donors.items()}
old_cursor = {domain: 0 for domain in old_donors}
schedule = []
with (RUN / 'schedule.jsonl').open('w', encoding='utf8') as schedule_file:
    for step in range(1, UPDATES + 1):
        if lupori_cursor == len(lupori_order):
            lupori_order = schedule_rng.permutation(train_indices)
            lupori_cursor = 0
        lupori_index = int(lupori_order[lupori_cursor])
        lupori_cursor += 1
        domain = 'sagittal' if step % 2 else 'coronal'
        if old_cursor[domain] == len(old_order[domain]):
            old_order[domain] = schedule_rng.permutation(old_donors[domain])
            old_cursor[domain] = 0
        donor = int(old_order[domain][old_cursor[domain]])
        old_cursor[domain] += 1
        cohort = old_by_donor[domain][donor]
        old_row = cohort[int(schedule_rng.integers(len(cohort)))]
        bases = [int(base) for base in schedule_rng.integers(56, size=2)]
        schedule.append((lupori_index, domain, old_row, bases))
        schedule_file.write(json.dumps({'step': step, 'synthetic_bases': bases,
            'lupori': records[lupori_index], 'old': {'domain': domain,
            'donor_id': donor, 'animal_id': old_row.get('animal_id', donor),
            'specimen_id': old_row['specimen_id'],
            'experiment_id': old_row['experiment_id'],
            'section_id': old_row['section_id'],
            'array_row_index': old_row['array_row_index'],
            'source_image_sha256': old_row.get('source_image_sha256',
                old_row.get('image_sha256'))}}, allow_nan=False) + '\n')

config = {'candidate': 305, 'seed': SEED, 'local_optimizer_batches': UPDATES,
    'readouts': STEPS, 'batch': {
        'control': ['v9', 'v9', 'lupori_train', 'alternating_allen_train'],
        'intervention': ['v9', 'v10', 'lupori_train', 'alternating_allen_train']},
    'synthetic_train_bases': list(range(56)),
    'synthetic_inner_bases': list(range(56, 64)),
    'lupori_gradient_animals': TRAIN, 'lupori_guard_animal': GUARD,
    'old_readout_sections_excluded_from_gradients': True,
    'old_heldout_donors_have_no_gradients': True, 'old_weak_loss_weight': .3,
    'optimized_modules': ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global'],
    'optimizer': 'restored rehearsal intervention AdamW moments; constant endpoint LR pose 4e-6, shared 1e-6',
    'parent_checkpoint_sha256': parent_sha,
    'parent_arm_completion_sha256': hashlib.sha256(
        (REHEARSAL / 'intervention/completed.json').read_bytes()).hexdigest(),
    'parent_run_completion_sha256': hashlib.sha256(
        (REHEARSAL / 'completed.json').read_bytes()).hexdigest(),
    'preflight_completion_sha256': hashlib.sha256(
        (PREFLIGHT / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'v10_generator_sha256': V10_SHA,
    'sampler_source_sha256': {name: hashlib.sha256(
        SOURCE.with_name(name).read_bytes()).hexdigest() for name in (
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_one_shot_slide_artifacts_v7.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py')},
    'v9_generator_input_sha256': v9['provenance']['input_sha256'],
    'v10_generator_provenance': v10['provenance']['fov_balance_v10'],
    'panel_sha256': hashlib.sha256((RUN / 'panels.pt').read_bytes()).hexdigest(),
    'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.jsonl').read_bytes()).hexdigest(),
    'panel_counts': panel_counts,
    'prior_cases_sha256': hashlib.sha256((PRIOR / 'cases.jsonl').read_bytes()).hexdigest(),
    'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    'lupori_archive_bytes': split['archive_bytes'],
    'lupori_archive_md5_from_split': split['archive_md5'], 'datasets': dataset_config,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

all_readouts = {}
for arm in ('control', 'intervention'):
    path = RUN / arm
    path.mkdir()
    model, parent_config = load_one_shot_checkpoint(PARENT, device='cuda')
    assert parent_config['candidate'] == 305
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
    assert saved['step'] == 1000 and saved['local_step'] == 400
    assert saved['arm'] == 'intervention'
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    assert [group['base_lr'] for group in optimizer.param_groups] == [2e-5, 5e-6]
    for group in optimizer.param_groups:
        group['lr'] = .2 * group['base_lr']
    parameters = [p for group in groups for p in group['params']]
    torch.manual_seed(SEED + (10 if arm == 'control' else 11))
    torch.cuda.manual_seed_all(SEED + (10 if arm == 'control' else 11))
    synthetic_rng = np.random.default_rng(SEED + (10 if arm == 'control' else 11))
    synthetic_sites = torch.Generator(device='cuda').manual_seed(
        SEED + (20 if arm == 'control' else 21))
    started = time.perf_counter()
    readouts = {}
    model.train()
    with (path / 'draws.jsonl').open('w', encoding='utf8') as draws, \
            (path / 'training.jsonl').open('w', encoding='utf8') as training, \
            (path / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
        for step in range(UPDATES + 1):
            if step:
                index, domain, old_row, bases = schedule[step - 1]
                samples = []
                for slot, base in enumerate(bases):
                    version = 'v10' if arm == 'intervention' and slot == 1 else 'v9'
                    context = v10 if version == 'v10' else v9
                    while True:
                        variant = int(synthetic_rng.integers(VARIANTS))
                        seed = int(synthetic_rng.integers(2**63 - 1, dtype=np.int64))
                        assert seed not in used_seeds
                        used_seeds.add(seed)
                        with torch.no_grad():
                            sample = (sample_fov_balance_v10(context,
                                [base * VARIANTS + variant], seed, SIDE) if version == 'v10'
                                else sample_postartifact_gauge_v1(context,
                                [base * VARIANTS + variant], seed, SIDE))
                        record = sample['provenance'][0]
                        physical_id = record['physical_section_id']
                        assert physical_id not in used_ids and record['split'] == 'train'
                        used_ids.add(physical_id)
                        eligible = bool(sample['eligible'][0])
                        draws.write(json.dumps({'kind': 'synthetic', 'step': step,
                            'slot': slot, 'version': version, 'base': base,
                            'variant': variant, 'seed': seed, 'accepted': eligible,
                            'provenance': record}, allow_nan=False) + '\n')
                        if eligible:
                            samples.append(sample)
                            break
                draws.write(json.dumps({'kind': 'lupori', 'step': step,
                    'slot': 2, **records[index]}, allow_nan=False) + '\n')
                draws.write(json.dumps({'kind': 'old', 'step': step,
                    'slot': 3, 'domain': domain,
                    'donor_id': old_row.get('donor_id', old_row.get('animal_id')),
                    'animal_id': old_row.get('animal_id', old_row.get('donor_id')),
                    'specimen_id': old_row['specimen_id'],
                    'experiment_id': old_row['experiment_id'],
                    'section_id': old_row['section_id'],
                    'array_row_index': old_row['array_row_index'],
                    'source_image_sha256': old_row.get('source_image_sha256',
                        old_row.get('image_sha256'))}, allow_nan=False) + '\n')

                synthetic = torch.cat([sample['inputs'] for sample in samples])
                target_state = torch.cat([sample['state'] for sample in samples])
                target_reflection = torch.cat([sample['reflection'] for sample in samples])
                valid = torch.cat([sample['valid_mask'] for sample in samples])
                centres = torch.cat([sample['centre'] for sample in samples])
                real = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
                real[0, 0] = images[index]
                old_image = torch.from_numpy(np.array(old_images[domain][
                    old_row['array_row_index'], 0], dtype=np.float32, copy=True))[None, None].cuda()
                if domain == 'sagittal':
                    old_image = F.interpolate(old_image, (SIDE, SIDE), mode='area')
                old_input = torch.cat((old_image, torch.zeros(1, 4, SIDE, SIDE,
                                                               device='cuda')), 1)
                prediction = model.predict(torch.cat((synthetic, real, old_input)))
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
                proposed = rigid_points_090(states[:2], reflections[:2],
                                            chart[:, None, None], (SIDE, SIDE))
                five = rigid_points_090(states[:2], reflections[:2], corners, (SIDE, SIDE))
                true_five = rigid_points_090(target_state, target_reflection,
                                             corners, (SIDE, SIDE))
                tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
                normal = full_frame_state_to_components(prediction['state'][:2])[1][..., :, 2]
                true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
                normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
                physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
                target_mass = F.softmax(-physical.detach() / .7, -1)
                nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
                near_mode = model.base_modes + nearest.indices
                near_cost = physical.reshape(2, model.modes, 2).min(-1).values.gather(1, near_mode)
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
                synthetic_loss = (-1.5 * torch.logsumexp(prior[:2] - physical / 1.5, -1)
                    + .5 * F.kl_div(prior[:2], target_mass, reduction='none').sum(-1)
                    + .5 * physical[:, :2 * model.base_modes].min(-1).values
                    + (neighbourhood * near_cost).sum(-1)).mean()

                rpixels = torch.multinomial(masks[index].flatten().float(), 128,
                    replacement=True,
                    generator=torch.Generator(device='cuda').manual_seed(SEED + 1000 + step))
                draws.write(json.dumps({'kind': 'site_indices', 'step': step,
                    'synthetic_sha256': hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest(),
                    'lupori_sha256': hashlib.sha256(rpixels.cpu().numpy().tobytes()).hexdigest()}) + '\n')
                rchart = torch.stack((rpixels.remainder(SIDE),
                    rpixels.div(SIDE, rounding_mode='floor')), -1).float()[None] / SIDE
                reference = ouvs[index, 0] + rchart[..., :1] * ouvs[index, 1] + \
                    rchart[..., 1:] * ouvs[index, 2]
                rproposed = rigid_points_090(states[2:3], reflections[2:3],
                                              rchart[:, None, None], (SIDE, SIDE))
                reference_five = ouvs[index, 0] + corners[:, :1] * ouvs[index, 1] + \
                    corners[:, 1:] * ouvs[index, 2]
                rfive = rigid_points_090(states[2:3], reflections[2:3], corners, (SIDE, SIDE))
                tissue_um = (rproposed - reference[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (rfive - reference_five).norm(dim=-1).mean(-1)
                rnormal = full_frame_state_to_components(prediction['state'][2:3])[1][..., :, 2]
                normal_um = 4000 * (1 - (rnormal[0] * normal_targets[index]).sum(-1
                    ).abs().clamp_max(1))
                real_cost = (.75 * tissue_um + .25 * five_um + normal_um[None, :, None]
                             ).flatten(1) / 1000
                real_mass = F.softmax(-real_cost.detach() / .7, -1)
                nearest = (normal_targets[index] @ model.normal_anchor_frames[:, :, 2].T
                           ).abs().topk(4)
                near_mode = model.base_modes + nearest.indices
                near_cost = real_cost.reshape(model.modes, 2).min(-1).values[near_mode]
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:1]), -1)
                real_loss = (-1.5 * torch.logsumexp(prior[2] - real_cost[0] / 1.5, -1)
                    + .5 * F.kl_div(prior[2], real_mass[0], reduction='sum')
                    + .5 * real_cost[0, :2 * model.base_modes].min()
                    + (neighbourhood * near_cost).sum())

                old_ouv = torch.as_tensor(old_row['_ouv'], device='cuda')
                old_reference = old_ouv[0] + corners[:, :1] * old_ouv[1] + \
                    corners[:, 1:] * old_ouv[2]
                old_five = rigid_points_090(states[3:4], reflections[3:4], corners,
                                             (SIDE, SIDE))
                old_five_um = (old_five - old_reference).norm(dim=-1).mean(-1)
                old_normal = full_frame_state_to_components(prediction['state'][3:4]
                    )[1][..., :, 2][0]
                old_true_normal = F.normalize(torch.cross(old_ouv[1], old_ouv[2],
                                                          dim=-1), dim=-1)
                old_normal_um = 4000 * (1 - (old_normal * old_true_normal).sum(-1
                    ).abs().clamp_max(1))
                old_cost = (old_five_um + old_normal_um[None, :, None]).flatten(1) / 1000
                old_mass = F.softmax(-old_cost.detach() / .7, -1)
                nearest = (old_true_normal @ model.normal_anchor_frames[:, :, 2].T
                           ).abs().topk(4)
                near_mode = model.base_modes + nearest.indices
                near_cost = old_cost.reshape(model.modes, 2).min(-1).values[near_mode]
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:1]), -1)
                old_loss = (-1.5 * torch.logsumexp(prior[3] - old_cost[0] / 1.5, -1)
                    + .5 * F.kl_div(prior[3], old_mass[0], reduction='sum')
                    + .5 * old_cost[0, :2 * model.base_modes].min()
                    + (neighbourhood * near_cost).sum())
                loss = synthetic_loss + real_loss + .3 * old_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                training.write(json.dumps({'step': step, 'synthetic_presentations': 2 * step,
                    'v10_presentations': step if arm == 'intervention' else 0,
                    'lupori_presentations': step, 'old_presentations': step,
                    'old_domain': domain, 'synthetic_loss': float(synthetic_loss.detach()),
                    'lupori_weak_loss': float(real_loss.detach()),
                    'old_weak_loss': float(old_loss.detach()), 'loss': float(loss.detach()),
                    'gradient_norm': float(gradient),
                    'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
                if step in STEPS:
                    training.flush()
                    draws.flush()
                    print(json.dumps({'event': 'train_milestone', 'arm': arm,
                        'step': step, 'loss': float(loss.detach())}), flush=True)

            if step in STEPS:
                model.eval()
                rows = []
                with torch.inference_mode():
                    for version, panel in panels.items():
                        for start in range(0, len(panel), 4):
                            group = panel[start:start + 4]
                            batch = torch.stack([case['inputs'] for case in group]).cuda()
                            chart = torch.stack([case['chart'] for case in group]).cuda()
                            reference = torch.stack([case['truth'] for case in group]).cuda()
                            true_states = torch.stack([case['state'] for case in group]).cuda()
                            true_normals = full_frame_state_to_components(true_states)[1][..., :, 2]
                            p = model.predict(batch)
                            s = p['state'][:, :, None].expand(-1, -1, 2, -1)
                            r = flags.expand(len(s), model.modes, 2)
                            mass = (p['log_mass'][..., None] + torch.stack((
                                F.logsigmoid(-p['reflection_logit']),
                                F.logsigmoid(p['reflection_logit'])), -1)).flatten(1)
                            mapped = rigid_points_090(s, r, chart[:, None, None], (SIDE, SIDE))
                            errors = (mapped - reference[:, None, None]).norm(dim=-1
                                ).mean(-1).flatten(1) / 1000
                            first = mass.argmax(-1)
                            top16 = mass.topk(16, -1).indices
                            pred_normals = full_frame_state_to_components(p['state'])[1][..., :, 2]
                            angle = torch.rad2deg(torch.acos((pred_normals * true_normals[:, None]
                                ).sum(-1).abs().clamp_max(1)))
                            for j, case in enumerate(group):
                                rows.append({'cohort': version, **case['meta'],
                                    'first_rigid_site_gap_mm': float(errors[j, first[j]]),
                                    'top16_oracle_gap_mm': float(errors[j, top16[j]].min()),
                                    'first_unsigned_normal_gap_deg': float(angle[j, first[j] // 2])})

                    for start in range(0, len(guard_indices), 4):
                        ids = guard_indices[start:start + 4]
                        batch = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
                        batch[:, 0] = images[ids]
                        chart = corners[None].expand(len(ids), -1, -1)
                        reference = ouvs[ids, 0, None] + chart[..., :1] * ouvs[ids, 1, None] \
                            + chart[..., 1:] * ouvs[ids, 2, None]
                        p = model.predict(batch)
                        s = p['state'][:, :, None].expand(-1, -1, 2, -1)
                        r = flags.expand(len(s), model.modes, 2)
                        mass = (p['log_mass'][..., None] + torch.stack((
                            F.logsigmoid(-p['reflection_logit']),
                            F.logsigmoid(p['reflection_logit'])), -1)).flatten(1)
                        mapped = rigid_points_090(s, r, chart[:, None, None], (SIDE, SIDE))
                        errors = (mapped - reference[:, None, None]).norm(dim=-1
                            ).mean(-1).flatten(1) / 1000
                        first = mass.argmax(-1)
                        top16 = mass.topk(16, -1).indices
                        pred_normals = full_frame_state_to_components(p['state'])[1][..., :, 2]
                        angle = torch.rad2deg(torch.acos((pred_normals * normal_targets[ids, None]
                            ).sum(-1).abs().clamp_max(1)))
                        for j, index in enumerate(ids):
                            rows.append({'cohort': 'guard', **records[index],
                                'first_five_site_gap_mm': float(errors[j, first[j]]),
                                'top16_oracle_gap_mm': float(errors[j, top16[j]].min()),
                                'first_unsigned_normal_gap_deg': float(angle[j, first[j] // 2])})

                    for domain, old_row in old_readout:
                        image = torch.from_numpy(np.array(old_images[domain][
                            old_row['array_row_index'], 0], dtype=np.float32,
                            copy=True))[None, None].cuda()
                        if domain == 'sagittal':
                            image = F.interpolate(image, (SIDE, SIDE), mode='area')
                        batch = torch.cat((image, torch.zeros(1, 4, SIDE, SIDE,
                                                              device='cuda')), 1)
                        p = model.predict(batch)
                        mass = (p['log_mass'][..., None] + torch.stack((
                            F.logsigmoid(-p['reflection_logit']),
                            F.logsigmoid(p['reflection_logit'])), -1)).flatten()
                        first = int(mass.argmax())
                        top16 = mass.topk(16).indices
                        centres, frames, _ = full_frame_state_to_components(p['state'][0])
                        normals = frames[:, :, 2]
                        mode = first // 2
                        weak_normal = torch.as_tensor(old_row['_normal'], device='cuda',
                                                      dtype=normals.dtype)
                        angle = torch.rad2deg(torch.acos((normals[mode] * weak_normal
                            ).sum().abs().clamp(0, 1)))
                        weak_ouv = old_row['_ouv']
                        weak_centre = weak_ouv[0] + .5 * (weak_ouv[1] + weak_ouv[2])
                        weak_axis = old_row['_axis']
                        rows.append({'cohort': 'old_' + domain,
                            'fully_heldout': old_row['_heldout'], 'domain': domain,
                            'donor_id': old_row.get('donor_id', old_row.get('animal_id')),
                            'animal_id': old_row.get('animal_id', old_row.get('donor_id')),
                            'specimen_id': old_row['specimen_id'],
                            'experiment_id': old_row['experiment_id'],
                            'section_id': old_row['section_id'],
                            'array_row_index': old_row['array_row_index'],
                            'source_image_sha256': old_row.get('source_image_sha256',
                                old_row.get('image_sha256')),
                            'weak_axis': ('AP', 'DV', 'ML')[weak_axis],
                            'first_axis_correct': int(normals[mode].abs().argmax()) == weak_axis,
                            'top16_contains_weak_axis': any(
                                int(normals[int(branch // 2)].abs().argmax()) == weak_axis
                                for branch in top16),
                            'first_weak_normal_angle_deg': float(angle),
                            'first_weak_centre_gap_mm': float(np.linalg.norm(
                                centres[mode].cpu().numpy() - weak_centre) / 1000)})

                metrics = {}
                for version in ('v9', 'v10'):
                    group = [row for row in rows if row['cohort'] == version]
                    metrics[version] = {
                        'n': len(group),
                        'mean_first_gap_mm': float(np.mean([
                            row['first_rigid_site_gap_mm'] for row in group])),
                        'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5
                                                   for row in group),
                        'per_base_mean_first_gap_mm': {str(base): float(np.mean([
                            row['first_rigid_site_gap_mm'] for row in group
                            if row['base'] == base])) for base in range(56, 64)},
                    }
                    strata = {
                        'sparse_lt15pct': [row for row in group if row['visible_fraction'] < .15],
                        'visibility_15to35pct': [row for row in group
                            if .15 <= row['visible_fraction'] < .35],
                        'visibility_35to65pct': [row for row in group
                            if .35 <= row['visible_fraction'] < .65],
                        'visibility_ge65pct': [row for row in group
                            if row['visible_fraction'] >= .65],
                        'oblique_ge45deg': [row for row in group if row['obliquity_deg'] >= 45],
                        'obliquity_lt15deg': [row for row in group
                            if row['obliquity_deg'] < 15],
                        'obliquity_15to30deg': [row for row in group
                            if 15 <= row['obliquity_deg'] < 30],
                        'obliquity_30to45deg': [row for row in group
                            if 30 <= row['obliquity_deg'] < 45],
                        'framing_box': [row for row in group
                            if row['framing'] == 'conservative_virtual_subject_box'],
                        'framing_tuple': [row for row in group
                            if row['framing'] == 'TRAIN_acquisition_tuple'],
                        'placement_support': [row for row in group
                            if row['sampling_branch'] == 'through_support_point'],
                        'placement_box_offset': [row for row in group
                            if row['sampling_branch'] == 'uniform_box_offset'],
                    }
                    metrics[version]['strata'] = {name: {'n': len(items),
                        'mean_first_gap_mm': (float(np.mean([
                            row['first_rigid_site_gap_mm'] for row in items])) if items else None),
                        'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5
                                                   for row in items)}
                        for name, items in strata.items()}
                group = [row for row in rows if row['cohort'] == 'guard']
                metrics['guard'] = {'n': len(group),
                    'mean_first_gap_mm': float(np.mean([
                        row['first_five_site_gap_mm'] for row in group])),
                    'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5
                                               for row in group),
                    'median_first_unsigned_normal_gap_deg': float(np.median([
                        row['first_unsigned_normal_gap_deg'] for row in group]))}
                for domain in ('coronal', 'sagittal'):
                    for panel_name in ('heldout', 'all'):
                        group = [row for row in rows if row['cohort'] == 'old_' + domain
                            and (panel_name == 'all' or row['fully_heldout'])]
                        metrics[f'old_{domain}_{panel_name}'] = {'donors': len(group),
                            'first_axis_correct': sum(row['first_axis_correct'] for row in group),
                            'top16_contains_weak_axis': sum(row['top16_contains_weak_axis']
                                                            for row in group),
                            'median_weak_normal_angle_deg': float(np.median([
                                row['first_weak_normal_angle_deg'] for row in group])),
                            'median_weak_centre_gap_mm': float(np.median([
                                row['first_weak_centre_gap_mm'] for row in group]))}
                readouts[str(step)] = metrics
                readout_file.write(json.dumps({'step': step, 'metrics': metrics,
                    'rows': rows}, allow_nan=False) + '\n')
                readout_file.flush()
                torch.save({'step': 1000 + step, 'local_step': step,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': config, 'arm': arm, 'calibrated': False},
                    path / f'pose_local_{step:05d}.pt')
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                    'metrics': metrics}), flush=True)
                model.train()

    all_readouts[arm] = readouts
    (path / 'completed.json').write_text(json.dumps({'arm': arm, 'updates': UPDATES,
        'readouts': readouts, 'output_sha256': {file.name: hashlib.sha256(
            file.read_bytes()).hexdigest() for file in path.iterdir()
            if file.name != 'completed.json'}, 'calibrated': False}, indent=2), encoding='utf8')
    del model, optimizer
    torch.cuda.empty_cache()

assert all_readouts['control']['0'] == all_readouts['intervention']['0']
control = all_readouts['control']['400']
intervention = all_readouts['intervention']['400']
parent = all_readouts['control']['0']
panel_sufficient = (panel_counts['v10']['sparse_lt15pct'] >= 4
    and panel_counts['v10']['oblique_ge45deg'] >= 4)
base_wins = sum(intervention['v10']['per_base_mean_first_gap_mm'][str(base)] <
    control['v10']['per_base_mean_first_gap_mm'][str(base)] for base in range(56, 64))
clauses = {
    'v10_mean_improvement_ge0p20mm': intervention['v10']['mean_first_gap_mm'] <=
        control['v10']['mean_first_gap_mm'] - .20,
    'v10_base_wins_ge6': base_wins >= 6,
    'v10_top16_loss_le4': intervention['v10']['top16_capture_1p5mm'] >=
        control['v10']['top16_capture_1p5mm'] - 4,
    'v9_mean_regression_le0p10mm': intervention['v9']['mean_first_gap_mm'] <=
        control['v9']['mean_first_gap_mm'] + .10,
    'v9_top16_loss_le4': intervention['v9']['top16_capture_1p5mm'] >=
        control['v9']['top16_capture_1p5mm'] - 4,
    'al1a_parent_regression_le0p20mm': intervention['guard']['mean_first_gap_mm'] <=
        parent['guard']['mean_first_gap_mm'] + .20,
    'al1a_control_regression_le0p20mm': intervention['guard']['mean_first_gap_mm'] <=
        control['guard']['mean_first_gap_mm'] + .20,
    'sagittal_heldout_ge15': intervention['old_sagittal_heldout']['first_axis_correct'] >= 15,
    'sagittal_heldout_control_loss_le1': intervention['old_sagittal_heldout'][
        'first_axis_correct'] >= control['old_sagittal_heldout']['first_axis_correct'] - 1,
    'coronal_heldout_ge19': intervention['old_coronal_heldout']['first_axis_correct'] >= 19,
    'coronal_heldout_control_loss_le1': intervention['old_coronal_heldout'][
        'first_axis_correct'] >= control['old_coronal_heldout']['first_axis_correct'] - 1,
}
gate = panel_sufficient and all(clauses.values())
summary = {'decision': ('inconclusive_insufficient_v10_tail' if not panel_sufficient
    else 'pass' if gate else 'fail'), 'step_400_gate_pass': gate,
    'panel_sufficient': panel_sufficient, 'panel_counts': panel_counts,
    'v10_base_wins': base_wins, 'gate_clauses': clauses,
    'readouts': all_readouts,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(RUN / 'completed.json').write_text(json.dumps({'summary': summary,
    'output_sha256': {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in ('config.json', 'panel_draws.jsonl', 'panels.pt', 'schedule.jsonl',
                     'summary.json', 'control/completed.json',
                     'intervention/completed.json')}}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'decision': summary['decision'],
    'step_400_gate_pass': gate}), flush=True)
