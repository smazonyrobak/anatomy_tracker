"""Render one native-256 observation for each new 061 synthetic DEV plane."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / 'cache/triton')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_coherent_subject_v6 import make_coherent_subject_section_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_subject_deformation_v2 import subject_deformation_plan_receipt_v2
from training.arbitrary_plane_subject_sampling_v6 import sample_subject_planes
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

PLANS = ROOT / 'data/pose_feedback_061_fresh_synthetic_dev_plans_001'
TRAIN_MAPS = ROOT / 'data/joint_v7_independent_subject_maps_001'
PRIOR_PANELS = (ROOT / 'data/one_shot_native256_synthetic_dev_001',
                ROOT / 'data/one_shot_fresh_synthetic_dev_panel_001',
                ROOT / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001')
OUT = ROOT / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
SEED, SIDE, PLANES = 2026100612, 256, 32
AUGMENTATION_SEED_PREFIX = 2026100613
MODES = ('raw', 'exact_black', 'imperfect_brush')
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


plan_index = json.loads((PLANS / 'completed.json').read_text())
plans = sorted(plan_index['subjects'], key=lambda row: row['animal_index'])
train_maps = json.loads((TRAIN_MAPS / 'completed.json').read_text())['new_train_subjects']
prior = [[json.loads(line) for line in (directory / 'records.jsonl').open()]
         for directory in PRIOR_PANELS]
assert len(plans) == 8 and all(row['split'] == 'development' for row in plans)
assert len({row['animal_id'] for row in plans}) == len(plans)
assert not {row['subject_plan_receipt_sha256'] for row in plans} & {
    row['plan_receipt_sha256'] for panel in prior for row in panel}
assert not {row['animal_id'] for row in plans} & {
    row['animal_id'] for panel in prior for row in panel}
for key in ('animal_id', 'synthetic_animal_id', 'subject_deformation_plan_id',
            'subject_deformation_realization_id', 'subject_plan_receipt_sha256'):
    assert not {row[key] for row in plans} & {row[key] for row in train_maps}
source_names = ('prepare_pose_feedback_061_fresh_dev_panel.py',
                'arbitrary_plane_subject_sampling_v6.py',
                'arbitrary_plane_one_shot_stream.py',
                'arbitrary_plane_coherent_subject_v6.py',
                'arbitrary_plane_subject_torch_v6.py',
                'arbitrary_plane_subject_deformation_v2.py',
                'arbitrary_plane_full_frame_primitives.py',
                'arbitrary_plane_geometry.py',
                'arbitrary_plane_allen_atlas_binding_v6.py',
                'arbitrary_plane_subject_section_v2.py',
                'arbitrary_plane_streaming_synthetic_v7.py',
                'subject_deformed_slab_multiresolution_bundle_v2.py')
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'source').mkdir()
source_hashes = {}
for name in source_names:
    source = repository / 'training' / name
    shutil.copyfile(source, OUT / 'source' / name)
    source_hashes[name] = sha(OUT / 'source' / name)
protocol = {
    'seed': SEED, 'augmentation_seed_prefix': AUGMENTATION_SEED_PREFIX,
    'side': SIDE, 'modes': MODES,
    'synthetic_subjects': len(plans), 'planes_per_subject': PLANES,
    'physical_sections': len(plans) * PLANES,
    'selection': 'first independently random tissue-intersecting plane per section; empty draws retried, marginal draws retained; one appearance per plane; no pairing or deliberate deduplication',
    'geometry': 'uniform antipodal RP2 normal and roll; random box-slab offset, finite 25-100um thickness; nonzero native256 tissue support; x/W,y/H physical raster',
    'sampling': 'native256 coherent subject/atlas render, then current one-shot observed-coordinate warp/artifact generator',
    'source_plan_completed_sha256': sha(PLANS / 'completed.json'),
    'train_maps_completed_sha256': sha(TRAIN_MAPS / 'completed.json'),
    'prior_panel_records_sha256': {directory.name: sha(directory / 'records.jsonl')
                                    for directory in PRIOR_PANELS},
    'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                               text=True).strip(),
    'source_sha256': source_hashes,
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'numpy': np.__version__, 'torch': str(torch.__version__),
    'eligibility': 'frozen before checkpoint evaluation: source visible support mass and observed valid pixels each >= side^2/144; retain and report ineligible sections',
    'scope': 'new identity-disjoint synthetic deformation plans from one Allen atlas; not independent biological animals, expert truth or public benchmark',
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
del atlas_array, annotation
pixel_y, pixel_x = np.meshgrid(np.arange(SIDE, dtype=np.float64),
                                np.arange(SIDE, dtype=np.float64), indexing='ij')
identity_yx = np.stack((pixel_y, pixel_x))
gy, gx = torch.meshgrid(torch.linspace(-1, 1, SIDE, device='cuda'),
                        torch.linspace(-1, 1, SIDE, device='cuda'), indexing='ij')
integer_masses = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float64)
weights = integer_masses / integer_masses.sum()
tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
               'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um',
               'global_scale', 'frozen_center_um')
records = []
for plan_row in plans:
    directory = PLANS / plan_row['directory']
    for name, digest in plan_row['artifact_sha256'].items():
        assert sha(directory / name) == digest
    plan = _read_raw_artifact(directory, plan_row['plan_files'])
    assert subject_deformation_plan_receipt_v2(plan)['receipt_sha256'] == plan_row['subject_plan_receipt_sha256']
    parameters = [torch.as_tensor(plan['state'][key], device='cuda', dtype=torch.float64)
                  for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])

    def exact_map(points):
        query = torch.from_numpy(np.ascontiguousarray(points)).to(device='cuda', dtype=torch.float64)
        return map_accepted_subject_points_torch_v6(
            query, *parameters, inverse=True, steps=steps, batch_size=8192).cpu().numpy()

    for index in range(PLANES):
        lineage = {key: plan_row[key] for key in
                   ('animal_id', 'subject_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id')}
        lineage['split'] = 'development'
        lineage['section_id'] = f"{plan_row['subject_id']}-section-{index:08d}"
        attempt = 0
        while True:
            geometry_seed = [SEED, plan_row['animal_index'], index, attempt]
            rng = np.random.default_rng(np.random.SeedSequence(geometry_seed))
            thickness = float(rng.uniform(25, 100))
            offsets = np.linspace(-.5, .5, 9) * thickness
            draw = sample_subject_planes(plan, rng, 1, (SIDE, SIDE), offsets)
            ouv = draw['physical_ouv_ap_dv_ml_um'][0]
            reflection = bool(rng.integers(2))
            section = make_coherent_subject_section_v6(
                plan, ouv, identity_yx, (reflection, False), offsets, weights, lineage,
                {'source_plan_receipt_sha256': plan_row['subject_plan_receipt_sha256'],
                 'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6},
                atlas, allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
                subject_to_ccf_mapper=exact_map)
            rendered = section['raw_rendered_channels'].to('cuda')
            clean, support = rendered[0], rendered[1].clamp(0, 1)
            if support.any().item():
                break
            attempt += 1
        appearance_seed = [SEED, plan_row['animal_index'], index, 1]
        appearance_rng = np.random.default_rng(np.random.SeedSequence(appearance_seed))
        mode = MODES[int(appearance_rng.integers(3))]
        appearance = {'gain': float(appearance_rng.uniform(.6, 1.4)),
            'gamma': float(np.exp(appearance_rng.uniform(np.log(.6), np.log(1.6)))),
            'invert': bool(appearance_rng.integers(2)),
            'noise_std': float(appearance_rng.uniform(.005, .05)),
            'background_mean': float(appearance_rng.uniform(0, .8)),
            'background_slope_yx': appearance_rng.uniform(-.15, .15, 2).tolist(),
            'illumination': float(appearance_rng.uniform(0, .15)),
            'texture': float(appearance_rng.uniform(0, .06)),
            'mask_dilate': bool(appearance_rng.integers(2)),
            'mask_radius': int(appearance_rng.integers(1, max(2, SIDE // 32 + 1))),
            'damage': bool(appearance_rng.random() < .2),
            'ellipse': [*appearance_rng.uniform(-.7, .7, 2),
                        *appearance_rng.uniform(.05, .25, 2)]}
        noise_seed_sequence = [SEED, plan_row['animal_index'], index, 2]
        noise_seed = int(np.random.SeedSequence(noise_seed_sequence).generate_state(1, dtype=np.uint64)[0])
        noise_rng = torch.Generator().manual_seed(noise_seed)
        illumination = F.interpolate(torch.randn(1, 1, 5, 5, generator=noise_rng),
                                     (SIDE, SIDE), mode='bilinear', align_corners=False)[0, 0].cuda()
        texture = F.interpolate(torch.randn(1, 1, 16, 16, generator=noise_rng),
                                (SIDE, SIDE), mode='bilinear', align_corners=False)[0, 0].cuda()
        noise = torch.randn(SIDE, SIDE, generator=noise_rng).cuda()
        tissue = (clean / support.clamp_min(1e-6)).clamp(0, 1).pow(appearance['gamma'])
        if appearance['invert']:
            tissue = 1 - tissue
        tissue = (tissue * appearance['gain'] *
                  (1 + appearance['illumination'] * illumination)).clamp(0, 1)
        cy, cx, ry, rx = appearance['ellipse']
        retained = (((gy - cy) / ry).square() + ((gx - cx) / rx).square() >= 1
                    if appearance['damage'] else torch.ones_like(support, dtype=torch.bool))
        visible = support * retained
        background = (appearance['background_mean'] + appearance['background_slope_yx'][0] * gy
                      + appearance['background_slope_yx'][1] * gx + appearance['texture'] * texture)
        before = (tissue * visible + background * (1 - visible)
                  + appearance['noise_std'] * noise).clamp(0, 1)
        brush = (support > 0) & retained
        if mode == 'raw':
            brush = torch.ones_like(brush)
        elif mode == 'imperfect_brush':
            radius = appearance['mask_radius']
            value = brush[None, None].float()
            brush = (F.max_pool2d(value, 2 * radius + 1, 1, radius)[0, 0] > 0
                     if appearance['mask_dilate'] else
                     -F.max_pool2d(-F.pad(value, (radius,) * 4),
                                   2 * radius + 1, 1)[0, 0] > 0)
        eroded = brush.clone()
        eroded[1:] &= brush[:-1]
        eroded[:-1] &= brush[1:]
        eroded[:, 1:] &= brush[:, :-1]
        eroded[:, :-1] &= brush[:, 1:]
        eroded[[0, -1], :] = False
        eroded[:, [0, -1]] = False
        available = mode != 'raw'
        inputs = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
        inputs[0, 0] = before * brush
        inputs[0, 1] = ((brush & ~eroded) & available).float()
        inputs[0, 2] = float(available)
        visible = visible * brush
        source_state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(
            section['canonical_anatomy_plane_fit']['arrays']['physical_ouv_ap_dv_ml_um_float64']))).float()
        physical_id = f"pose-feedback-061-{lineage['section_id']}"
        source = {'inputs': inputs, 'state': source_state[None].cuda(),
                  'reflection': torch.tensor([reflection], device='cuda', dtype=torch.long),
                  'offsets': torch.as_tensor(offsets[None], device='cuda', dtype=torch.float32),
                  'weights': torch.as_tensor(weights[None], device='cuda', dtype=torch.float32),
                  'centre': torch.as_tensor(section['target_centre_ccf_coordinates_ap_dv_ml_um_float64'][None],
                                            device='cuda', dtype=torch.float32),
                  'visible': visible[None], 'support': support[None], 'masks': brush[None],
                  'eligible': (visible.sum() >= SIDE * SIDE / 144)[None],
                  'provenance': [{'physical_section_id': physical_id, 'base_lineage': lineage,
                      'split': 'development', 'mode': mode, 'appearance': appearance,
                      'noise_seed_uint64': noise_seed,
                      'source_plan_receipt_sha256': plan_row['subject_plan_receipt_sha256']}]}
        augmentation_seed = AUGMENTATION_SEED_PREFIX * 10000000 + plan_row['animal_index'] * 1000 + index
        sample = sample_one_shot_stream(None, [plan_row['animal_index']], augmentation_seed,
                                        side=SIDE, source_sample=source)
        filename = f"subject_{plan_row['animal_index']:04d}_section_{index:02d}.npz"
        np.savez_compressed(OUT / filename, inputs=sample['inputs'][0].cpu().numpy(),
            target_state=sample['state'][0].cpu().numpy(),
            target_centre_um=sample['centre'][0].cpu().numpy(),
            valid_mask=sample['valid_mask'][0].cpu().numpy(),
            reflection=sample['reflection'][0].cpu().numpy(),
            offsets_um=sample['offsets'][0].cpu().numpy(),
            weights=sample['weights'][0].cpu().numpy(),
            eligible=sample['eligible'][0].cpu().numpy(),
            subject_ouv_ap_dv_ml_um=ouv)
        record = {'file': filename, 'sha256': sha(OUT / filename),
                  **{key: lineage[key] for key in ('animal_id', 'subject_id', 'specimen_id',
                                                 'experiment_id', 'section_id')},
                  'synthetic_animal_id': plan_row['synthetic_animal_id'],
                  'synthetic_subject_plan_id': plan_row['subject_deformation_plan_id'],
                  'synthetic_subject_realization_id': plan_row['subject_deformation_realization_id'],
                  'panel_physical_section_id': physical_id,
                  'plan_receipt_sha256': plan_row['subject_plan_receipt_sha256'],
                  'section_index': index, 'geometry_seed_sequence': geometry_seed,
                  'empty_geometry_draws_before_intersection': attempt,
                  'appearance_seed_sequence': appearance_seed,
                  'noise_seed_sequence': noise_seed_sequence,
                  'subject_ouv_sha256': hashlib.sha256(np.ascontiguousarray(ouv).tobytes()).hexdigest(),
                  'plane_normal_ap_dv_ml': draw['normal_ap_dv_ml'][0].tolist(),
                  'plane_roll_rad': float(draw['roll_rad'][0]),
                  'signed_offset_from_box_centre_um': float(draw['signed_offset_from_box_centre_um'][0]),
                  'thickness_um': thickness, 'horizontal_reflection': reflection,
                  'appearance_mode': mode, 'augmentation_seed': augmentation_seed,
                  'eligible': bool(sample['eligible'][0]),
                  'valid_pixels': int(sample['valid_mask'][0].sum()),
                  'provenance': sample['provenance'][0]}
        records.append(record)
        print(json.dumps({'panel_section': len(records), 'synthetic_subject_id':
                          plan_row['subject_id'], 'eligible': record['eligible']}), flush=True)
    del parameters, plan
assert len(records) == len(plans) * PLANES
assert len({row['panel_physical_section_id'] for row in records}) == len(records)
assert len({row['subject_ouv_sha256'] for row in records}) == len(records)
assert not {row['section_id'] for row in records} & {
    row['section_id'] for panel in prior for row in panel}
for name in source_names:
    assert sha(repository / 'training' / name) == source_hashes[name]
(OUT / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
completed = {'physical_sections': len(records), 'synthetic_subjects': len(plans),
    'eligible': sum(row['eligible'] for row in records),
    'ineligible': sum(not row['eligible'] for row in records),
    'modes': {mode: sum(row['appearance_mode'] == mode for row in records)
              for mode in MODES},
    'plan_completed_sha256': protocol['source_plan_completed_sha256'],
    'protocol_sha256': sha(OUT / 'protocol.json'),
    'records_sha256': sha(OUT / 'records.jsonl'),
    'scope': protocol['scope']}
(OUT / 'completed.json').write_text(json.dumps(completed, indent=2))
print(json.dumps({'event': 'panel_frozen', **completed}), flush=True)
