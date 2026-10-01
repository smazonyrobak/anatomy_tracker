"""One native-256 appearance per frozen DEVELOPMENT plane, then the unchanged one-shot augmentation."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
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
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

SIDE, SEED = 256, 2026100601
MODES = ('raw', 'exact_black', 'imperfect_brush')
OLD = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
PLANS = ROOT / 'data/joint_v6_coherent_subject_plans_002'
OUT = ROOT / 'data/one_shot_native256_synthetic_dev_001'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


cohort = json.loads((OLD / 'completed.json').read_text())
plan_index = json.loads((PLANS / 'completed.json').read_text())
plans = sorted((row for row in plan_index['subjects'] if row['split'] == 'development'),
               key=lambda row: row['animal_index'])
sections = sorted((row for row in cohort['sections'] if row['lineage']['split'] == 'development'),
                  key=lambda row: (row['lineage']['animal_id'], row['section_index']))
assert len(plans) == 4 and len(sections) == 128
assert {row['animal_id'] for row in plans} == {row['lineage']['animal_id'] for row in sections}
assert all(sum(row['lineage']['animal_id'] == plan['animal_id'] for row in sections) == 32
           for plan in plans)
assert not {row['animal_id'] for row in plans} & {
    row['animal_id'] for row in plan_index['subjects'] if row['split'] == 'train'}
assert len({row['lineage']['section_id'] for row in sections}) == 128
source_names = ('prepare_one_shot_native256_synthetic_dev.py', 'arbitrary_plane_one_shot_stream.py',
                'arbitrary_plane_coherent_subject_v6.py', 'arbitrary_plane_subject_torch_v6.py',
                'arbitrary_plane_subject_deformation_v2.py', 'arbitrary_plane_full_frame_primitives.py',
                'arbitrary_plane_geometry.py', 'arbitrary_plane_allen_atlas_binding_v6.py',
                'arbitrary_plane_subject_section_v2.py', 'arbitrary_plane_streaming_synthetic_v7.py',
                'subject_deformed_slab_multiresolution_bundle_v2.py')
protocol = {'seed': SEED, 'side': SIDE, 'modes': MODES, 'physical_sections': 128,
            'selection': 'all 32 frozen preselected planes in each of four DEVELOPMENT deformation plans; one seeded appearance per physical plane, no same-plane mode copies',
            'sampling': 'native256 subject/atlas rerender before current sample_one_shot_stream augmentation, with exact same local warp/tear/missing/fold/bubble/seam and observed-coordinate target refit as TRAIN',
            'source_plan_sha256': sha(PLANS / 'completed.json'), 'source_cohort_sha256': sha(OLD / 'completed.json'),
            'source_sha256': {name: sha(Path(__file__).parent / name) for name in source_names},
            'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
            'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
            'numpy': np.__version__, 'torch': str(torch.__version__),
            'scope': 'held-out synthetic deformation plans from one Allen atlas; not independent biological animals or expert truth'}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
del atlas_array, annotation
pixel_y, pixel_x = np.meshgrid(np.arange(SIDE, dtype=np.float64),
                                np.arange(SIDE, dtype=np.float64), indexing='ij')
identity_yx = np.stack((pixel_y, pixel_x))
gy, gx = torch.meshgrid(torch.linspace(-1, 1, SIDE, device='cuda'),
                        torch.linspace(-1, 1, SIDE, device='cuda'), indexing='ij')
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

    for old in (row for row in sections if row['lineage']['animal_id'] == plan_row['animal_id']):
        for name, digest in old['artifact_sha256'].items():
            assert sha(OLD / name) == digest
        metadata = json.loads((OLD / old['artifacts']['metadata']).read_text())
        with np.load(OLD / old['artifacts']['arrays'], allow_pickle=False) as arrays:
            ouv = arrays[metadata['subject_ouv_ap_dv_ml_um_float64']['__ndarray__']].copy()
            offsets = arrays[metadata['axial_offsets_um_float64']['__ndarray__']].copy()
            weights = arrays[metadata['axial_weights_float64']['__ndarray__']].copy()
            reflection = bool(arrays[metadata['reflection_xy']['__ndarray__']][0])
        lineage = old['lineage']
        section = make_coherent_subject_section_v6(
            plan, ouv, identity_yx, (reflection, False), offsets, weights, lineage,
            {'parent_section_id': lineage['section_id'],
             'parent_metadata_sha256': old['artifact_sha256'][old['artifacts']['metadata']]},
            atlas, allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
            subject_to_ccf_mapper=exact_map)
        rendered = section['raw_rendered_channels'].to('cuda')
        clean, support = rendered[0], rendered[1].clamp(0, 1)
        rng = np.random.default_rng(np.random.SeedSequence([
            SEED, plan_row['animal_index'], old['section_index'], 1]))
        mode = MODES[int(rng.integers(3))]
        appearance = {'gain': float(rng.uniform(.6, 1.4)),
            'gamma': float(np.exp(rng.uniform(np.log(.6), np.log(1.6)))),
            'invert': bool(rng.integers(2)), 'noise_std': float(rng.uniform(.005, .05)),
            'background_mean': float(rng.uniform(0, .8)),
            'background_slope_yx': rng.uniform(-.15, .15, 2).tolist(),
            'illumination': float(rng.uniform(0, .15)), 'texture': float(rng.uniform(0, .06)),
            'mask_dilate': bool(rng.integers(2)),
            'mask_radius': int(rng.integers(1, max(2, SIDE // 32 + 1))),
            'damage': bool(rng.random() < .2),
            'ellipse': [*rng.uniform(-.7, .7, 2), *rng.uniform(.05, .25, 2)]}
        noise_seed = int(np.random.SeedSequence([
            SEED, plan_row['animal_index'], old['section_index'], 2]).generate_state(1, dtype=np.uint64)[0])
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
                     -F.max_pool2d(-F.pad(value, (radius,) * 4), 2 * radius + 1, 1)[0, 0] > 0)
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
        physical_id = f"dev256-{lineage['section_id']}"
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
                      'noise_seed_uint64': noise_seed, 'source_plan_receipt_sha256':
                      plan_row['subject_plan_receipt_sha256']} ]}
        augmentation_seed = SEED * 10000000 + plan_row['animal_index'] * 1000 + old['section_index']
        sample = sample_one_shot_stream(None, [plan_row['animal_index']], augmentation_seed,
                                        side=SIDE, source_sample=source)
        filename = f"animal_{plan_row['animal_index']:02d}_section_{old['section_index']:02d}.npz"
        np.savez_compressed(OUT / filename, inputs=sample['inputs'][0].cpu().numpy(),
            target_state=sample['state'][0].cpu().numpy(),
            target_centre_um=sample['centre'][0].cpu().numpy(),
            valid_mask=sample['valid_mask'][0].cpu().numpy(),
            reflection=sample['reflection'][0].cpu().numpy(),
            offsets_um=sample['offsets'][0].cpu().numpy(),
            weights=sample['weights'][0].cpu().numpy(),
            eligible=sample['eligible'][0].cpu().numpy())
        record = {'file': filename, 'sha256': sha(OUT / filename),
                  **{key: lineage[key] for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id', 'section_id')},
                  'panel_physical_section_id': physical_id, 'plan_receipt_sha256':
                  plan_row['subject_plan_receipt_sha256'], 'parent_section_index': old['section_index'],
                  'parent_artifact_sha256': old['artifact_sha256'],
                  'appearance_mode': mode, 'augmentation_seed': augmentation_seed,
                  'eligible': bool(sample['eligible'][0]), 'valid_pixels': int(sample['valid_mask'][0].sum()),
                  'provenance': sample['provenance'][0]}
        records.append(record)
        print(json.dumps({'panel_section': len(records), 'animal_id': record['animal_id'],
                          'eligible': record['eligible']}), flush=True)
    del parameters, plan
assert len(records) == 128 and len({row['panel_physical_section_id'] for row in records}) == 128
(OUT / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
(OUT / 'completed.json').write_text(json.dumps({'physical_sections': len(records),
    'animals': len({row['animal_id'] for row in records}),
    'eligible': sum(row['eligible'] for row in records), 'modes': {mode: sum(row['appearance_mode'] == mode
        for row in records) for mode in MODES}, 'protocol_sha256': sha(OUT / 'protocol.json'),
    'records_sha256': sha(OUT / 'records.jsonl'), 'scope': protocol['scope']}, indent=2))
