"""Fixed 192px physical re-render of held-out v6 synthetic DEV sections.

Sixteen preselected arbitrary planes per each of four untouched synthetic
subjects; three appearance/background modes per plane. No learned model, no
upsampling of the old 96px images, and no checkpoint selection here.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_coherent_subject_v6 import make_coherent_subject_section_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_subject_deformation_v2 import subject_deformation_plan_receipt_v2
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

OLD = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
PLANS = ROOT / 'data/joint_v6_coherent_subject_plans_002'
OUTPUT = ROOT / 'data/joint_v7_synthetic_dev192_001'
SIDE, SEED = 192, 2026100101
MODES = ('raw', 'exact_black', 'imperfect_brush')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


cohort = json.loads((OLD / 'completed.json').read_text())
plan_index = json.loads((PLANS / 'completed.json').read_text())
plans = sorted((r for r in plan_index['subjects'] if r['split'] == 'development'), key=lambda r: r['animal_index'])
sections = sorted((r for r in cohort['sections'] if r['lineage']['split'] == 'development' and r['section_index'] % 2 == 0),
                  key=lambda r: (r['lineage']['animal_id'], r['section_index']))
assert len(plans) == 4 and len(sections) == 64
assert {r['animal_id'] for r in plans} == {r['lineage']['animal_id'] for r in sections}
assert not {r['animal_id'] for r in plans} & {r['animal_id'] for r in plan_index['subjects'] if r['split'] == 'train'}
assert all(sum(r['lineage']['animal_id'] == p['animal_id'] for r in sections) == 16 for p in plans)

repository = Path(__file__).resolve().parents[1]
sources = ('prepare_joint_v7_synthetic_dev192.py', 'arbitrary_plane_coherent_subject_v6.py',
           'arbitrary_plane_subject_torch_v6.py', 'arbitrary_plane_subject_deformation_v2.py',
           'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
           'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_subject_section_v2.py')
protocol = {'seed': SEED, 'resolution': [SIDE, SIDE], 'modes': MODES, 'physical_sections': 64,
    'observations': 192, 'split': 'development', 'source': 'four frozen disjoint v6 development subject plans',
    'selection': 'section indices 0,2,...,30 in each of four development subjects; fixed before checkpoint evaluation; no support-based selection',
    'geometry': 'same preselected physical OUV and 9-point PSF as frozen96 DEV, newly mapped through accepted 3D subject plan at192px; not image upscaling',
    'coordinates': 'O+x/W U+y/H V, exact accepted subject-to-CCF inverse at full192 grid; horizontal reflection once',
    'appearance': 'fresh fixed gain/gamma/inversion, gradient/texture/noise, optional damage; shared pre-brush image across modes',
    'eligibility': 'visible finite-support mass >=192*192/144; every plane and mode retained even when ineligible',
    'limits': 'same one Allen atlas and four synthetic deformation subjects; not independent biological animals or expert truth',
    'old_completed_sha256': sha(OLD / 'completed.json'), 'plan_completed_sha256': sha(PLANS / 'completed.json'),
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6, 'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'source_sha256': {name: sha(repository / 'training' / name) for name in sources},
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'numpy': np.__version__, 'torch': torch.__version__}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
del annotation, atlas_array
yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIDE), torch.linspace(-1, 1, SIDE), indexing='ij')
py, px = np.meshgrid(np.arange(SIDE, dtype=np.float64), np.arange(SIDE, dtype=np.float64), indexing='ij')
identity_yx = np.stack((py, px))
tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
               'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um', 'global_scale', 'frozen_center_um')
records = []
for plan_row in plans:
    directory = PLANS / plan_row['directory']
    for name, expected in plan_row['artifact_sha256'].items():
        assert sha(directory / name) == expected
    plan = _read_raw_artifact(directory, plan_row['plan_files'])
    assert subject_deformation_plan_receipt_v2(plan)['receipt_sha256'] == plan_row['subject_plan_receipt_sha256']
    parameters = [torch.as_tensor(plan['state'][key], device='cuda', dtype=torch.float64) for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])
    def exact_map(points):
        query = torch.from_numpy(np.ascontiguousarray(points)).to(device='cuda', dtype=torch.float64)
        return map_accepted_subject_points_torch_v6(query, *parameters, inverse=True, steps=steps, batch_size=8192).cpu().numpy()

    for old in (r for r in sections if r['lineage']['animal_id'] == plan_row['animal_id']):
        for name, expected in old['artifact_sha256'].items():
            assert sha(OLD / name) == expected
        metadata = json.loads((OLD / old['artifacts']['metadata']).read_text())
        with np.load(OLD / old['artifacts']['arrays']) as arrays:
            ouv = arrays[metadata['subject_ouv_ap_dv_ml_um_float64']['__ndarray__']].copy()
            offsets = arrays[metadata['axial_offsets_um_float64']['__ndarray__']].copy()
            weights = arrays[metadata['axial_weights_float64']['__ndarray__']].copy()
            reflection = bool(arrays[metadata['reflection_xy']['__ndarray__']][0])
        lineage = {**old['lineage'], 'section_id': f"dev192-{old['lineage']['section_id']}"}
        section = make_coherent_subject_section_v6(plan, ouv, identity_yx, (reflection, False), offsets, weights,
            lineage, {'parent_section_id': old['lineage']['section_id'], 'parent_metadata_sha256': old['artifact_sha256'][old['artifacts']['metadata']]},
            atlas, allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
            subject_to_ccf_mapper=exact_map)
        rendered = section['raw_rendered_channels']
        support = rendered[1].clamp(0, 1)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, plan_row['animal_index'], old['section_index']]))
        noise_seed = int(np.random.SeedSequence([SEED, plan_row['animal_index'], old['section_index'], 2]).generate_state(1, dtype=np.uint64)[0])
        noise = torch.Generator().manual_seed(noise_seed)
        gain, gamma = rng.uniform(.6, 1.4), np.exp(rng.uniform(np.log(.6), np.log(1.6)))
        invert = bool(rng.integers(2))
        tissue = (rendered[0] / support.clamp_min(1e-6)).clamp(0, 1).pow(float(gamma))
        if invert:
            tissue = 1 - tissue
        illumination = F.interpolate(torch.randn(1, 1, 5, 5, generator=noise), (SIDE, SIDE), mode='bilinear', align_corners=False)[0, 0]
        tissue = (tissue * gain * (1 + rng.uniform(0, .15) * illumination)).clamp(0, 1)
        damage = bool(rng.random() < .2)
        ellipse = rng.uniform(-.7, .7, 2).tolist() + rng.uniform(.05, .25, 2).tolist()
        retained = (((yy - ellipse[0]) / ellipse[2]).square() + ((xx - ellipse[1]) / ellipse[3]).square() >= 1) if damage else torch.ones_like(support, dtype=torch.bool)
        visible = support * retained
        texture = F.interpolate(torch.randn(1, 1, 16, 16, generator=noise), (SIDE, SIDE), mode='bilinear', align_corners=False)[0, 0]
        slope = rng.uniform(-.15, .15, 2)
        background = rng.uniform(0, .8) + slope[0] * yy + slope[1] * xx + rng.uniform(0, .06) * texture
        before = (tissue * visible + background * (1 - visible) + rng.uniform(.005, .05) * torch.randn(SIDE, SIDE, generator=noise)).clamp(0, 1)
        exact = (support > 0) & retained
        radius, dilate = int(rng.integers(1, 7)), bool(rng.integers(2))
        imperfect = (F.max_pool2d(exact[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0 if dilate else
            -F.max_pool2d(-F.pad(exact[None, None].float(), (radius,) * 4), 2 * radius + 1, 1)[0, 0] > 0)
        masks = torch.stack((torch.ones_like(exact), exact, imperfect))
        eroded = masks.clone()
        eroded[:, 1:] &= masks[:, :-1]
        eroded[:, :-1] &= masks[:, 1:]
        eroded[:, :, 1:] &= masks[:, :, :-1]
        eroded[:, :, :-1] &= masks[:, :, 1:]
        eroded[:, [0, -1], :] = False
        eroded[:, :, [0, -1]] = False
        available = torch.tensor([False, True, True])
        outline = (masks & ~eroded) & available[:, None, None]
        inputs = torch.stack((before[None] * masks, outline.float(), available[:, None, None].expand(-1, SIDE, SIDE).float(),
                              torch.zeros(3, SIDE, SIDE), torch.zeros(3, SIDE, SIDE)), 1)
        visible = visible[None] * masks
        mass = visible.sum((1, 2))
        eligible = mass >= SIDE * SIDE / 144
        canonical_ouv = section['canonical_anatomy_plane_fit']['arrays']['physical_ouv_ap_dv_ml_um_float64']
        state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(canonical_ouv))).float()
        filename = f"subject_{plan_row['animal_index']:02d}_section_{old['section_index']:02d}.npz"
        np.savez_compressed(OUTPUT / filename, inputs=inputs.numpy().astype(np.float32), target_state=state.numpy(),
            target_centre_um=section['target_centre_ccf_coordinates_ap_dv_ml_um_float64'].astype(np.float32),
            visible_support=visible.numpy().astype(np.float32), offsets_um=offsets.astype(np.float32),
            weights=weights.astype(np.float32), reflection=np.array(reflection), eligible=eligible.numpy())
        row = {'file': filename, 'sha256': sha(OUTPUT / filename), 'animal_id': lineage['animal_id'],
            'subject_id': lineage['subject_id'], 'specimen_id': lineage['specimen_id'], 'experiment_id': lineage['experiment_id'],
            'section_id': lineage['section_id'], 'parent_section_id': old['lineage']['section_id'],
            'parent_metadata_sha256': old['artifact_sha256'][old['artifacts']['metadata']],
            'parent_arrays_sha256': old['artifact_sha256'][old['artifacts']['arrays']],
            'plan_receipt_sha256': plan_row['subject_plan_receipt_sha256'], 'split': 'development',
            'section_index': old['section_index'], 'reflection': reflection, 'modes': MODES,
            'visible_support_mass': mass.tolist(), 'eligible': eligible.tolist(),
            'normal_ap_dv_ml': section['canonical_anatomy_plane_fit']['arrays']['fitted_plane_unit_normal_ap_dv_ml_float64'].tolist()}
        records.append(row)
        print(json.dumps({'event': 'dev192_physical_section', 'animal_id': row['animal_id'],
            'section_index': row['section_index'], 'eligible': row['eligible']}), flush=True)
    del parameters, plan
(OUTPUT / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf8')
(OUTPUT / 'completed.json').write_text(json.dumps({'physical_sections': len(records), 'observations': len(records) * 3,
    'subjects': 4, 'protocol_sha256': sha(OUTPUT / 'protocol.json'), 'records_sha256': sha(OUTPUT / 'records.jsonl'),
    'all_eligible_mode_counts': np.array([r['eligible'] for r in records]).sum(0).tolist(),
    'scope': protocol['limits']}, indent=2), encoding='utf8')
