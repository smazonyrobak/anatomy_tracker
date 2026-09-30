"""Stream fresh sections from 64 TRAIN deformation bases without resident GPU maps.

Load only after both map preparations have frozen their completed.json files.
This extends the v7 sampler; the original eight bases, four synthetic DEV bases,
atlas binding, appearance draws, PSF integration and physical targets are unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_streaming_synthetic_v7 import (
    MAP_ROOT, PLAN_ROOT, SUPPORT_SEED, VARIANTS, VIRTUAL_SEED,
    load_streaming_synthetic_v7, sample_streaming_synthetic_v7,
)
from training.arbitrary_plane_subject_sampling_v6 import subject_support_bounds_um
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

NEW_ROOT = Path('I:/AnatomyTracker/data/joint_v7_independent_subject_maps_001')


def load_streaming_synthetic_v7_64(device='cuda'):
    context = load_streaming_synthetic_v7(device)
    for base in context['bases']:
        base['displacement_cpu'] = base.pop('displacement').cpu()
    old_virtual = context['subjects']
    assert len(context['bases']) == 8 and len(old_virtual) == 8 * VARIANTS

    completed_path = NEW_ROOT / 'completed.json'
    completed_bytes = completed_path.read_bytes()
    completed = json.loads(completed_bytes)
    protocol_path = NEW_ROOT / 'protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == completed['protocol_sha256']
    protocol = json.loads(protocol_bytes)
    assert completed['total_independent_train_deformations'] == 64
    assert completed['existing_train_subjects'] == 8
    assert completed['held_out_synthetic_dev_subjects'] == 4
    assert completed['new_sections'] == 0
    assert completed['parent_plan_completed_sha256'] == context['bindings'][str(PLAN_ROOT / 'completed.json')]
    assert protocol['parent_completed_sha256'] == completed['parent_plan_completed_sha256']
    assert len(completed['new_train_subjects']) == 56
    bindings = context['bindings']
    bindings[str(completed_path)] = hashlib.sha256(completed_bytes).hexdigest()
    bindings[str(protocol_path)] = completed['protocol_sha256']
    for name, expected in protocol['source_sha256'].items():
        path = NEW_ROOT / 'source' / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected
        bindings[str(path)] = digest

    # Match the original sampler's fixed atlas-interior point selection exactly.
    atlas_support = context['atlas'][1].cpu().numpy() > 0
    selected = np.random.default_rng(SUPPORT_SEED).choice(np.flatnonzero(atlas_support), 4096, replace=False)
    support_ccf = (np.array(np.unravel_index(selected, atlas_support.shape)).T.astype(np.float64) + .5) * 25.
    del atlas_support
    assert hashlib.sha256(support_ccf.tobytes()).hexdigest() == context['provenance']['support_ccf_points_sha256']
    tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
        'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um', 'global_scale', 'frozen_center_um')
    for index, record in enumerate(completed['new_train_subjects'], 8):
        assert record['animal_index'] == index and record['split'] == 'train'
        directory = NEW_ROOT / record['directory']
        frozen_path = directory / 'completed.json'
        frozen_bytes = frozen_path.read_bytes()
        assert hashlib.sha256(frozen_bytes).hexdigest() == record['completed_sha256']
        bindings[str(frozen_path)] = record['completed_sha256']
        subject_path = directory / 'subject.json'
        subject_bytes = subject_path.read_bytes()
        assert hashlib.sha256(subject_bytes).hexdigest() == record['subject_sha256']
        bindings[str(subject_path)] = record['subject_sha256']
        subject = json.loads(subject_bytes)
        assert subject['lineage']['split'] == 'train'
        assert subject['lineage']['animal_index'] == index
        assert subject['protocol_sha256'] == completed['protocol_sha256']
        map_path = directory / 'inverse_map.npz'
        with map_path.open('rb') as stream:
            map_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert map_sha == record['inverse_map_sha256'] == subject['inverse_map_sha256']
        bindings[str(map_path)] = map_sha
        with np.load(map_path) as arrays:
            displacement_cpu = torch.from_numpy(np.moveaxis(arrays['displacement_um'], -1, 0).copy())
            lower = arrays['lower_unscaled_um'].copy()
            upper = arrays['upper_unscaled_um'].copy()
            centre = arrays['global_centre_um'].copy()
            scale = arrays['global_scale'].copy()
        for name, expected in record['artifact_sha256'].items():
            path = directory / name
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == expected
            bindings[str(path)] = digest
        plan = _read_raw_artifact(directory, record['plan_files'])
        assert plan['receipt_sha256'] == subject['lineage']['subject_plan_receipt_sha256']
        parameters = [torch.as_tensor(plan['state'][key], device=device, dtype=torch.float64) for key in tensor_keys]
        forward_support = map_accepted_subject_points_torch_v6(
            torch.as_tensor(support_ccf, device=device, dtype=torch.float64), *parameters,
            inverse=False, steps=int(plan['resolved_config']['flow']['steps']), batch_size=8192).cpu().numpy()
        anchor = np.asarray(subject['mapped_atlas_centre_um'])
        check = subject['mapping_checks'][-1]
        assert check['passed'] and check['max_error_um'] <= 5
        base = {'lineage': subject['lineage'], 'displacement_cpu': displacement_cpu,
            'lower': torch.as_tensor(lower, device=device, dtype=torch.float32),
            'upper': torch.as_tensor(upper, device=device, dtype=torch.float32),
            'map_centre': torch.as_tensor(centre, device=device, dtype=torch.float32),
            'map_scale': torch.as_tensor(scale, device=device, dtype=torch.float32),
            'bounds': subject_support_bounds_um(plan), 'anchor': anchor,
            'support_points': forward_support, 'plan_receipt': plan['receipt_sha256'],
            'mapping_check': check,
            'support_points_sha256': hashlib.sha256(forward_support.tobytes()).hexdigest()}
        context['bases'].append(base)
        for variant in range(VARIANTS):
            vrng = np.random.default_rng(np.random.SeedSequence([VIRTUAL_SEED, index, variant]))
            virtual_scale = np.exp(vrng.uniform(-.1, .1, 3))
            old_virtual.append({'virtual_index': len(old_virtual), 'base_index': index, 'variant': variant,
                'virtual_subject_id': f"{subject['lineage']['subject_id']}-virtual-affine-v7-{variant:03d}",
                'base_lineage': subject['lineage'], 'anchor_um': anchor.tolist(),
                'scale_ap_dv_ml': virtual_scale.tolist(),
                'bounds_um': (anchor + virtual_scale * (base['bounds'] - anchor)).tolist(),
                'scope': 'derived TRAIN synthetic subject; fixed global affine, no independent local anatomy or biology'})
        del plan, parameters

    provenance = context['provenance']
    provenance['base_subjects'] = [
        {'lineage': b['lineage'], 'plan_receipt': b['plan_receipt'], 'mapping_check': b['mapping_check'],
         'forward_support_points_sha256': b['support_points_sha256']} for b in context['bases']]
    provenance['virtual_subjects'] = old_virtual
    provenance['input_sha256'] = bindings
    provenance['independent_map_protocol_sha256'] = completed['protocol_sha256']
    provenance['new_independent_train_bases'] = 56
    provenance['all_train_bases'] = 64
    provenance['map_residency'] = 'CPU; transfer only selected base maps per generated batch, release after sampling'
    provenance['map_bytes_cpu_total'] = sum(b['displacement_cpu'].numel() * b['displacement_cpu'].element_size()
                                            for b in context['bases'])
    provenance['map_bytes_largest_single_base'] = max(b['displacement_cpu'].numel() * b['displacement_cpu'].element_size()
                                                      for b in context['bases'])
    provenance['source_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert len(context['bases']) == 64 and len(old_virtual) == 64 * VARIANTS
    if context['atlas'].is_cuda:
        torch.cuda.empty_cache()
    return context


def sample_streaming_synthetic_v7_64(context, subject_indices, seed, side=192):
    selected = sorted({context['subjects'][int(i)]['base_index'] for i in subject_indices})
    device = context['atlas'].device
    for index in selected:
        base = context['bases'][index]
        base['displacement'] = base['displacement_cpu'].to(device)
    batch = sample_streaming_synthetic_v7(context, subject_indices, seed, side)
    for index in selected:
        context['bases'][index].pop('displacement')
    return batch
