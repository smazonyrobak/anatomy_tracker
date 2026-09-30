"""Optional synthetic clicks/metadata from one physical probe per TRAIN animal.

Preparation only: not wired into the model, its trainer, or the GUI. A straight
subject-space insertion maps to a generally curved CCF trajectory. Surgical
bounds below are SUBJECT stereotaxic coordinates relative to mapped bregma;
they are not hard constraints on a straight CCF ray. Only identity section
processing is supported; stored finite-raster reflections are applied once.
"""
import hashlib
import json

import numpy as np
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_subject_deformation_v2 import (
    _ccf_to_subject_points_from_verified_plan_v2,
    _subject_to_ccf_points_from_verified_plan_v2,
    subject_deformation_plan_receipt_v2,
)


STEREOTAXIC_SIGN = np.array([-1., -1., 1.])
CONTEXT_DIVISORS = np.array([10000., 10000., 1., 10000., 10000., 10000., 1., 90., 90., 1., 10000., 1.])
CONTEXT_FIELDS = (
    'section_ap_low_um', 'section_ap_high_um', 'section_ap_available',
    'entry_ap_um', 'entry_ml_um', 'entry_radius_um', 'entry_available',
    'elevation_low_deg', 'elevation_high_deg', 'elevation_available',
    'maximum_insertion_depth_um', 'depth_available',
)


def _rng(root_seed, *identity):
    payload = json.dumps(['joint-v7-physical-probe', int(root_seed), *identity], separators=(',', ':'))
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64DXSM(seed)), seed


def sample_subject_probe(
    plan, annotation_ap_dv_ml, cortical_region_ids, lineage, probe_id,
    root_seed, bregma_ccf_um, source_identifiers, *,
    origin_um=(0., 0., 0.), voxel_um=(25., 25., 25.),
    elevation_range_deg=(45., 90.), depth_range_um=(1500., 6000.), track_step_um=25.,
    dorsal_cortical_voxels=None,
):
    """Sample independently of every section; caller supplies pinned atlas/plan.

    Caller authenticates source files once. The supplied receipt is checked
    against live accepted coefficients here. Dorsal Isocortex voxel-face entries
    follow the GUI's first-tissue-column rule. The physical origin is a voxel
    BOUNDARY: index i has centre origin+(i+.5)*voxel. bregma_ccf_um must use that
    same convention (GUI legacy voxel indices need the +.5 conversion).
    Entry sampling is uniform over cortical
    AP/ML columns, not surface area. Tracks stop at the first sampled brain exit;
    25um stepping is an occupancy approximation, not an exact surface mesh.
    The caller may precompute identical [AP,first-DV,ML] cortical voxels in
    AP/ML row-major order once; this changes no random draws or geometry.
    """
    receipt = subject_deformation_plan_receipt_v2(plan)['receipt_sha256']
    if receipt != source_identifiers['subject_plan_receipt_sha256']:
        raise ValueError('probe plan differs from the caller-authenticated receipt')
    if lineage['split'] != 'train' or lineage['animal_id'] != plan['provenance']['animal_id']:
        raise ValueError('probe generation requires the matching TRAIN animal')
    rng, seed = _rng(root_seed, lineage['animal_id'], probe_id, 'shared-track')
    annotation = np.asarray(annotation_ap_dv_ml)
    origin, voxel = np.asarray(origin_um), np.asarray(voxel_um)
    if dorsal_cortical_voxels is None:
        top = (annotation != 0).argmax(axis=1)
        ap, ml = np.indices(top.shape)
        ap, ml = np.argwhere(np.isin(annotation[ap, top, ml], list(cortical_region_ids))).T
        candidates = np.column_stack((ap, top[ap, ml], ml))
    else:
        candidates = np.asarray(dorsal_cortical_voxels)
    bregma_subject = _ccf_to_subject_points_from_verified_plan_v2(np.asarray(bregma_ccf_um)[None], plan)[0]
    for attempt in range(32):
        ap, dv, ml = candidates[rng.integers(len(candidates))]
        entry_ccf = origin + voxel * np.array([ap + .5, dv, ml + .5])
        entry = _ccf_to_subject_points_from_verified_plan_v2(entry_ccf[None], plan)[0]
        elevation, azimuth = np.deg2rad(rng.uniform(*elevation_range_deg)), rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(elevation) * np.cos(azimuth), np.sin(elevation), np.cos(elevation) * np.sin(azimuth)])
        requested_depth = rng.uniform(*depth_range_um)
        depth = np.arange(track_step_um / 2, requested_depth, track_step_um)
        query = entry + depth[:, None] * direction
        ccf = _subject_to_ccf_points_from_verified_plan_v2(query, plan)
        ijk = np.floor((ccf - origin) / voxel).astype(int)
        valid = ((ijk >= 0) & (ijk < annotation.shape)).all(1)
        inside = np.zeros(len(depth), dtype=bool)
        inside[valid] = annotation[tuple(ijk[valid].T)] > 0
        exits = np.flatnonzero(~inside)
        retained = int(exits[0]) if len(exits) else len(depth)
        if retained and depth[retained - 1] >= 500.:
            break
    else:
        raise ValueError('no physically entering >=500um TRAIN probe in 32 independent attempts')
    actual_depth = float(depth[retained - 1])
    depths = np.r_[0., depth[:retained]]
    subject_points = entry + depths[:, None] * direction
    ccf_points = _subject_to_ccf_points_from_verified_plan_v2(subject_points, plan)
    return {
        'lineage': {**lineage, 'probe_id': str(probe_id)},
        'source_identifiers': dict(source_identifiers),
        'subject_plan_receipt_sha256': receipt,
        'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
        'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
        'ccf_context_sha256': plan['provenance']['ccf_context_sha256'],
        'root_seed': int(root_seed), 'track_seed_uint64': seed, 'physical_draw_attempt': attempt,
        'entry_subject_ap_dv_ml_um': entry, 'entry_ccf_ap_dv_ml_um': entry_ccf,
        'entry_dorsal_atlas_voxel_ap_dv_ml': np.array([ap, dv, ml]),
        'direction_subject_ap_dv_ml': direction, 'elevation_deg': float(np.rad2deg(elevation)),
        'latent_azimuth_rad': float(azimuth), 'azimuth_is_observed': False,
        'requested_depth_um': float(requested_depth), 'depth_um': actual_depth,
        'stopped_before_sampled_exit': bool(retained < len(depth)),
        'occupancy_step_um': float(track_step_um), 'track_depth_um': depths,
        'track_subject_ap_dv_ml_um': subject_points, 'track_ccf_ap_dv_ml_um': ccf_points,
        'bregma_ccf_ap_dv_ml_um': np.asarray(bregma_ccf_um).copy(),
        'bregma_subject_ap_dv_ml_um': bregma_subject,
        'entry_subject_stereotaxic_ap_dv_ml_um': (entry - bregma_subject) * STEREOTAXIC_SIGN,
        'metadata_frame': 'subject AP anterior, DV dorsal, ML right; origin forward-mapped bregma',
        'surface_rule': 'first nonzero dorsal atlas voxel must descend from Isocortex315; forward-map its dorsal voxel face',
        'annotation_origin_ap_dv_ml_um': origin, 'annotation_voxel_ap_dv_ml_um': voxel,
        'physical_origin_convention': 'voxel boundary; voxel i centre is origin+(i+.5)*voxel',
        'sampling': {'elevation_range_deg': elevation_range_deg, 'depth_range_um': depth_range_um,
                     'independent_of_section_geometry': True, 'minimum_in_brain_depth_um': 500.},
    }


def sample_probe_metadata(probe, root_seed, variant_id='default', *, missing_probability=.35, contradiction_probability=.1):
    """Shared surgery metadata; hard bounds are NOT Gaussian uncertainty SDs.

    Corruption/consistency flags are training targets, never extra input features.
    AP-range slots stay unavailable until an explicit physical anchor is agreed.
    Context normalization is returned explicitly; no current GUI adapter exists.
    """
    rng, seed = _rng(root_seed, probe['lineage']['animal_id'], probe['lineage']['probe_id'], 'metadata', variant_id)
    truth_entry = probe['entry_subject_stereotaxic_ap_dv_ml_um'][[0, 2]]
    radius = rng.uniform(100., 700.)
    phi = rng.uniform(-np.pi, np.pi)
    centre = truth_entry + .7 * radius * np.sqrt(rng.random()) * np.array([np.cos(phi), np.sin(phi)])
    elevation = probe['elevation_deg']
    bounds = np.clip(elevation + np.array([-1., 1.]) * rng.uniform(3., 15., 2), 0., 90.)
    max_depth = probe['depth_um'] + rng.uniform(100., 1000.)
    available = rng.random(3) >= missing_probability
    contradictory = (rng.random(3) < contradiction_probability) & available
    if contradictory[0]:
        centre = truth_entry + (radius + rng.uniform(200., 1000.)) * np.array([np.cos(phi), np.sin(phi)])
    if contradictory[1]:
        bounds = np.clip((elevation + 45.) % 90. + np.array([-5., 5.]), 0., 90.)
    if contradictory[2]:
        max_depth = probe['depth_um'] * rng.uniform(.5, .8)
    raw = np.array([0., 0., 0., *centre, radius, available[0], *bounds, available[1], max_depth, available[2]])
    for keep, slots in zip(available, (slice(3, 6), slice(7, 9), slice(10, 11))):
        if not keep:
            raw[slots] = 0.
    return {
        'lineage': dict(probe['lineage']), 'metadata_variant_id': str(variant_id),
        'root_seed': int(root_seed), 'metadata_seed_uint64': seed,
        'context_float32': (raw / CONTEXT_DIVISORS).astype(np.float32),
        'context_raw_units': raw, 'context_fields': CONTEXT_FIELDS,
        'context_divisors': CONTEXT_DIVISORS.copy(), 'metadata_frame': probe['metadata_frame'],
        'bregma_subject_ap_dv_ml_um': probe['bregma_subject_ap_dv_ml_um'].copy(),
        'section_ap_anchor': None, 'azimuth_is_observed': False,
        'bound_type': 'hard AP/ML disk in um, hard elevation interval in degrees, hard maximum path length in um',
        'training_only_available_entry_angle_depth': available,
        'training_only_contradictory_entry_angle_depth': contradictory,
        'sampling': {'missing_probability': missing_probability, 'contradiction_probability': contradiction_probability},
    }


def observe_probe_section(
    probe, plan, subject_ouv_um, reflection_xy, thickness_um, visible_support,
    section_identifiers, observation_id, root_seed, *, mark_sigma_px=.5,
    missing_probability=.2, drop_probability=.15, mark_spacing_px=4.,
    observed_pullback_yx_px=None, source_identifiers=None,
):
    """Project a physical shared track into a finite slab, with no pose retries.

    OUV uses O+x/W U+y/H V, AP/DV/ML um. visible_support is the caller's exact
    finite support * retained tissue * selected brush mask, not inferred pixels.
    Finite thickness is a geometric support interval, not PSF quadrature weights.
    Returned track points retain their signed depth inside that interval. They
    must not be supervised as exact centre-plane CCF coordinates. Marks are
    optional synthetic user clicks; no fluorescent track is painted into images.
    Gaussian noise has SD in OBSERVED pixels, separate from hard surgery bounds.
    """
    for key in ('animal_id', 'specimen_id', 'experiment_id', 'split'):
        if section_identifiers[key] != probe['lineage'][key]:
            raise ValueError('probe/section subject lineage differs')
    if plan['receipt_sha256'] != probe['subject_plan_receipt_sha256']:
        raise ValueError('probe/section subject plan differs')
    support = np.asarray(visible_support)
    height, width = support.shape
    identity = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing='ij'))
    if observed_pullback_yx_px is not None and not np.array_equal(observed_pullback_yx_px, identity):
        raise ValueError('nonidentity section-processing pullback needs an explicit inverse before using probe observations')
    # Geometry/noise branch excludes brush mode: paired presentations share clicks.
    rng, seed = _rng(root_seed, probe['lineage']['animal_id'], probe['lineage']['probe_id'], section_identifiers['section_id'], 'marks')
    ouv = np.asarray(subject_ouv_um, dtype=np.float64).reshape(3, 3)
    normal = np.cross(ouv[1], ouv[2])
    normal /= np.linalg.norm(normal)
    direction, entry = probe['direction_subject_ap_dv_ml'], probe['entry_subject_ap_dv_ml_um']
    basis = np.linalg.pinv(ouv[1:].T)
    xy0 = basis @ (entry - ouv[0]) * (width, height)
    dxy = basis @ direction * (width, height)
    signed0, slope = np.dot(entry - ouv[0], normal), np.dot(direction, normal)
    interval = np.array([0., probe['depth_um']])
    # Intersect path-length parameter with thickness and both finite FOV bounds.
    for value, rate, low, high in ((signed0, slope, -thickness_um / 2, thickness_um / 2),
                                  (xy0[0], dxy[0], 0., width - 1.), (xy0[1], dxy[1], 0., height - 1.)):
        if abs(rate) < 1e-12:
            if not low <= value <= high:
                interval = np.array([1., 0.])
                break
        else:
            limits = np.sort((np.array([low, high]) - value) / rate)
            interval = np.array([max(interval[0], limits[0]), min(interval[1], limits[1])])
    count = 0 if interval[1] < interval[0] else max(1, min(64, int(np.ceil(np.linalg.norm(dxy) * np.diff(interval)[0] / mark_spacing_px))))
    depths = interval[0] + (np.arange(count) + .5) / max(count, 1) * np.diff(interval)[0]
    canonical_xy = xy0 + depths[:, None] * dxy
    flags = np.asarray(reflection_xy, dtype=bool)
    xy = canonical_xy * (1 - 2 * flags.astype(int)) + flags * (width - 1, height - 1)
    noisy_xy = xy + rng.normal(0., mark_sigma_px, xy.shape)
    provided = bool(rng.random() >= missing_probability)
    not_dropped = rng.random(count) >= drop_probability
    visibility = map_coordinates(support.astype(float), xy[:, ::-1].T, order=1, mode='constant', cval=0.) if count else np.empty(0)
    noisy_in_fov = ((noisy_xy >= 0) & (noisy_xy <= (width - 1, height - 1))).all(1)
    selected = (visibility > .05) & not_dropped & noisy_in_fov & provided
    subject_points = entry + depths[:, None] * direction
    ccf_points = _subject_to_ccf_points_from_verified_plan_v2(subject_points, plan) if count else np.empty((0, 3))
    offsets = signed0 + depths * slope
    centre_subject = subject_points - offsets[:, None] * normal
    centre_ccf = _subject_to_ccf_points_from_verified_plan_v2(centre_subject, plan) if count else np.empty((0, 3))
    yy, xx = identity
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in noisy_xy[selected]:
        heatmap = np.maximum(heatmap, np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * max(.75, mark_sigma_px)**2))).astype(np.float32)
    return {
        'lineage': {**section_identifiers, 'probe_id': probe['lineage']['probe_id'], 'observation_id': str(observation_id)},
        'source_identifiers': dict(source_identifiers or {}),
        'subject_plan_receipt_sha256': probe['subject_plan_receipt_sha256'],
        'root_seed': int(root_seed), 'mark_seed_uint64': seed,
        'reflection_xy': flags.copy(), 'slab_thickness_um': float(thickness_um),
        'geometric_path_interval_um': interval if count else np.empty(0),
        'candidate_pixel_xy': xy, 'candidate_noisy_pixel_xy': noisy_xy,
        'candidate_visibility_weight': visibility, 'candidate_selected': selected,
        'candidate_depth_um': depths, 'candidate_through_plane_offset_um': offsets,
        'candidate_track_subject_ap_dv_ml_um': subject_points,
        'candidate_track_ccf_ap_dv_ml_um': ccf_points,
        'candidate_projected_centre_subject_ap_dv_ml_um': centre_subject,
        'candidate_projected_centre_ccf_ap_dv_ml_um': centre_ccf,
        'mark_pixel_xy': noisy_xy[selected], 'mark_sigma_px': float(mark_sigma_px),
        'mark_heatmap_float32': heatmap, 'marks_available': bool(selected.any()),
        'marks_missing_by_draw': not provided, 'geometric_intersection': bool(count),
        'sampling': {'missing_probability': missing_probability, 'drop_probability': drop_probability,
                     'mark_spacing_px': mark_spacing_px, 'visibility_threshold': .05,
                     'maximum_candidates': 64, 'heatmap_sigma_px': max(.75, mark_sigma_px)},
    }
