"""Future v7 conditioning inputs only; not wired into any model or GUI worker."""
import numpy as np


CONTEXT_FIELDS = (
    'section_ap_low_um', 'section_ap_high_um', 'section_ap_available',
    'entry_ap_um', 'entry_ml_um', 'entry_radius_um', 'entry_available',
    'elevation_low_deg', 'elevation_high_deg', 'elevation_available',
    'maximum_insertion_depth_um', 'depth_available',
)
CONTEXT_DIVISORS = np.array([10000., 10000., 1., 10000., 10000., 10000., 1., 90., 90., 1., 10000., 1.])


def pack_joint_slice_constraints(*, source_identifiers, bregma_metadata,
                                probe_identifiers=None, subject_entry_ap_ml_radius_um=None,
                                subject_elevation_bounds_deg=None, maximum_insertion_depth_um=None,
                                atlas_stereotaxic_image_centre_ap_bounds_um=None):
    """One explicitly identified probe/animal, optional values in um/degrees.

    Surgical entry/elevation/depth are SUBJECT metadata, not a straight CCF-ray
    likelihood or a slice normal. AP bounds, if explicitly supplied, refer to the
    processed input IMAGE pixel-centre midpoint (after any selected crop), mapped
    through its curved surface into
    ATLAS stereotaxic AP, not section O,
    bregma intersection, or legacy coronal AP search controls. No azimuth exists.
    None means absent; zero is a supplied value. Bounds/radius are hard limits,
    never Gaussian SDs. Unsupported/conflicting inputs retain raw data and return
    context_float32=None, not a plausible-looking clipped or all-zero context.
    A packed result certifies encoding only, not consistency with anatomy/marks,
    trained conditioning, calibrated probability, or uncertainty coverage.
    """
    raw_input = dict(subject_entry_ap_ml_radius_um=subject_entry_ap_ml_radius_um,
                     subject_elevation_bounds_deg=subject_elevation_bounds_deg,
                     maximum_insertion_depth_um=maximum_insertion_depth_um,
                     atlas_stereotaxic_image_centre_ap_bounds_um=atlas_stereotaxic_image_centre_ap_bounds_um)
    raw = np.zeros(12)
    errors = []
    for name, supplied, slots, flag in (
        ('image-centre atlas AP bounds', atlas_stereotaxic_image_centre_ap_bounds_um, slice(0, 2), 2),
        ('subject entry AP/ML/radius', subject_entry_ap_ml_radius_um, slice(3, 6), 6),
        ('subject elevation bounds', subject_elevation_bounds_deg, slice(7, 9), 9),
        ('subject maximum insertion depth', None if maximum_insertion_depth_um is None else [maximum_insertion_depth_um], slice(10, 11), 11),
    ):
        if supplied is None:
            continue
        value = np.asarray(supplied, dtype=float)
        if value.shape != (slots.stop - slots.start,) or not np.isfinite(value).all():
            errors.append(f'{name}: requires one finite value set, not multiple or incomplete constraints')
            continue
        raw[slots], raw[flag] = value, 1.
    if raw[2] and raw[0] > raw[1]:
        errors.append('image-centre atlas AP bounds: lower exceeds upper')
    if raw[6] and raw[5] < 0:
        errors.append('subject entry radius must be nonnegative')
    if raw[9] and not 0. <= raw[7] <= raw[8] <= 90.:
        errors.append('subject elevation bounds must satisfy 0 <= lower <= upper <= 90; no clipping applied')
    if raw[11] and raw[10] < 0:
        errors.append('subject maximum insertion depth must be nonnegative')
    if any(value is not None for value in (subject_entry_ap_ml_radius_um, subject_elevation_bounds_deg, maximum_insertion_depth_um)):
        probe_identifiers = {} if probe_identifiers is None else probe_identifiers
        if not probe_identifiers.get('probe_id'):
            errors.append('Surgical metadata requires one explicitly selected probe ID')
        animal = source_identifiers.get('animal_id')
        if not animal or not probe_identifiers.get('animal_id') or animal != probe_identifiers['animal_id']:
            errors.append('Surgical metadata requires matching nonempty section/probe animal IDs')
        if (source_identifiers.get('specimen_id') and probe_identifiers.get('specimen_id')
                and source_identifiers['specimen_id'] != probe_identifiers['specimen_id']):
            errors.append('Section/probe specimen IDs conflict')
    return {
        'status': 'unsupported_or_conflicting' if errors else 'packed', 'errors': errors,
        'context_float32': None if errors else (raw / CONTEXT_DIVISORS).astype(np.float32),
        'context_raw_units': raw, 'context_fields': CONTEXT_FIELDS, 'context_divisors': CONTEXT_DIVISORS.copy(),
        'raw_input': raw_input, 'source_identifiers': dict(source_identifiers),
        'probe_identifiers': None if probe_identifiers is None else dict(probe_identifiers),
        'bregma_metadata': dict(bregma_metadata),
        'metadata_frame': 'SUBJECT stereotaxic surgery; AP anterior, DV dorsal, ML follows supplied positive axis',
        'section_ap_anchor': None if atlas_stereotaxic_image_centre_ap_bounds_um is None else
            'processed input IMAGE-CENTRE after selected crop: ((W-1)/2,(H-1)/2), mapped through the full predicted surface; ATLAS stereotaxic AP',
        'bound_type': 'hard AP/ML disk in um, hard elevation interval in degrees, hard maximum path length in um',
        'azimuth_is_observed': False, 'ccf_hard_ray_likelihood': False,
        'conditioning_enabled': False, 'probabilities_calibrated': False,
    }


def pack_gui_joint_slice_constraints(constraints_by_probe, selected_probe, source_identifiers, *,
                                    probe_identifiers, bregma_voxel_ap_dv_ml, voxel_um=25.,
                                    atlas_stereotaxic_image_centre_ap_bounds_um=None):
    """Pack the raw GUI ProbeInsertionConstraint, NOT its effective/clipped copy.

    The caller must explicitly select the same probe as any supplied mark channel.
    Raw section marks can use prepare_joint_slice_input; this function never pools
    probes, creates marks, invents azimuth, or infers animal IDs. Legacy max-depth
    fallback to physical shank length is deliberately not treated as user metadata.
    GUI positive-ML numbers are preserved without silently changing laterality.
    """
    constraint = constraints_by_probe.get(selected_probe) if selected_probe else None
    enabled = constraint is not None and constraint.enabled
    bregma_voxel = np.asarray(bregma_voxel_ap_dv_ml, dtype=float)
    result = pack_joint_slice_constraints(
        source_identifiers=source_identifiers, probe_identifiers=probe_identifiers,
        bregma_metadata={'atlas_bregma_voxel_ap_dv_ml': bregma_voxel.tolist(),
                         'atlas_bregma_ccf_ap_dv_ml_um': ((bregma_voxel + .5) * voxel_um).tolist(),
                         'atlas_voxel_um': np.asarray(voxel_um).tolist(),
                         'atlas_origin_convention': 'voxel boundary; centre i = (i+.5)*voxel',
                         'subject_origin': 'surgical bregma; its deformed CCF position is not inferred',
                         'subject_bregma_stereotaxic_ap_dv_ml_um': [0., 0., 0.],
                         'stereotaxic_axis_sign_ap_dv_ml': [-1., -1., 1.],
                         'ml_axis_note': 'GUI +ML numeric axis retained: Allen CCF right; historical GUI tooltip incorrectly said animal left; no numeric flip'},
        subject_entry_ap_ml_radius_um=(constraint.ap_um, constraint.ml_um, constraint.radius_um) if enabled else None,
        subject_elevation_bounds_deg=(constraint.angle_deg - constraint.angle_tolerance_deg,
                                      constraint.angle_deg + constraint.angle_tolerance_deg) if enabled else None,
        maximum_insertion_depth_um=constraint.maximum_insertion_depth_um if enabled else None,
        atlas_stereotaxic_image_centre_ap_bounds_um=atlas_stereotaxic_image_centre_ap_bounds_um,
    )
    result['raw_gui_constraints_by_probe'] = {name: vars(value).copy() for name, value in constraints_by_probe.items()}
    result['selected_probe'] = selected_probe
    if constraints_by_probe and not selected_probe:
        result['errors'].append('Explicit probe selection required; no probe is inferred or pooled')
    if selected_probe and probe_identifiers is not None and selected_probe != probe_identifiers.get('probe_id'):
        result['errors'].append('Selected GUI probe and supplied probe ID differ')
    if result['errors']:
        result.update(status='unsupported_or_conflicting', context_float32=None)
    return result
