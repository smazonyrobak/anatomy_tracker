"""Prepared same-model full-catalogue retrieval -> conditional native ribbon path."""

import torch


def imagekey_ribbon_inference_v6(
    model, image, outline, outline_available, gallery_descriptors, atlas_volume,
    origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um, axial_offsets_um, axial_weights,
    *, top_k=32, cell_chunk_size=4, refinement_steps=3, temperature=0.1,
):
    """Non-gradient inference; no checkpoint/gallery loading or preprocessing.

    Caller supplies one eval-mode model with image-key and ribbon heads, its
    authenticated complete catalogue runtime, matching N,2,D gallery on the
    image device, and matching atlas/physical geometry. Image/outline are the
    unchanged prepared B,1,96,96 tensors; availability is B. An absent brush is
    a zero outline with availability zero, never inferred segmentation.

    Gallery rows follow catalogue order, R=(identity,horizontal raster flip),
    generated with this SAME model's atlas stem/shared encoder/descriptor and
    the frozen 96x96 gallery PSF/preprocessing contract (currently 50um/S9).
    Rebuild cached descriptors whenever any of those weights or rendering
    contracts change. This function cannot authenticate external cache ancestry.
    Refinement uses the supplied physical PSF offsets/weights, S or B,S, with
    identity section processing and a single spatial reflection per component.

    Returns complete coarse cell log probabilities and separate B,K,R refined
    modes, not averaged mirrored geometry. Selection is conditional on retained
    cells. The coarse omitted mass is NOT a post-refinement tail estimate;
    neither distribution is calibrated. No teacher pose, support mask, surgical
    constraint, continuous covariance, or additional encoder is used. This glue
    does not establish joint training, global capture, or deployment readiness.
    """
    assert not model.training
    assert image.shape[1:] == outline.shape[1:] == (1, 96, 96)
    assert cell_chunk_size > 0
    with torch.inference_mode(), torch.autocast(device_type=image.device.type, enabled=False):
        pose = model.pose_model
        catalogue = pose._catalogue(pose.catalogue_runtime_v6.expand(image.shape[0]), image)
        cell_count = pose.catalogue_runtime_v6.cell_count
        assert 1 <= top_k <= cell_count
        assert gallery_descriptors.shape[:2] == (cell_count, 2)
        affine = catalogue["representation_to_canonical_raster_affine"]
        expected_affine = affine.new_tensor([
            [[1., 0., 0.], [0., 1., 0.]],
            [[-1., 0., 0.], [0., 1., 0.]],
        ])
        assert torch.equal(affine, expected_affine.expand_as(affine))
        reflection = torch.tensor([False, True], device=image.device)
        query = pose.descriptor_from_features(pose.encode_histology(image, outline, outline_available))
        representation_score = (
            pose.image_key_cosine_logits(query, gallery_descriptors, temperature)
            + catalogue["representation_log_weight"].float()
        )
        # Same reduction order as the frozen full-gallery experiment readouts.
        cell_log_probability = (
            torch.logsumexp(representation_score, dim=-1)
            + catalogue["cell_log_mass"].float()
        ).log_softmax(dim=-1)
        component_log_probability = cell_log_probability[..., None] + representation_score.log_softmax(dim=-1)
        top = torch.argsort(cell_log_probability, dim=1, descending=True, stable=True)[:, :top_k]
        rows = torch.arange(image.shape[0], device=image.device)[:, None]
        initial_log_mass = component_log_probability[rows, top]
        initial_states = catalogue["cell_states"][rows, top]
        retained_log_mass = torch.logsumexp(cell_log_probability[rows, top], dim=1).clamp_max(0.)
        omitted = torch.ones_like(cell_log_probability, dtype=torch.bool).scatter_(1, top, False)
        omitted_log_mass = torch.logsumexp(cell_log_probability.masked_fill(~omitted, -torch.inf), dim=1)
        keys = (
            "final_component_state", "final_canonical_residual_local_um",
            "final_canonical_director_delta_local",
            "final_observed_centre_ccf_ap_dv_ml_um",
            "final_observed_ccf_slab_ap_dv_ml_um",
            "final_component_refinement_log_score",
        )
        parts = {key: [] for key in keys}
        for start in range(0, top_k, cell_chunk_size):
            stop = min(start + cell_chunk_size, top_k)
            result = model.refine_ribbon(
                image, outline, outline_available, atlas_volume,
                initial_states[:, start:stop], initial_log_mass[:, start:stop],
                reflection, (96, 96), origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um,
                catalogue["support_origin_ap_dv_ml_um"], axial_offsets_um,
                axial_weights, refinement_steps,
            )
            # Chunk-normalized probabilities are deliberately discarded.
            for key in keys:
                parts[key].append(result[key])
            del result
        final = {key: torch.cat(value, dim=1) for key, value in parts.items()}
        score = final["final_component_refinement_log_score"]
        refined_log_score = initial_log_mass.to(score) + score
        conditional_log_probability = torch.log_softmax(
            refined_log_score.flatten(1), dim=1,
        ).reshape_as(refined_log_score)
        selected = conditional_log_probability.flatten(1).argmax(dim=1)
        selected_cell_rank, selected_representation = selected // 2, selected % 2
        return {
            "coarse_full_cell_log_probability": cell_log_probability,
            "top_cell_index": top,
            "initial_component_log_mass": initial_log_mass,
            "coarse_retained_log_mass": retained_log_mass,
            "coarse_retained_mass": retained_log_mass.exp(),
            "coarse_omitted_log_mass": omitted_log_mass,
            "coarse_omitted_mass": omitted_log_mass.exp(),
            **final,
            "final_component_log_score": refined_log_score,
            "conditional_kept_component_log_probability": conditional_log_probability,
            "conditional_kept_component_probability": conditional_log_probability.exp(),
            "selected_component_index": selected,
            "selected_cell_rank": selected_cell_rank,
            "selected_cell_index": top.gather(1, selected_cell_rank[:, None])[:, 0],
            "selected_representation_index": selected_representation,
            "horizontal_reflection": reflection,
            "probability_scope": "conditional_retained_cells_and_raster_representations",
            "post_refinement_omitted_mass": None,
            "probabilities_calibrated": False,
            "processing_scope": "identity_section_processing_supplied_psf_no_acquired_constraints",
        }
