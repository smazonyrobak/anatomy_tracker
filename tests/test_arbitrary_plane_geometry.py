import pytest
import torch
import torch.nn.functional as F

from training.arbitrary_plane_geometry import (
    MAX_ABS_INPLANE_SHEAR,
    MAX_INPLANE_SPAN_VOXELS,
    MIN_INPLANE_SPAN_VOXELS,
    allen_index_plane_to_physical_um_plane,
    allen_index_to_physical_um_points,
    allen_index_to_physical_um_vectors,
    allen_to_quicknii_points,
    allen_to_quicknii_vectors,
    flip_frame,
    frame_to_quicknii_ouv,
    horizontal_flip_quicknii_ouv,
    identity_biased_rotation_6d_to_frame,
    legacy_quicknii_boundary_pose_to_frame,
    legacy_renderer_pose_to_frame,
    normalized_raster_to_ccf,
    normalized_raster_to_quicknii,
    physical_um_plane_to_allen_index_plane,
    physical_um_to_allen_index_points,
    physical_um_to_allen_index_vectors,
    positive_inplane_basis,
    quicknii_ouv_to_frame,
    quicknii_to_allen_points,
    render_arbitrary_plane,
    render_legacy_inclusive_plane,
    rotation_6d_to_frame,
    vertical_flip_quicknii_ouv,
)


def test_invalid_rotation_ouv_and_basis_states_are_rejected_and_zero_residual_is_valid():
    for rotation in (
        torch.zeros(6),
        torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0, float("nan"), 0.0]),
    ):
        with pytest.raises(ValueError):
            rotation_6d_to_frame(rotation)

    residual = torch.zeros(2, 6, dtype=torch.float64, requires_grad=True)
    frame = identity_biased_rotation_6d_to_frame(residual)
    assert torch.equal(frame, torch.eye(3, dtype=torch.float64).expand(2, -1, -1))
    (frame * torch.arange(9, dtype=torch.float64).reshape(3, 3)).sum().backward()
    assert torch.isfinite(residual.grad).all()

    for ouv in (
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        torch.tensor([float("inf"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ):
        with pytest.raises(ValueError):
            quicknii_ouv_to_frame(ouv)

    bounded_basis = positive_inplane_basis(
        torch.log(torch.tensor([MIN_INPLANE_SPAN_VOXELS, MAX_INPLANE_SPAN_VOXELS])),
        torch.tensor(MAX_ABS_INPLANE_SHEAR),
    )
    half_basis = positive_inplane_basis(
        torch.tensor([10.0, 10.0], dtype=torch.float16),
        torch.tensor(MAX_ABS_INPLANE_SHEAR, dtype=torch.float16),
    )
    assert torch.isfinite(bounded_basis).all()
    assert torch.isfinite(half_basis).all() and half_basis.dtype == torch.float32
    for log_spans, shear in (
        (torch.log(torch.tensor([MIN_INPLANE_SPAN_VOXELS / 2, 1.0])), torch.tensor(0.0)),
        (torch.log(torch.tensor([MAX_INPLANE_SPAN_VOXELS * 2, 1.0])), torch.tensor(0.0)),
        (torch.zeros(2), torch.tensor(MAX_ABS_INPLANE_SHEAR * 2)),
        (torch.tensor([0.0, float("nan")]), torch.tensor(0.0)),
    ):
        with pytest.raises(ValueError):
            positive_inplane_basis(log_spans, shear)


def test_rotation_6d_is_right_handed_and_basis_has_positive_orientation():
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.0, -2.0, 4.0, 1.0]],
        dtype=torch.float64,
    )
    frame = rotation_6d_to_frame(rotation)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    assert torch.allclose(frame.transpose(-1, -2) @ frame, identity, atol=1e-12)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(2, dtype=torch.float64), atol=1e-12)

    basis = positive_inplane_basis(
        torch.log(torch.tensor([[455.0, 319.0], [271.0, 193.0]], dtype=torch.float64)),
        torch.tensor([0.0, -0.37], dtype=torch.float64),
    )
    assert torch.all(torch.linalg.det(basis) > 0)
    assert basis[1, 0, 1] != 0


def test_aligned_two_plane_batch_and_explicit_grid_singletons_match_individual_calls():
    torch.manual_seed(92)
    volume = torch.rand(7, 5, 9, dtype=torch.float64)
    labels = torch.arange(volume.numel()).reshape(volume.shape)
    center = torch.tensor([[3.0, 2.0, 4.0], [2.4, 2.1, 4.3]], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.7, -0.2, 1.1, 0.3, 1.2, -0.4]],
        dtype=torch.float64,
    )
    frame = rotation_6d_to_frame(rotation)
    basis = positive_inplane_basis(
        torch.log(torch.tensor([[6.0, 4.0], [5.0, 3.0]], dtype=torch.float64)),
        torch.tensor([0.0, 0.2], dtype=torch.float64),
    )
    image, rendered_labels = render_arbitrary_plane(
        volume, center, frame, basis, (4, 6), labels
    )
    individual = [
        render_arbitrary_plane(volume, center[i], frame[i], basis[i], (4, 6), labels)
        for i in range(2)
    ]
    assert torch.allclose(image, torch.cat([item[0] for item in individual]))
    assert torch.equal(rendered_labels, torch.cat([item[1] for item in individual]))

    st = torch.rand(2, 4, 6, 2, dtype=torch.float64)
    ccf = normalized_raster_to_ccf(
        center[:, None, None], frame[:, None, None], basis[:, None, None], st
    )
    ouv = frame_to_quicknii_ouv(center, frame, basis, volume.shape)
    quicknii = normalized_raster_to_quicknii(ouv[:, None, None], st)
    assert torch.allclose(allen_to_quicknii_points(ccf, volume.shape), quicknii, atol=1e-12)

    with pytest.raises(ValueError, match="leading dimensions"):
        frame_to_quicknii_ouv(center, frame[0], basis[0], volume.shape)


def test_anisotropic_voxel_centres_planes_and_unilateral_render_share_one_contract():
    shape = (5, 4, 7)
    origin = torch.tensor([-100.0, 50.0, 300.0], dtype=torch.float64)
    spacing = torch.tensor([40.0, 20.0, 10.0], dtype=torch.float64)
    index = torch.tensor([2.0, 1.0, 5.0], dtype=torch.float64)
    physical = origin + (index + 0.5) * spacing
    assert torch.equal(physical_um_to_allen_index_points(physical, origin, spacing), index)
    assert torch.equal(allen_index_to_physical_um_points(index, origin, spacing), physical)

    physical_vector = torch.tensor([80.0, -40.0, 30.0], dtype=torch.float64)
    index_vector = physical_um_to_allen_index_vectors(physical_vector, spacing)
    assert torch.equal(index_vector, torch.tensor([2.0, -2.0, 3.0], dtype=torch.float64))
    assert torch.equal(allen_index_to_physical_um_vectors(index_vector, spacing), physical_vector)

    normal_um = F.normalize(torch.tensor([2.0, -3.0, 5.0], dtype=torch.float64), dim=0)
    offset_um = torch.tensor(37.0, dtype=torch.float64)
    normal_index, offset_index = physical_um_plane_to_allen_index_plane(
        normal_um, offset_um, spacing
    )
    recovered_normal, recovered_offset = allen_index_plane_to_physical_um_plane(
        normal_index, offset_index, spacing
    )
    assert torch.allclose(recovered_normal, normal_um, atol=1e-12)
    assert torch.allclose(recovered_offset, offset_um, atol=1e-12)
    on_plane_physical = physical + offset_um * normal_um
    on_plane_index = physical_um_to_allen_index_points(on_plane_physical, origin, spacing)
    assert torch.allclose(
        torch.dot(normal_index, on_plane_index - index), offset_index, atol=1e-12
    )

    volume = torch.zeros(shape, dtype=torch.float64)
    labels = torch.zeros(shape, dtype=torch.int64)
    volume[2, 1, 5] = 0.75
    labels[2, 1, 5] = 917
    frame = rotation_6d_to_frame(
        torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    )
    basis = torch.eye(2, dtype=torch.float64)
    raster_center = index + torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    image, rendered_labels = render_arbitrary_plane(
        volume, raster_center, frame, basis, (1, 1), labels
    )
    assert image.item() == pytest.approx(0.75)
    assert rendered_labels.item() == 917
    ouv = frame_to_quicknii_ouv(raster_center, frame, basis, shape)
    quicknii_point = normalized_raster_to_quicknii(
        ouv, torch.tensor([0.0, 0.0], dtype=torch.float64)
    )
    recovered_index = quicknii_to_allen_points(quicknii_point, shape)
    assert torch.allclose(recovered_index, index, atol=1e-12)
    assert torch.equal(
        allen_index_to_physical_um_points(recovered_index, origin, spacing), physical
    )


def test_explicit_allen_quicknii_axis_and_laterality_convention():
    shape = (31, 23, 19)
    point = torch.tensor([7.0, 11.0, 13.0], dtype=torch.float64)
    vector = torch.tensor([2.0, -3.0, 5.0], dtype=torch.float64)
    assert torch.equal(allen_to_quicknii_points(point, shape), torch.tensor([13.0, 24.0, 12.0], dtype=torch.float64))
    assert torch.equal(allen_to_quicknii_vectors(vector), torch.tensor([5.0, -2.0, 3.0], dtype=torch.float64))

    left = allen_to_quicknii_points(torch.tensor([7.0, 11.0, 2.0]), shape)
    right = allen_to_quicknii_points(torch.tensor([7.0, 11.0, 16.0]), shape)
    assert right[0] > left[0]
    assert torch.equal(right[1:], left[1:])


def test_coronal_sagittal_horizontal_and_extreme_oblique_ouv_round_trips():
    rotations = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, -2.0, 4.0, 1.0],
        ],
        dtype=torch.float64,
    )
    center = torch.tensor(
        [[216.0, 159.5, 227.5], [263.5, 159.5, 180.0],
         [263.5, 120.0, 227.5], [90.0, 61.0, 310.0]],
        dtype=torch.float64,
    )
    frame = rotation_6d_to_frame(rotations)
    basis = positive_inplane_basis(
        torch.log(torch.tensor([[455.0, 319.0], [319.0, 527.0],
                                [455.0, 527.0], [388.0, 276.0]], dtype=torch.float64)),
        torch.tensor([0.0, 0.0, 0.0, 0.61], dtype=torch.float64),
    )
    ouv = frame_to_quicknii_ouv(center, frame, basis)
    recovered = quicknii_ouv_to_frame(ouv)
    assert torch.allclose(recovered[0], center, atol=1e-11)
    assert torch.allclose(recovered[1], frame, atol=1e-11)
    assert torch.allclose(recovered[2], basis, atol=1e-11)
    assert torch.allclose(frame_to_quicknii_ouv(*recovered), ouv, atol=1e-11)

    st = torch.tensor([[0.0, 0.0], [0.2, 0.7], [1.0, 1.0]], dtype=torch.float64)
    ccf = normalized_raster_to_ccf(
        center[3, None], frame[3, None], basis[3, None], st
    )
    assert torch.allclose(allen_to_quicknii_points(ccf), normalized_raster_to_quicknii(ouv[3, None], st), atol=1e-11)


def test_horizontal_and_vertical_flips_are_involutions_and_only_reparameterize_raster():
    rotation = torch.tensor([1.0, 2.0, 3.0, -2.0, 4.0, 1.0], dtype=torch.float64)
    center = torch.tensor([170.0, 111.0, 301.0], dtype=torch.float64)
    frame = rotation_6d_to_frame(rotation)
    basis = positive_inplane_basis(
        torch.log(torch.tensor([370.0, 241.0], dtype=torch.float64)),
        torch.tensor(-0.42, dtype=torch.float64),
    )
    ouv = frame_to_quicknii_ouv(center, frame, basis)
    height, width = 241, 370
    horizontal = horizontal_flip_quicknii_ouv(ouv, width)
    vertical = vertical_flip_quicknii_ouv(ouv, height)
    assert torch.allclose(horizontal_flip_quicknii_ouv(horizontal, width), ouv, atol=1e-12)
    assert torch.allclose(vertical_flip_quicknii_ouv(vertical, height), ouv, atol=1e-12)

    st = torch.tensor([0.17, 0.81], dtype=torch.float64)
    assert torch.allclose(
        normalized_raster_to_quicknii(horizontal, st),
        normalized_raster_to_quicknii(
            ouv, torch.tensor([(width - 1) / width - st[0], st[1]])
        ),
        atol=1e-12,
    )
    assert torch.allclose(
        normalized_raster_to_quicknii(vertical, st),
        normalized_raster_to_quicknii(
            ouv, torch.tensor([st[0], (height - 1) / height - st[1]])
        ),
        atol=1e-12,
    )
    flipped_state = flip_frame(center, frame, basis, (height, width), horizontal=True)
    assert torch.allclose(frame_to_quicknii_ouv(*flipped_state), horizontal, atol=1e-11)
    restored_state = flip_frame(*flipped_state, (height, width), horizontal=True)
    assert torch.allclose(frame_to_quicknii_ouv(*restored_state), ouv, atol=1e-11)


def _cardinal_state(name, shape, index):
    depth, height, width = shape
    if name == "coronal":
        center = torch.tensor([index, height / 2, width / 2], dtype=torch.float64)
        rotation = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        spans, output = (width, height), (height, width)
    elif name == "sagittal":
        center = torch.tensor([depth / 2, height / 2, index], dtype=torch.float64)
        rotation = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        spans, output = (height, depth), (depth, height)
    else:
        center = torch.tensor([depth / 2, index, width / 2], dtype=torch.float64)
        rotation = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        spans, output = (width, depth), (depth, width)
    frame = rotation_6d_to_frame(rotation)
    basis = torch.diag(torch.tensor(spans, dtype=torch.float64))
    return center, frame, basis, output


def test_cardinal_planes_match_direct_scalar_and_label_slices():
    shape = (7, 5, 9)
    ap, dv, ml = torch.meshgrid(*(torch.arange(size, dtype=torch.float64) for size in shape), indexing="ij")
    volume = 100.0 * ap + 10.0 * dv + ml
    labels = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    expected = {
        "coronal": (3, volume[3], labels[3]),
        "sagittal": (4, volume[:, :, 4], labels[:, :, 4]),
        "horizontal": (2, volume[:, 2, :], labels[:, 2, :]),
    }
    for name, (index, scalar_slice, label_slice) in expected.items():
        center, frame, basis, output = _cardinal_state(name, shape, index)
        image, rendered_labels = render_arbitrary_plane(
            volume, center, frame, basis, output, labels
        )
        assert torch.allclose(image[0, 0], scalar_slice, atol=1e-11)
        assert torch.equal(rendered_labels[0, 0], label_slice)


@pytest.mark.parametrize("label_shape", [(8, 6, 10), (6, 4, 8)])
def test_renderer_rejects_mismatched_label_atlas_shape(label_shape):
    shape = (7, 5, 9)
    volume = torch.zeros(shape, dtype=torch.float64)
    labels = torch.zeros(label_shape, dtype=torch.int64)
    state = _cardinal_state("coronal", shape, 3)
    with pytest.raises(ValueError, match="identical AP-DV-ML shapes"):
        render_arbitrary_plane(volume, *state, labels)


def test_imported_quicknii_ouv_uses_first_voxel_origin_and_W_H_raster_spans():
    shape = (7, 5, 9)
    ap, dv, ml = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float64) for size in shape), indexing="ij"
    )
    volume = 100.0 * ap + 10.0 * dv + ml
    labels = torch.arange(volume.numel()).reshape(shape)
    imported_ouv = torch.tensor(
        [0.0, shape[0] - 3.0, float(shape[1]), float(shape[2]), 0.0, 0.0,
         0.0, 0.0, -float(shape[1])],
        dtype=torch.float64,
    )
    state = quicknii_ouv_to_frame(imported_ouv, shape)
    image, rendered_labels = render_arbitrary_plane(
        volume, *state, shape[1:], labels
    )
    assert torch.allclose(frame_to_quicknii_ouv(*state, shape), imported_ouv, atol=1e-12)
    assert torch.allclose(image[0, 0], volume[3], atol=1e-12)
    assert torch.equal(rendered_labels[0, 0], labels[3])


def test_oblique_horizontal_raster_flip_preserves_pose_but_atlas_reflection_flips_laterality():
    shape = (25, 9, 13)
    pose = torch.tensor([0.0, 12.0, -7.0], dtype=torch.float64)
    ouv = frame_to_quicknii_ouv(
        *legacy_quicknii_boundary_pose_to_frame(
            pose, shape, bregma_ap_index=12.0, voxel_um=1.0
        ),
        shape,
    )
    horizontal = horizontal_flip_quicknii_ouv(ouv, shape[2])
    landmark_st = torch.tensor([0.18, 0.37], dtype=torch.float64)
    unilateral_landmark = normalized_raster_to_quicknii(ouv, landmark_st)
    reparameterized_landmark = normalized_raster_to_quicknii(
        horizontal,
        torch.stack(((shape[2] - 1) / shape[2] - landmark_st[0], landmark_st[1])),
    )
    assert unilateral_landmark[0] < (shape[2] - 1) / 2
    assert torch.allclose(reparameterized_landmark, unilateral_landmark, atol=1e-12)

    atlas_reflection = ouv.clone()
    atlas_reflection[0] = shape[2] - ouv[0]
    atlas_reflection[3] = -ouv[3]
    atlas_reflection[6] = -ouv[6]
    reflected_landmark = normalized_raster_to_quicknii(atlas_reflection, landmark_st)
    assert reflected_landmark[0] > (shape[2] - 1) / 2
    assert not torch.allclose(atlas_reflection, horizontal)

    def lr_tilt_deg(plane):
        normal = torch.cross(plane[3:6], plane[6:9], dim=0)
        normal = -normal if normal[1] < 0 else normal
        return torch.rad2deg(torch.atan(normal[0] / normal[1]))

    assert lr_tilt_deg(ouv).item() == pytest.approx(pose[1].item())
    assert lr_tilt_deg(horizontal).item() == pytest.approx(pose[1].item())
    assert lr_tilt_deg(atlas_reflection).item() == pytest.approx(-pose[1].item())


def test_arbitrary_plane_renderer_is_differentiable_through_frame_center_and_basis():
    torch.manual_seed(4)
    volume = torch.rand(11, 9, 13)
    labels = torch.arange(volume.numel()).reshape(volume.shape)
    rotation = torch.tensor([0.8, -0.3, 1.1, 0.2, 1.3, -0.4], requires_grad=True)
    center = torch.tensor([5.1, 4.2, 5.7], requires_grad=True)
    log_spans = torch.log(torch.tensor([7.0, 5.0])).requires_grad_()
    shear = torch.tensor(0.23, requires_grad=True)
    frame = rotation_6d_to_frame(rotation)
    basis = positive_inplane_basis(log_spans, shear)
    image, rendered_labels = render_arbitrary_plane(
        volume, center, frame, basis, (6, 8), labels
    )
    weights = torch.linspace(0.3, 1.7, image.numel()).reshape_as(image)
    (image * weights).sum().backward()
    for value in (rotation.grad, center.grad, log_spans.grad, shear.grad):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0
    assert rendered_labels.dtype == labels.dtype
    assert rendered_labels.shape == image.shape


def test_legacy_renderer_adapter_matches_inclusive_voxel_center_geometry():
    shape = (25, 9, 13)
    pose = torch.tensor([[0.0, 12.0, -7.0], [2.0, -9.0, 6.0]], dtype=torch.float64)
    center, frame, basis = legacy_renderer_pose_to_frame(
        pose, shape, bregma_ap_index=12.0, voxel_um=1.0
    )
    y, x = torch.meshgrid(
        torch.arange(shape[1], dtype=torch.float64),
        torch.arange(shape[2], dtype=torch.float64), indexing="ij",
    )
    st = torch.stack((x / (shape[2] - 1), y / (shape[1] - 1)), -1)
    points = normalized_raster_to_ccf(
        center[:, None, None], frame[:, None, None], basis[:, None, None],
        st[None].expand(len(pose), -1, -1, -1),
    )
    expected_ap = (
        12.0 - pose[:, 0, None, None]
        + torch.tan(torch.deg2rad(pose[:, 1]))[:, None, None] * (x - (shape[2] - 1) / 2)
        + torch.tan(torch.deg2rad(pose[:, 2]))[:, None, None] * (y - (shape[1] - 1) / 2)
    )
    assert torch.allclose(points[..., 0], expected_ap, atol=1e-12)
    assert torch.allclose(points[..., 1], y.expand_as(points[..., 1]), atol=1e-12)
    assert torch.allclose(points[..., 2], x.expand_as(points[..., 2]), atol=1e-12)

    volume = torch.rand(shape, dtype=torch.float64)
    image, _ = render_legacy_inclusive_plane(volume, center, frame, basis, shape[1:])
    manual_grid = torch.stack(
        (x[None].expand(len(pose), -1, -1) / (shape[2] - 1) * 2 - 1,
         y[None].expand(len(pose), -1, -1) / (shape[1] - 1) * 2 - 1,
         expected_ap / (shape[0] - 1) * 2 - 1), -1,
    )[:, None]
    expected_image = F.grid_sample(
        volume[None, None].expand(len(pose), -1, -1, -1, -1), manual_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, :, 0]
    assert torch.allclose(image, expected_image, atol=1e-11)
    quicknii_state = legacy_quicknii_boundary_pose_to_frame(
        pose, shape, bregma_ap_index=12.0, voxel_um=1.0
    )
    quicknii_image, _ = render_arbitrary_plane(
        volume, *quicknii_state, shape[1:]
    )
    assert torch.allclose(quicknii_image, expected_image, atol=1e-11)


def test_quicknii_raster_indices_and_legacy_inclusive_centres_cannot_be_mixed():
    shape = (7, 5, 9)
    volume = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape)
    center, frame, quicknii_basis, output = _cardinal_state("coronal", shape, 3)
    legacy_center = torch.tensor(
        [3.0, (shape[1] - 1) / 2, (shape[2] - 1) / 2], dtype=torch.float64
    )
    legacy_basis = torch.diag(
        torch.tensor([shape[2] - 1, shape[1] - 1], dtype=torch.float64)
    )

    quicknii_image, _ = render_arbitrary_plane(
        volume, center, frame, quicknii_basis, output
    )
    legacy_image, _ = render_legacy_inclusive_plane(
        volume, legacy_center, frame, legacy_basis, output
    )
    wrong_quicknii, _ = render_arbitrary_plane(
        volume, legacy_center, frame, legacy_basis, output
    )
    wrong_legacy, _ = render_legacy_inclusive_plane(
        volume, center, frame, quicknii_basis, output
    )
    assert torch.allclose(quicknii_image[0, 0], volume[3], atol=1e-11)
    assert torch.allclose(legacy_image[0, 0], volume[3], atol=1e-11)
    assert not torch.allclose(wrong_quicknii[0, 0], volume[3])
    assert not torch.allclose(wrong_legacy[0, 0], volume[3])


def test_historical_quicknii_boundary_adapter_is_exact_and_distinct_from_renderer_span():
    shape = (528, 320, 456)
    pose = torch.tensor([-1375.0, 11.0, -8.0], dtype=torch.float64)
    state = legacy_quicknii_boundary_pose_to_frame(pose, shape)
    ouv = frame_to_quicknii_ouv(*state, shape)
    slope_lr, slope_dv = torch.tan(torch.deg2rad(pose[1:])); ap_index = 216.0 - pose[0] / 25.0
    origin_ap = ap_index - slope_lr * 227.5 - slope_dv * 159.5
    expected = torch.stack(
        (pose.new_tensor(0.0), pose.new_tensor(528.0) - origin_ap, pose.new_tensor(320.0),
         pose.new_tensor(456.0), -456.0 * slope_lr, pose.new_tensor(0.0),
         pose.new_tensor(0.0), -320.0 * slope_dv, pose.new_tensor(-320.0))
    )
    assert torch.allclose(ouv, expected, atol=1e-11)

    renderer_ouv = frame_to_quicknii_ouv(*legacy_renderer_pose_to_frame(pose, shape), shape)
    assert torch.linalg.vector_norm(renderer_ouv[3:6]) < torch.linalg.vector_norm(ouv[3:6])
    assert torch.linalg.vector_norm(renderer_ouv[6:9]) < torch.linalg.vector_norm(ouv[6:9])
    assert not torch.allclose(renderer_ouv, ouv)
