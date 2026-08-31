"""Versioned arbitrary-plane CCF geometry and differentiable rendering.

Allen arrays use ``(AP, DV, ML)`` voxel indices.  QuickNII RAS coordinates
use ``(ML, AP_size - AP, DV_size - DV)``; vectors therefore map as
``(dML, -dAP, -dDV)``.  A plane is a centre, a right-handed frame whose
columns are ``[u, v, n]``, and a positive-orientation 2-D basis.  The raster
edge vectors are ``[U, V] = [u, v] @ basis``.  The basis is retained because
valid QuickNII planes, including the legacy two-tilt plane, need not have
orthogonal raster axes.

Physical atlas coordinates use voxel-boundary origins: array index ``i`` is
the centre at ``origin_um + (i + 0.5) * voxel_size_um``.  Low-level raster
mappers use ordinary Torch broadcasting; a ``B``-plane image grid therefore
passes state tensors with two explicit singleton grid axes.  A QuickNII pixel
``(x,y)`` maps with normalized coordinates ``(x/W,y/H)``; no half-pixel term
is added.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


ARBITRARY_PLANE_GEOMETRY_VERSION = "ccf-arbitrary-plane-frame-basis-v1"
ALLEN_AXIS_ORDER = ("AP", "DV", "ML")
QUICKNII_AXIS_ORDER = ("ML", "AP_size-AP", "DV_size-DV")
DEFAULT_ALLEN_SHAPE_AP_DV_ML = (528, 320, 456)
LEGACY_RENDERER_CENTRE_CONTRACT = "legacy-renderer-voxel-centres-Wminus1-Hminus1-v1"
LEGACY_QUICKNII_BOUNDARY_CONTRACT = "legacy-quicknii-boundary-spans-W-H-v1"
QUICKNII_RASTER_INDEX_SAMPLING = "quicknii-raster-index-x-over-W-y-over-H-v1"
LEGACY_INCLUSIVE_CENTRE_SAMPLING = "legacy-inclusive-voxel-centres-v1"
# These prediction-time bounds dwarf the mouse atlas while preventing exp and
# shear overflow. Exact imported O/U/V is factored without applying them.
MIN_INPLANE_SPAN_VOXELS = 1e-3
MAX_INPLANE_SPAN_VOXELS = 1e6
MAX_ABS_INPLANE_SHEAR = 1e3


def _finite_floating(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")
    return value


def _aligned_state(
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = _finite_floating(center_ap_dv_ml, "center")
    frame = _finite_floating(frame_ap_dv_ml, "frame")
    basis = _finite_floating(inplane_basis, "in-plane basis")
    if center.shape[-1:] != (3,) or frame.shape[-2:] != (3, 3) or basis.shape[-2:] != (2, 2):
        raise ValueError("State tails must be (...,3), (...,3,3), and (...,2,2)")
    if center.shape[:-1] != frame.shape[:-2] or center.shape[:-1] != basis.shape[:-2]:
        raise ValueError("State leading dimensions must be exactly aligned")
    if frame.device != center.device or basis.device != center.device:
        raise ValueError("State tensors must share one device")
    if frame.dtype != center.dtype or basis.dtype != center.dtype:
        raise ValueError("State tensors must share one dtype")
    return center, frame, basis


def _origin_and_spacing(
    value: torch.Tensor,
    origin_um: torch.Tensor | tuple[float, float, float],
    voxel_size_um: torch.Tensor | tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(value)
    if not torch.is_floating_point(value):
        value = value.to(torch.get_default_dtype())
    origin = torch.as_tensor(origin_um, device=value.device, dtype=value.dtype)
    spacing = torch.as_tensor(voxel_size_um, device=value.device, dtype=value.dtype)
    if value.shape[-1:] != (3,) or origin.shape != (3,) or spacing.shape != (3,):
        raise ValueError("Coordinates, origin, and voxel size must end in three axes")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(origin).all()):
        raise ValueError("Coordinates and origin must be finite")
    if not bool(torch.isfinite(spacing).all()) or bool((spacing <= 0).any()):
        raise ValueError("Voxel sizes must be finite and positive")
    return value, origin, spacing


def rotation_6d_to_frame(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Map two 3-D vectors to continuous right-handed frame columns [u,v,n]."""
    rotation = _finite_floating(rotation_6d, "6-D rotation")
    if rotation.shape[-1:] != (6,):
        raise ValueError("6-D rotation must end in six coordinates")
    first, second = rotation.split(3, dim=-1)
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    tolerance = 64.0 * torch.finfo(rotation.dtype).eps
    if not bool(torch.isfinite(first_norm).all()) or not bool(torch.isfinite(second_norm).all()):
        raise ValueError("6-D rotation vector norms must be finite")
    if bool((first_norm <= tolerance).any()) or bool((second_norm <= tolerance).any()):
        raise ValueError("6-D rotation vectors must be nonzero")
    u = first / first_norm
    orthogonal_second = second - (second * u).sum(-1, keepdim=True) * u
    orthogonal_norm = torch.linalg.vector_norm(orthogonal_second, dim=-1, keepdim=True)
    if not bool(torch.isfinite(orthogonal_norm).all()) or bool(
        (orthogonal_norm <= tolerance * second_norm).any()
    ):
        raise ValueError("6-D rotation vectors must be non-collinear")
    v = orthogonal_second / orthogonal_norm
    n = torch.cross(u, v, dim=-1)
    return torch.stack((u, v, n), dim=-1)


def identity_biased_rotation_6d_to_frame(residual_6d: torch.Tensor) -> torch.Tensor:
    """Project a residual whose exact zero value represents the identity frame."""
    residual = _finite_floating(residual_6d, "6-D rotation residual")
    if residual.shape[-1:] != (6,):
        raise ValueError("6-D rotation residual must end in six coordinates")
    bias = residual.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    return rotation_6d_to_frame(residual + bias)


def positive_inplane_basis(log_spans: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    """Return a finite positive basis within generous physical training bounds."""
    log_spans = _finite_floating(log_spans, "log spans")
    if log_spans.shape[-1:] != (2,):
        raise ValueError("log spans must end in two coordinates")
    shear = _finite_floating(
        torch.as_tensor(shear, device=log_spans.device, dtype=log_spans.dtype), "shear"
    )
    if shear.shape != log_spans.shape[:-1]:
        raise ValueError("shear must have the same leading dimensions as log spans")
    work_log_spans = log_spans.float() if log_spans.dtype in (torch.float16, torch.bfloat16) else log_spans
    work_shear = shear.float() if shear.dtype in (torch.float16, torch.bfloat16) else shear
    lower = work_log_spans.new_tensor(MIN_INPLANE_SPAN_VOXELS).log()
    upper = work_log_spans.new_tensor(MAX_INPLANE_SPAN_VOXELS).log()
    if bool(((work_log_spans < lower) | (work_log_spans > upper)).any()):
        raise ValueError(
            f"In-plane spans must lie in [{MIN_INPLANE_SPAN_VOXELS}, "
            f"{MAX_INPLANE_SPAN_VOXELS}] voxels"
        )
    if bool((work_shear.abs() > MAX_ABS_INPLANE_SHEAR).any()):
        raise ValueError(f"Absolute in-plane shear must not exceed {MAX_ABS_INPLANE_SHEAR}")
    spans = torch.exp(work_log_spans)
    zeros = torch.zeros_like(work_shear)
    basis = torch.stack(
        (torch.stack((spans[..., 0], work_shear * spans[..., 1]), -1),
         torch.stack((zeros, spans[..., 1]), -1)),
        -2,
    )
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("In-plane basis must remain finite")
    return basis


def physical_um_to_allen_index_points(
    points_ap_dv_ml_um: torch.Tensor,
    origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Map physical voxel-centre coordinates to zero-based Allen array indices."""
    points, origin, spacing = _origin_and_spacing(
        points_ap_dv_ml_um, origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um
    )
    return (points - origin) / spacing - 0.5


def allen_index_to_physical_um_points(
    points_ap_dv_ml: torch.Tensor,
    origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Map zero-based Allen indices to physical voxel-centre coordinates."""
    points, origin, spacing = _origin_and_spacing(
        points_ap_dv_ml, origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um
    )
    return origin + (points + 0.5) * spacing


def physical_um_to_allen_index_vectors(
    vectors_ap_dv_ml_um: torch.Tensor,
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Convert physical displacements to Allen-index displacements."""
    vectors, _, spacing = _origin_and_spacing(
        vectors_ap_dv_ml_um, (0.0, 0.0, 0.0), voxel_size_ap_dv_ml_um
    )
    return vectors / spacing


def allen_index_to_physical_um_vectors(
    vectors_ap_dv_ml: torch.Tensor,
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Convert Allen-index displacements to physical displacements."""
    vectors, _, spacing = _origin_and_spacing(
        vectors_ap_dv_ml, (0.0, 0.0, 0.0), voxel_size_ap_dv_ml_um
    )
    return vectors * spacing


def physical_um_plane_to_allen_index_plane(
    normal_ap_dv_ml_um: torch.Tensor,
    signed_offset_um: torch.Tensor,
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``n·(p-q)=d`` from physical units to a unit index covector."""
    normal, _, spacing = _origin_and_spacing(
        normal_ap_dv_ml_um, (0.0, 0.0, 0.0), voxel_size_ap_dv_ml_um
    )
    offset = _finite_floating(
        torch.as_tensor(signed_offset_um, device=normal.device, dtype=normal.dtype), "plane offset"
    )
    if offset.shape != normal.shape[:-1]:
        raise ValueError("Plane offset must have the normal's leading dimensions")
    covector = normal * spacing
    scale = torch.linalg.vector_norm(covector, dim=-1)
    if not bool(torch.isfinite(scale).all()) or bool(
        (scale <= 64.0 * torch.finfo(normal.dtype).eps).any()
    ):
        raise ValueError("Plane normal must be nonzero")
    return covector / scale[..., None], offset / scale


def allen_index_plane_to_physical_um_plane(
    normal_ap_dv_ml: torch.Tensor,
    signed_offset_index: torch.Tensor,
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``n·(i-q)=d`` to a unit physical covector and micrometre offset."""
    normal, _, spacing = _origin_and_spacing(
        normal_ap_dv_ml, (0.0, 0.0, 0.0), voxel_size_ap_dv_ml_um
    )
    offset = _finite_floating(
        torch.as_tensor(signed_offset_index, device=normal.device, dtype=normal.dtype), "plane offset"
    )
    if offset.shape != normal.shape[:-1]:
        raise ValueError("Plane offset must have the normal's leading dimensions")
    covector = normal / spacing
    scale = torch.linalg.vector_norm(covector, dim=-1)
    if not bool(torch.isfinite(scale).all()) or bool(
        (scale <= 64.0 * torch.finfo(normal.dtype).eps).any()
    ):
        raise ValueError("Plane normal must be nonzero")
    return covector / scale[..., None], offset / scale


def allen_to_quicknii_points(
    points_ap_dv_ml: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
) -> torch.Tensor:
    ap, dv, ml = torch.as_tensor(points_ap_dv_ml).unbind(-1)
    return torch.stack((ml, atlas_shape_ap_dv_ml[0] - ap, atlas_shape_ap_dv_ml[1] - dv), -1)


def quicknii_to_allen_points(
    points_ml_ap_dv: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
) -> torch.Tensor:
    ml, ap, dv = torch.as_tensor(points_ml_ap_dv).unbind(-1)
    return torch.stack((atlas_shape_ap_dv_ml[0] - ap, atlas_shape_ap_dv_ml[1] - dv, ml), -1)


def allen_to_quicknii_vectors(vectors_ap_dv_ml: torch.Tensor) -> torch.Tensor:
    ap, dv, ml = torch.as_tensor(vectors_ap_dv_ml).unbind(-1)
    return torch.stack((ml, -ap, -dv), -1)


def quicknii_to_allen_vectors(vectors_ml_ap_dv: torch.Tensor) -> torch.Tensor:
    ml, ap, dv = torch.as_tensor(vectors_ml_ap_dv).unbind(-1)
    return torch.stack((-ap, -dv, ml), -1)


def frame_to_quicknii_ouv(
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
) -> torch.Tensor:
    """Convert a constrained plane to exact flattened QuickNII O/U/V."""
    center, frame, basis = _aligned_state(
        center_ap_dv_ml, frame_ap_dv_ml, inplane_basis
    )
    edges = torch.matmul(frame[..., :, :2], basis)
    u, v = edges.unbind(-1)
    origin = center - 0.5 * (u + v)
    return torch.cat(
        (allen_to_quicknii_points(origin, atlas_shape_ap_dv_ml),
         allen_to_quicknii_vectors(u), allen_to_quicknii_vectors(v)),
        -1,
    )


def quicknii_ouv_to_frame(
    ouv: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly factor non-collinear QuickNII O/U/V into centre, frame, basis."""
    values = _finite_floating(ouv, "QuickNII O/U/V")
    if values.shape[-1:] != (9,):
        raise ValueError("QuickNII O/U/V must end in nine coordinates")
    origin = quicknii_to_allen_points(values[..., :3], atlas_shape_ap_dv_ml)
    edge_u = quicknii_to_allen_vectors(values[..., 3:6])
    edge_v = quicknii_to_allen_vectors(values[..., 6:9])
    return _allen_edges_to_frame(origin + 0.5 * (edge_u + edge_v), edge_u, edge_v)


def normalized_raster_to_ccf(
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
    raster_st: torch.Tensor,
) -> torch.Tensor:
    """Map ``(s,t)`` to CCF; insert singleton state axes before raster grid axes."""
    center, frame, basis = _aligned_state(
        center_ap_dv_ml, frame_ap_dv_ml, inplane_basis
    )
    st = _finite_floating(
        torch.as_tensor(raster_st, device=center.device, dtype=center.dtype), "raster coordinates"
    )
    if st.shape[-1:] != (2,):
        raise ValueError("Raster coordinates must end in (s,t)")
    edges = torch.matmul(frame[..., :, :2], basis)
    return center + torch.matmul(
        edges, (st - 0.5).unsqueeze(-1)
    ).squeeze(-1)


def normalized_raster_to_quicknii(ouv: torch.Tensor, raster_st: torch.Tensor) -> torch.Tensor:
    """Map ``(s,t)`` to QuickNII; use ``ouv[:,None,None]`` for batched grids."""
    values = _finite_floating(ouv, "QuickNII O/U/V")
    if values.shape[-1:] != (9,):
        raise ValueError("QuickNII O/U/V must end in nine coordinates")
    st = _finite_floating(
        torch.as_tensor(raster_st, device=values.device, dtype=values.dtype), "raster coordinates"
    )
    if st.shape[-1:] != (2,):
        raise ValueError("Raster coordinates must end in (s,t)")
    return values[..., :3] + st[..., :1] * values[..., 3:6] + st[..., 1:] * values[..., 6:9]


def horizontal_flip_quicknii_ouv(ouv: torch.Tensor, raster_width: int) -> torch.Tensor:
    """Reparameterize a discrete ``W``-pixel raster after reversing x."""
    if raster_width <= 0:
        raise ValueError("Raster width must be positive")
    values = torch.as_tensor(ouv)
    shift = (raster_width - 1) / raster_width
    return torch.cat(
        (values[..., :3] + shift * values[..., 3:6],
         -values[..., 3:6], values[..., 6:9]),
        -1,
    )


def vertical_flip_quicknii_ouv(ouv: torch.Tensor, raster_height: int) -> torch.Tensor:
    """Reparameterize a discrete ``H``-pixel raster after reversing y."""
    if raster_height <= 0:
        raise ValueError("Raster height must be positive")
    values = torch.as_tensor(ouv)
    shift = (raster_height - 1) / raster_height
    return torch.cat(
        (values[..., :3] + shift * values[..., 6:9],
         values[..., 3:6], -values[..., 6:9]),
        -1,
    )


def flip_frame(
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
    raster_shape: tuple[int, int],
    *,
    horizontal: bool = False,
    vertical: bool = False,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ouv = frame_to_quicknii_ouv(
        center_ap_dv_ml, frame_ap_dv_ml, inplane_basis, atlas_shape_ap_dv_ml
    )
    height, width = raster_shape
    if horizontal:
        ouv = horizontal_flip_quicknii_ouv(ouv, width)
    if vertical:
        ouv = vertical_flip_quicknii_ouv(ouv, height)
    return quicknii_ouv_to_frame(ouv, atlas_shape_ap_dv_ml)


def _allen_edges_to_frame(
    center: torch.Tensor, edge_u: torch.Tensor, edge_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = _finite_floating(center, "plane centre")
    edge_u = _finite_floating(edge_u, "plane U edge")
    edge_v = _finite_floating(edge_v, "plane V edge")
    if center.shape[-1:] != (3,) or edge_u.shape != center.shape or edge_v.shape != center.shape:
        raise ValueError("Plane centre, U, and V must have identical (...,3) shapes")
    if edge_u.device != center.device or edge_v.device != center.device:
        raise ValueError("Plane centre and edges must share one device")
    if edge_u.dtype != center.dtype or edge_v.dtype != center.dtype:
        raise ValueError("Plane centre and edges must share one dtype")
    work = torch.stack((edge_u, edge_v), -1)
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    span_u = torch.linalg.vector_norm(work[..., :, 0], dim=-1)
    span_v = torch.linalg.vector_norm(work[..., :, 1], dim=-1)
    unit_u = work[..., :, 0] / span_u[..., None]
    unit_v = work[..., :, 1] / span_v[..., None]
    sine = torch.linalg.vector_norm(torch.cross(unit_u, unit_v, dim=-1), dim=-1)
    tolerance = 64.0 * torch.finfo(work.dtype).eps
    if not bool(torch.isfinite(span_u).all()) or not bool(torch.isfinite(span_v).all()):
        raise ValueError("Plane edge norms must be finite")
    if bool((span_u <= tolerance).any()) or bool((span_v <= tolerance).any()):
        raise ValueError("Plane U and V edges must be nonzero")
    if not bool(torch.isfinite(sine).all()) or bool((sine <= tolerance).any()):
        raise ValueError("Plane U and V edges must be non-collinear")
    q, basis = torch.linalg.qr(work, mode="reduced")
    diagonal = torch.linalg.diagonal(basis, dim1=-2, dim2=-1)
    sign = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * sign[..., None, :]
    basis = sign[..., :, None] * basis
    u, v = q.unbind(-1)
    frame = torch.stack((u, v, torch.cross(u, v, dim=-1)), -1)
    frame = frame.to(center.dtype)
    basis = basis.to(center.dtype)
    return center, frame, basis


def legacy_renderer_pose_to_frame(
    pose_ap_um_lr_deg_dv_deg: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
    *,
    bregma_ap_index: float = 216.0,
    voxel_um: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt legacy pose using the renderer's inclusive voxel-centre W-1/H-1 spans."""
    pose = torch.as_tensor(pose_ap_um_lr_deg_dv_deg)
    ap_um, tilt_lr, tilt_dv = pose.unbind(-1)
    slope_lr, slope_dv = torch.tan(torch.deg2rad(tilt_lr)), torch.tan(torch.deg2rad(tilt_dv))
    ap_size, dv_size, ml_size = atlas_shape_ap_dv_ml
    center = torch.stack(
        (bregma_ap_index - ap_um / voxel_um, ap_um.new_full(ap_um.shape, (dv_size - 1) / 2),
         ap_um.new_full(ap_um.shape, (ml_size - 1) / 2)),
        -1,
    )
    zero = torch.zeros_like(ap_um)
    edge_u = torch.stack(((ml_size - 1) * slope_lr, zero, zero + ml_size - 1), -1)
    edge_v = torch.stack(((dv_size - 1) * slope_dv, zero + dv_size - 1, zero), -1)
    return _allen_edges_to_frame(center, edge_u, edge_v)


def legacy_quicknii_boundary_pose_to_frame(
    pose_ap_um_lr_deg_dv_deg: torch.Tensor,
    atlas_shape_ap_dv_ml: tuple[int, int, int] = DEFAULT_ALLEN_SHAPE_AP_DV_ML,
    *,
    bregma_ap_index: float = 216.0,
    voxel_um: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt the historical QuickNII W/H boundary-span serialization exactly."""
    pose = torch.as_tensor(pose_ap_um_lr_deg_dv_deg)
    ap_um, tilt_lr, tilt_dv = pose.unbind(-1)
    slope_lr, slope_dv = torch.tan(torch.deg2rad(tilt_lr)), torch.tan(torch.deg2rad(tilt_dv))
    ap_size, dv_size, ml_size = atlas_shape_ap_dv_ml
    ap_index = bregma_ap_index - ap_um / voxel_um
    origin_ap = ap_index - slope_lr * ((ml_size - 1) / 2) - slope_dv * ((dv_size - 1) / 2)
    zero = torch.zeros_like(ap_um)
    origin = torch.stack((origin_ap, zero, zero), -1)
    edge_u = torch.stack((ml_size * slope_lr, zero, zero + ml_size), -1)
    edge_v = torch.stack((dv_size * slope_dv, zero + dv_size, zero), -1)
    center = origin + 0.5 * (edge_u + edge_v)
    return _allen_edges_to_frame(center, edge_u, edge_v)


def render_arbitrary_plane(
    scalar_volume_ap_dv_ml: torch.Tensor,
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
    output_shape: tuple[int, int],
    labels_ap_dv_ml: torch.Tensor | None = None,
    *,
    sampling_contract: str = QUICKNII_RASTER_INDEX_SAMPLING,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Render aligned states; QuickNII pixels use ``s=x/W`` and ``t=y/H``."""
    volume = torch.as_tensor(scalar_volume_ap_dv_ml)
    if volume.ndim != 3 or not torch.is_floating_point(volume):
        raise ValueError("Scalar volume must be one floating-point AP-DV-ML array")
    center = torch.as_tensor(center_ap_dv_ml, device=volume.device, dtype=volume.dtype)
    frame = torch.as_tensor(frame_ap_dv_ml, device=volume.device, dtype=volume.dtype)
    basis = torch.as_tensor(inplane_basis, device=volume.device, dtype=volume.dtype)
    center, frame, basis = _aligned_state(center, frame, basis)
    if center.ndim == 1:
        center, frame, basis = center[None], frame[None], basis[None]
    elif center.ndim != 2:
        raise ValueError("Renderer state must be unbatched or have one aligned batch dimension")
    batch = center.shape[0]
    height, width = output_shape
    if sampling_contract == QUICKNII_RASTER_INDEX_SAMPLING:
        s = torch.arange(width, device=volume.device, dtype=volume.dtype) / width
        t = torch.arange(height, device=volume.device, dtype=volume.dtype) / height
    elif sampling_contract == LEGACY_INCLUSIVE_CENTRE_SAMPLING:
        s = torch.linspace(0.0, 1.0, width, device=volume.device, dtype=volume.dtype)
        t = torch.linspace(0.0, 1.0, height, device=volume.device, dtype=volume.dtype)
    else:
        raise ValueError(f"Unknown plane sampling contract: {sampling_contract}")
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    st = torch.stack((ss, tt), -1)[None].expand(batch, -1, -1, -1)
    points = normalized_raster_to_ccf(
        center[:, None, None], frame[:, None, None], basis[:, None, None], st
    )
    depth, native_height, native_width = volume.shape
    grid = torch.stack(
        (points[..., 2] / (native_width - 1) * 2 - 1,
         points[..., 1] / (native_height - 1) * 2 - 1,
         points[..., 0] / (depth - 1) * 2 - 1),
        -1,
    )[:, None]
    image = F.grid_sample(
        volume[None, None].expand(batch, -1, -1, -1, -1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, :, 0]
    if labels_ap_dv_ml is None:
        return image, None
    labels = torch.as_tensor(labels_ap_dv_ml, device=volume.device)
    if labels.shape != volume.shape:
        raise ValueError("Label and scalar atlas volumes must have identical AP-DV-ML shapes")
    indices = torch.round(points).long()
    valid = (
        (indices[..., 0] >= 0) & (indices[..., 0] < depth)
        & (indices[..., 1] >= 0) & (indices[..., 1] < native_height)
        & (indices[..., 2] >= 0) & (indices[..., 2] < native_width)
    )
    clipped = torch.stack(
        (indices[..., 0].clamp(0, depth - 1),
         indices[..., 1].clamp(0, native_height - 1),
         indices[..., 2].clamp(0, native_width - 1)),
        -1,
    )
    sampled_labels = labels[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
    return image, torch.where(valid, sampled_labels, torch.zeros_like(sampled_labels))[:, None]


def render_legacy_inclusive_plane(
    scalar_volume_ap_dv_ml: torch.Tensor,
    center_ap_dv_ml: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis: torch.Tensor,
    output_shape: tuple[int, int],
    labels_ap_dv_ml: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Render the frozen W-1/H-1 plane on inclusive normalized coordinates."""
    return render_arbitrary_plane(
        scalar_volume_ap_dv_ml, center_ap_dv_ml, frame_ap_dv_ml,
        inplane_basis, output_shape, labels_ap_dv_ml,
        sampling_contract=LEGACY_INCLUSIVE_CENTRE_SAMPLING,
    )
