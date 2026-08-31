from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST_SCHEMA = "anatomy-tracker.synthetic-arbitrary-plane-manifest/v3"
SAMPLER_ALGORITHM = "uniform-rp2-support-projection/v1"
CANDIDATE_SCHEMA = "anatomy-tracker.arbitrary-plane-candidates/v1"
REFERENCE_STRATUM = "reference"
STRESS_STRATUM = "stress"
RNG_FIELDS = ("stratum", "normal", "roll", "stress_side", "offset")
UINT64_SEED_ENCODING = "canonical-lowercase-uint64-hex/v1"
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    array = np.asarray(array)
    dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": dtype.str, "shape": list(array.shape)}).encode("utf-8"))
    digest.update(memoryview(normalized).cast("B"))
    return digest.hexdigest()


def _uint64_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return seed


def _seed_hex(seed: int) -> str:
    return f"0x{_uint64_seed(seed):016x}"


def _parse_seed_hex(seed: object) -> int:
    if (
        not isinstance(seed, str)
        or len(seed) != 18
        or not seed.startswith("0x")
        or any(character not in "0123456789abcdef" for character in seed[2:])
    ):
        raise ValueError("Serialized seeds must use canonical lowercase 0x plus 16 hex digits")
    return int(seed[2:], 16)


def _generator_source_sha256() -> str:
    return _LOADED_SOURCE_SHA256


def _source_commit(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value)
    if len(value) not in {40, 64} or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("generator_source_commit must be a full lowercase hexadecimal Git object id")
    return value


def _generator_receipt(resolved_config: dict[str, object], source_commit: str | None) -> dict[str, object]:
    return {
        "implementation": {
            "source_path": "training/arbitrary_plane_manifest.py",
            "source_sha256": _generator_source_sha256(),
            "source_commit": _source_commit(source_commit),
            "source_commit_semantics": (
                "caller-supplied commit containing this source, or null when the exact source-file SHA-256 is authoritative"
            ),
        },
        "resolved_config": resolved_config,
        "resolved_config_sha256": _payload_sha256(resolved_config),
    }


def _validate_generator_receipt(receipt: dict[str, object], source_commit: str | None) -> None:
    if receipt["resolved_config_sha256"] != _payload_sha256(receipt["resolved_config"]):
        raise ValueError("Generator resolved config does not match resolved_config_sha256")
    implementation = receipt["implementation"]
    if implementation["source_sha256"] != _generator_source_sha256():
        raise ValueError("Generator source does not match the recorded source_sha256")
    if implementation["source_commit"] != _source_commit(source_commit):
        raise ValueError("Generator source commit does not match the expected source commit")


def _derived_seed(seed: int, field: str, index: int | None = None) -> int:
    payload = f"{SAMPLER_ALGORITHM}\0{_uint64_seed(seed)}\0{field}"
    if index is not None:
        payload += f"\0{int(index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _rng(seed: int, field: str, index: int | None = None) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_derived_seed(seed, field, index)))


def canonicalize_rp2_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape[-1] != 3:
        raise ValueError("Normals must have a final dimension of three")
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm == 0.0) or not np.isfinite(normal).all() or not np.isfinite(norm).all():
        raise ValueError("Normals must be finite and nonzero")
    unit = normal / norm
    pivot = np.argmax(np.abs(unit), axis=-1)
    pivot_value = np.take_along_axis(unit, np.expand_dims(pivot, -1), axis=-1)[..., 0]
    canonical = unit * np.where(pivot_value < 0.0, -1.0, 1.0)[..., None]
    return np.where(canonical == 0.0, 0.0, canonical)


def canonicalize_plane(normal: np.ndarray, signed_offset_um: float) -> tuple[np.ndarray, float, int]:
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if (
        normal.shape != (3,)
        or norm == 0.0
        or not np.isfinite(normal).all()
        or not np.isfinite(norm)
        or not np.isfinite(signed_offset_um)
    ):
        raise ValueError("Plane normal must be a finite nonzero 3-vector")
    unit = normal / norm
    pivot = int(np.argmax(np.abs(unit)))
    sign = -1 if unit[pivot] < 0.0 else 1
    canonical = np.where(sign * unit == 0.0, 0.0, sign * unit)
    offset = float(sign * signed_offset_um / norm)
    return canonical, 0.0 if offset == 0.0 else offset, sign


def build_annotation_support(
    annotation: np.ndarray,
    *,
    atlas_id: str,
    atlas_version: str,
    source_uri: str,
    voxel_size_um: tuple[float, float, float],
    source_entity_type: str,
    coordinate_axis_directions: tuple[str, str, str],
    origin_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    projection_origin_um: tuple[float, float, float] | None = None,
    coordinate_axes: tuple[str, str, str] = ("AP", "DV", "ML"),
    source_sha256: str | None = None,
) -> dict[str, object]:
    annotation = np.asarray(annotation)
    if annotation.ndim != 3:
        raise ValueError("Annotation must be a three-dimensional array")
    mask = annotation != 0
    if not mask.any():
        raise ValueError("Annotation support is empty")
    spacing = np.asarray(voxel_size_um, dtype=np.float64)
    origin = np.asarray(origin_um, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0.0) or not np.isfinite(spacing).all():
        raise ValueError("voxel_size_um must contain three positive values")
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or len(coordinate_axes) != 3
        or len(coordinate_axis_directions) != 3
    ):
        raise ValueError("origin_um, coordinate_axes and coordinate_axis_directions must contain three values")
    if not str(source_entity_type):
        raise ValueError("source_entity_type must be nonempty")
    projection_origin = (
        origin + np.asarray(annotation.shape, dtype=np.float64) * spacing / 2.0
        if projection_origin_um is None
        else np.asarray(projection_origin_um, dtype=np.float64)
    )
    if projection_origin.shape != (3,) or not np.isfinite(projection_origin).all():
        raise ValueError("projection_origin_um must contain three finite values")

    annotation_sha256 = _array_sha256(annotation)
    packed_mask = np.packbits(mask.reshape(-1), bitorder="little")
    mask_digest = hashlib.sha256()
    mask_digest.update(_canonical_json({"shape": list(mask.shape), "bitorder": "little"}).encode("utf-8"))
    mask_digest.update(packed_mask.tobytes())
    support_mask_sha256 = mask_digest.hexdigest()
    if source_sha256 is None:
        raise ValueError("source_sha256 must explicitly identify the raw bytes at source_uri")
    source_sha256 = str(source_sha256).lower()
    if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
        raise ValueError("source_sha256 must be a 64-digit hexadecimal SHA-256 digest")
    points_um = origin + (np.argwhere(mask).astype(np.float64) + 0.5) * spacing
    identity = {
        "atlas": {
            "id": str(atlas_id),
            "version": str(atlas_version),
            "coordinate_axes": list(coordinate_axes),
            "coordinate_axis_directions": list(coordinate_axis_directions),
            "coordinate_unit": "um",
        },
        "source": {
            "source_entity_type": str(source_entity_type),
            "annotation_uri": str(source_uri),
            "source_sha256": source_sha256,
            "source_sha256_semantics": "sha256-of-raw-bytes-at-source-uri",
            "annotation_array_sha256": annotation_sha256,
        },
        "annotation_shape": list(annotation.shape),
        "voxel_size_um": spacing.tolist(),
        "origin_um": origin.tolist(),
        "projection_origin_um": projection_origin.tolist(),
        "occupied_voxel_count": int(mask.sum()),
        "occupied_points_um_sha256": _array_sha256(points_um),
        "support_mask_sha256": support_mask_sha256,
        "voxel_coordinate_semantics": "origin-plus-index-plus-half-voxel",
        "projection_backend": "exact-all-occupied-voxel-centres/v1",
        "projection_scaling_status": "prototype-only; not approved for full-Allen-atlas generation",
        "projection_scaling_limit": (
            "materializes float64 coordinates for every occupied voxel and scans all occupied voxels "
            "for every plane; a compact chunked/batched exact backend is required before full-scale training"
        ),
    }
    support_sha256 = _payload_sha256({"schema": "annotation-support/v1", **identity})
    points_um.setflags(write=False)
    return {**identity, "support_sha256": support_sha256, "points_um": points_um}


def _support_metadata(support: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in support.items() if key != "points_um"}


def _validate_annotation_support(support: dict[str, object]) -> None:
    metadata = _support_metadata(support)
    supplied_hash = metadata.pop("support_sha256", None)
    if supplied_hash != _payload_sha256({"schema": "annotation-support/v1", **metadata}):
        raise ValueError("Annotation support metadata does not match support_sha256")
    points = np.asarray(support["points_um"], dtype=np.float64)
    if points.shape != (int(support["occupied_voxel_count"]), 3):
        raise ValueError("Annotation support points do not match occupied_voxel_count")
    if _array_sha256(points) != support["occupied_points_um_sha256"]:
        raise ValueError("Annotation support points do not match occupied_points_um_sha256")


def support_projection(normal: np.ndarray, support: dict[str, object]) -> dict[str, object]:
    normal = canonicalize_rp2_normal(np.asarray(normal, dtype=np.float64))
    if normal.shape != (3,):
        raise ValueError("support_projection accepts one normal")
    points = np.asarray(support["points_um"], dtype=np.float64)
    projection_origin = np.asarray(support["projection_origin_um"], dtype=np.float64)
    half_extent = float(np.dot(np.abs(normal), np.asarray(support["voxel_size_um"], dtype=np.float64) / 2.0))
    projected = (points - projection_origin) @ normal
    lower = float(projected.min() - half_extent)
    upper = float(projected.max() + half_extent)
    payload = {
        "support_sha256": str(support["support_sha256"]),
        "normal": normal.tolist(),
        "projection_origin_um": projection_origin.tolist(),
        "bounds_um": [lower, upper],
        "directional_voxel_half_extent_um": half_extent,
    }
    return {**payload, "projection_sha256": _payload_sha256(payload)}


def plane_intersects_annotation_support(
    normal: np.ndarray,
    signed_offset_um: float,
    support: dict[str, object],
) -> tuple[bool, int]:
    normal, signed_offset_um, _ = canonicalize_plane(normal, signed_offset_um)
    points = np.asarray(support["points_um"], dtype=np.float64)
    projection_origin = np.asarray(support["projection_origin_um"], dtype=np.float64)
    half_extent = float(np.dot(np.abs(normal), np.asarray(support["voxel_size_um"], dtype=np.float64) / 2.0))
    hits = int(
        np.count_nonzero(
            np.abs((points - projection_origin) @ normal - signed_offset_um) <= half_extent + 1e-10
        )
    )
    return hits > 0, hits


def _sample_uniform_rp2(seed: int, index: int) -> np.ndarray:
    generator = _rng(seed, "normal", index)
    return canonicalize_rp2_normal(generator.normal(0.0, 1.0, 3))


def sample_uniform_rp2_normals(count: int, seed: int) -> np.ndarray:
    return np.stack([_sample_uniform_rp2(seed, index) for index in range(count)])


def sample_uniform_rolls(count: int, seed: int) -> np.ndarray:
    return np.asarray([_rng(seed, "roll", index).uniform(0.0, 2.0 * np.pi) for index in range(count)])


def _sample_offset(
    normal: np.ndarray,
    stratum: str,
    support: dict[str, object],
    seed: int,
    index: int,
    reference_fraction_bounds: tuple[float, float],
    stress_boundary_fraction: float,
    max_rejection_attempts: int,
) -> tuple[float, float, int, int, dict[str, object], str | None]:
    projection = support_projection(normal, support)
    lower, upper = projection["bounds_um"]
    generator = _rng(seed, "offset", index)
    stress_side = None
    if stratum == STRESS_STRATUM:
        stress_side = "lower" if int(_rng(seed, "stress_side", index).integers(0, 2)) == 0 else "upper"
    for attempt in range(1, max_rejection_attempts + 1):
        if stratum == REFERENCE_STRATUM:
            fraction = generator.uniform(*reference_fraction_bounds)
        else:
            edge_fraction = generator.uniform(0.0, stress_boundary_fraction)
            fraction = edge_fraction if stress_side == "lower" else 1.0 - edge_fraction
        offset = float(lower + fraction * (upper - lower))
        intersects, hit_count = plane_intersects_annotation_support(normal, offset, support)
        if intersects:
            return offset, float(fraction), attempt - 1, hit_count, projection, stress_side
    raise RuntimeError(f"Could not sample an intersecting {stratum} plane in {max_rejection_attempts} attempts")


def make_arbitrary_plane_manifest(
    count: int,
    split: str,
    seed: int,
    support: dict[str, object],
    *,
    stress_fraction: float = 0.20,
    reference_fraction_bounds: tuple[float, float] = (0.0, 1.0),
    stress_boundary_fraction: float = 0.10,
    animal_id: str | int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    max_rejection_attempts: int = 4096,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    count = int(count)
    split = str(split)
    seed = _uint64_seed(seed)
    stress_fraction = float(stress_fraction)
    reference_fraction_bounds = tuple(float(value) for value in reference_fraction_bounds)
    stress_boundary_fraction = float(stress_boundary_fraction)
    animal_id = _python_scalar(animal_id)
    specimen_id = _python_scalar(specimen_id)
    experiment_id = _python_scalar(experiment_id)
    max_rejection_attempts = int(max_rejection_attempts)
    generator_source_commit = _source_commit(generator_source_commit)
    if count <= 0:
        raise ValueError("count must be positive")
    if split not in {"train", "development"}:
        raise ValueError("Development-stage manifests permit only train or development splits")
    if not 0.0 <= stress_fraction <= 1.0:
        raise ValueError("stress_fraction must lie in [0, 1]")
    if not 0.0 <= reference_fraction_bounds[0] < reference_fraction_bounds[1] <= 1.0:
        raise ValueError("reference_fraction_bounds must be ordered inside [0, 1]")
    if tuple(reference_fraction_bounds) != (0.0, 1.0):
        raise ValueError("The frozen reference measure spans the full support projection [0, 1]")
    if not 0.0 < stress_boundary_fraction <= 0.5:
        raise ValueError("stress_boundary_fraction must lie in (0, 0.5]")

    _validate_annotation_support(support)
    split_domain_seed = _derived_seed(seed, f"split:{split}")
    resolved_config = {
        "schema_version": MANIFEST_SCHEMA,
        "sampler_algorithm": SAMPLER_ALGORITHM,
        "count": count,
        "split": split,
        "root_seed": _seed_hex(seed),
        "support_sha256": support["support_sha256"],
        "stress_fraction": stress_fraction,
        "reference_fraction_bounds": list(reference_fraction_bounds),
        "stress_boundary_fraction": stress_boundary_fraction,
        "animal_id": animal_id,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "max_rejection_attempts": max_rejection_attempts,
        "numpy_version": np.__version__,
    }
    generator_receipt = _generator_receipt(resolved_config, generator_source_commit)
    stress_count = int(np.floor(count * stress_fraction + 0.5))
    strata = np.full(count, REFERENCE_STRATUM, dtype=object)
    stress_indices = _rng(split_domain_seed, "stratum").permutation(count)[:stress_count]
    strata[stress_indices] = STRESS_STRATUM
    normals = sample_uniform_rp2_normals(count, split_domain_seed)
    rolls = sample_uniform_rolls(count, split_domain_seed)
    samples = []
    for index in range(count):
        normal = normals[index]
        roll_rad = float(rolls[index])
        offset, offset_fraction, rejections, hits, projection, stress_side = _sample_offset(
            normal,
            str(strata[index]),
            support,
            split_domain_seed,
            index,
            reference_fraction_bounds,
            stress_boundary_fraction,
            max_rejection_attempts,
        )
        sample_payload = {
            "sample_index": index,
            "split": split,
            "animal_id": animal_id,
            "specimen_id": specimen_id,
            "experiment_id": experiment_id,
            "stratum": str(strata[index]),
            "normal_rp2": normal.tolist(),
            "signed_offset_um": offset,
            "roll_rad": roll_rad,
            "offset_fraction_of_support_projection": offset_fraction,
            "stress_projection_side": stress_side,
            "rng": {
                "field_stream_index": index,
                "field_stream_seed_uint64": {
                    field: _seed_hex(_derived_seed(split_domain_seed, field, index))
                    for field in RNG_FIELDS
                    if field != "stratum"
                },
                "offset_rejections": rejections,
                "offset_attempts_total": rejections + 1,
            },
            "support": {
                "support_sha256": support["support_sha256"],
                "projection_sha256": projection["projection_sha256"],
                "projection_origin_um": projection["projection_origin_um"],
                "projection_bounds_um": projection["bounds_um"],
                "directional_voxel_half_extent_um": projection["directional_voxel_half_extent_um"],
                "intersecting_voxel_count": hits,
            },
        }
        realization_payload = {
            "schema": "synthetic-plane-realization/v1",
            "sampler": SAMPLER_ALGORITHM,
            "support_sha256": support["support_sha256"],
            "root_seed": _seed_hex(seed),
            "generator_source_sha256": generator_receipt["implementation"]["source_sha256"],
            "resolved_config_sha256": generator_receipt["resolved_config_sha256"],
            **sample_payload,
        }
        geometry_payload = {
            "schema": "synthetic-plane-geometry/v1",
            "support_sha256": support["support_sha256"],
            "normal_rp2": sample_payload["normal_rp2"],
            "signed_offset_um": sample_payload["signed_offset_um"],
            "roll_rad": sample_payload["roll_rad"],
        }
        samples.append(
            {
                **sample_payload,
                "plane_geometry_sha256": _payload_sha256(geometry_payload),
                "plane_realization_id": _payload_sha256(realization_payload),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "sampler_algorithm": SAMPLER_ALGORITHM,
        "split": split,
        "count": count,
        "root_seed": _seed_hex(seed),
        "generator": generator_receipt,
        "rng": {
            "bit_generator": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "seed_encoding": UINT64_SEED_ENCODING,
            "split_domain_separation": "sha256(algorithm, root_seed, split:<split>) first 64 bits little-endian",
            "split_domain_seed_uint64": _seed_hex(split_domain_seed),
            "field_stream_derivation": "sha256(algorithm, split_domain_seed, field[, sample_index]) first 64 bits little-endian",
            "global_stream_seed_uint64": {
                "stratum": _seed_hex(_derived_seed(split_domain_seed, "stratum"))
            },
            "per_sample_stream_fields": [field for field in RNG_FIELDS if field != "stratum"],
        },
        "provenance": {
            **_support_metadata(support),
            "animal_id": animal_id,
            "specimen_id": specimen_id,
            "experiment_id": experiment_id,
        },
        "sampling": {
            "identifier_contract": {
                "plane_realization_id": "identifies only this replayable plane draw and its recorded provenance",
                "synthetic_realization_id": (
                    "reserved and intentionally absent until a record binds the full frame, in-plane basis, "
                    "QuickNII O/U/V, deformation, appearance, mask and rendered artifacts"
                ),
            },
            "normal_distribution": "normalize g~N(0,I3) for surface-area-uniform S2, then deterministic antipodal fold to RP2",
            "reference_measure": (
                "orientation-balanced: Haar-uniform RP2 normal, then Lebesgue-uniform signed offset "
                "conditional on the projected union of occupied voxel boxes; this is not the Crofton "
                "measure, whose orientation marginal is proportional to projection width"
            ),
            "offset_sampling": (
                "uniform proposal on the min/max projection envelope followed by rejection unless the "
                "infinite plane intersects an occupied voxel box"
            ),
            "roll_distribution": "uniform on [0, 2pi)",
            "roll_semantics": "independent draw only; finite-raster frame construction and transport are not implemented in this plane manifest",
            "stress_fraction": stress_fraction,
            "strata": {
                REFERENCE_STRATUM: {
                    "offset_fraction_bounds": list(reference_fraction_bounds),
                    "description": "frozen reference measure: full-RP2 normals and the full brain-support projection",
                },
                STRESS_STRATUM: {
                    "lower_offset_fraction_bounds": [0.0, stress_boundary_fraction],
                    "upper_offset_fraction_bounds": [1.0 - stress_boundary_fraction, 1.0],
                    "description": "deliberate peripheral/grazing edge oversampling layered on the full reference measure",
                    "side_distribution": "one frozen lower/upper Bernoulli(0.5) draw per stress sample before offset rejection",
                },
            },
            "max_rejection_attempts": max_rejection_attempts,
            "intersection_contract": "plane intersects at least one occupied annotation voxel box",
            "finite_raster_support_status": (
                "not evaluated here; after frame, in-plane centre and basis exist, rendered tissue "
                "support must be checked before a sample is training-eligible"
            ),
        },
        "samples": samples,
    }
    return {**manifest, "manifest_sha256": _payload_sha256(manifest)}


def replay_arbitrary_plane_manifest(
    manifest: dict[str, object],
    support: dict[str, object],
    *,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    _validate_annotation_support(support)
    supplied_hash = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied_hash != _payload_sha256(unhashed):
        raise ValueError("Manifest content does not match manifest_sha256")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["sampler_algorithm"] != SAMPLER_ALGORITHM:
        raise ValueError("Unsupported arbitrary-plane manifest version")
    _validate_generator_receipt(manifest["generator"], generator_source_commit)
    if manifest["provenance"]["support_sha256"] != support["support_sha256"]:
        raise ValueError("Annotation support does not match the manifest")
    sampling = manifest["sampling"]
    reference_bounds = tuple(sampling["strata"][REFERENCE_STRATUM]["offset_fraction_bounds"])
    stress_bounds = sampling["strata"][STRESS_STRATUM]["lower_offset_fraction_bounds"]
    replayed = make_arbitrary_plane_manifest(
        int(manifest["count"]),
        str(manifest["split"]),
        _parse_seed_hex(manifest["root_seed"]),
        support,
        stress_fraction=float(sampling["stress_fraction"]),
        reference_fraction_bounds=reference_bounds,
        stress_boundary_fraction=float(stress_bounds[1]),
        animal_id=manifest["provenance"]["animal_id"],
        specimen_id=manifest["provenance"]["specimen_id"],
        experiment_id=manifest["provenance"]["experiment_id"],
        max_rejection_attempts=int(sampling["max_rejection_attempts"]),
        generator_source_commit=generator_source_commit,
    )
    if replayed["manifest_sha256"] != supplied_hash:
        raise ValueError("Manifest replay did not reproduce the recorded hash")
    return replayed


def save_arbitrary_plane_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    Path(path).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")


def load_arbitrary_plane_manifest(path: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied_hash = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied_hash != _payload_sha256(unhashed):
        raise ValueError("Manifest content does not match manifest_sha256")
    return manifest


def _verified_manifest_sample(
    manifest: dict[str, object],
    support: dict[str, object],
    sample_index: int,
    generator_source_commit: str | None,
) -> dict[str, object]:
    verified = replay_arbitrary_plane_manifest(
        manifest,
        support,
        generator_source_commit=generator_source_commit,
    )
    samples = verified["samples"]
    if not 0 <= int(sample_index) < len(samples):
        raise IndexError("sample_index is outside the verified manifest")
    sample = samples[int(sample_index)]
    if int(sample["sample_index"]) != int(sample_index) or sample["split"] != verified["split"]:
        raise ValueError("Manifest sample index or split is inconsistent")
    if sample["support"]["support_sha256"] != support["support_sha256"]:
        raise ValueError("Manifest sample support is inconsistent")
    return sample


def rp2_geodesic_plane_delta(
    base_normal: np.ndarray,
    base_signed_offset_um: float,
    candidate_normal: np.ndarray,
    candidate_signed_offset_um: float,
) -> dict[str, object]:
    base_normal, base_signed_offset_um, _ = canonicalize_plane(base_normal, base_signed_offset_um)
    candidate_normal, candidate_signed_offset_um, _ = canonicalize_plane(
        candidate_normal, candidate_signed_offset_um
    )
    alignment_sign = -1 if float(np.dot(base_normal, candidate_normal)) < 0.0 else 1
    aligned_normal = alignment_sign * candidate_normal
    aligned_offset = alignment_sign * candidate_signed_offset_um
    cosine = float(np.clip(np.dot(base_normal, aligned_normal), -1.0, 1.0))
    sine = float(np.linalg.norm(np.cross(base_normal, aligned_normal)))
    angle = float(np.arctan2(sine, cosine))
    if sine < 1e-12:
        tangent = np.zeros(3, dtype=np.float64)
    else:
        direction = aligned_normal - cosine * base_normal
        direction -= np.dot(direction, base_normal) * base_normal
        direction /= np.linalg.norm(direction)
        tangent = angle * direction
    return {
        "normal_delta_logmap_ap_dv_ml_rad": tangent.tolist(),
        "normal_geodesic_rad": angle,
        "offset_delta_um": float(aligned_offset - base_signed_offset_um),
        "rp2_alignment_sign_to_base": alignment_sign,
    }


def make_brain_intersecting_candidates(
    manifest: dict[str, object],
    sample_index: int,
    support: dict[str, object],
    count: int,
    seed: int,
    *,
    max_geodesic_deg: float,
    max_offset_um: float,
    include_center: bool = True,
    max_rejection_attempts: int = 4096,
    generator_source_commit: str | None = None,
    manifest_generator_source_commit: str | None = None,
) -> dict[str, object]:
    sample_index = int(sample_index)
    count = int(count)
    seed = _uint64_seed(seed)
    max_geodesic_deg = float(max_geodesic_deg)
    max_offset_um = float(max_offset_um)
    include_center = bool(include_center)
    max_rejection_attempts = int(max_rejection_attempts)
    generator_source_commit = _source_commit(generator_source_commit)
    manifest_generator_source_commit = _source_commit(manifest_generator_source_commit)
    if count <= 0:
        raise ValueError("count must be positive")
    sample = _verified_manifest_sample(
        manifest, support, sample_index, manifest_generator_source_commit
    )
    if not 0.0 <= max_geodesic_deg <= 90.0 or max_offset_um < 0.0:
        raise ValueError("Candidate radii must be nonnegative and geodesic radius at most 90 degrees")
    resolved_config = {
        "schema_version": CANDIDATE_SCHEMA,
        "sampler_algorithm": SAMPLER_ALGORITHM,
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_index": sample_index,
        "plane_realization_id": sample["plane_realization_id"],
        "support_sha256": support["support_sha256"],
        "count": count,
        "root_seed": _seed_hex(seed),
        "max_geodesic_deg": max_geodesic_deg,
        "max_offset_um": max_offset_um,
        "include_center": include_center,
        "max_rejection_attempts": max_rejection_attempts,
        "manifest_generator_source_commit": manifest_generator_source_commit,
        "numpy_version": np.__version__,
    }
    generator_receipt = _generator_receipt(resolved_config, generator_source_commit)
    base_normal = np.asarray(sample["normal_rp2"], dtype=np.float64)
    base_offset = float(sample["signed_offset_um"])
    max_angle = np.deg2rad(max_geodesic_deg)
    candidates = []
    pose_hashes = set()
    center_seed = _derived_seed(seed, f"candidate-center:{sample['plane_realization_id']}")
    center_candidate_index = (
        int(np.random.Generator(np.random.PCG64(center_seed)).integers(0, count))
        if include_center
        else None
    )
    for index in range(count):
        normal_seed = _derived_seed(seed, f"candidate-normal:{sample['plane_realization_id']}", index)
        offset_seed = _derived_seed(seed, f"candidate-offset:{sample['plane_realization_id']}", index)
        if index == center_candidate_index:
            canonical_normal = base_normal.copy()
            canonical_offset = base_offset
            rejections = 0
        else:
            normal_generator = np.random.Generator(np.random.PCG64(normal_seed))
            offset_generator = np.random.Generator(np.random.PCG64(offset_seed))
            for attempt in range(1, max_rejection_attempts + 1):
                cosine = normal_generator.uniform(np.cos(max_angle), 1.0)
                angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
                raw_direction = normal_generator.normal(0.0, 1.0, 3)
                direction = raw_direction - np.dot(raw_direction, base_normal) * base_normal
                direction /= np.linalg.norm(direction)
                pre_normal = np.cos(angle) * base_normal + np.sin(angle) * direction
                pre_offset = base_offset + offset_generator.uniform(-max_offset_um, max_offset_um)
                canonical_normal, canonical_offset, _ = canonicalize_plane(pre_normal, pre_offset)
                intersects, _ = plane_intersects_annotation_support(canonical_normal, canonical_offset, support)
                pose_hash = _payload_sha256(
                    {"normal_rp2": canonical_normal.tolist(), "signed_offset_um": canonical_offset}
                )
                if intersects and pose_hash not in pose_hashes:
                    rejections = attempt - 1
                    break
            else:
                raise RuntimeError(f"Could not sample intersecting candidate {index}")

        intersects, hit_count = plane_intersects_annotation_support(canonical_normal, canonical_offset, support)
        projection = support_projection(canonical_normal, support)
        pose_hash = _payload_sha256(
            {"normal_rp2": canonical_normal.tolist(), "signed_offset_um": canonical_offset}
        )
        if pose_hash in pose_hashes:
            raise RuntimeError(f"Candidate {index} duplicates an earlier plane")
        pose_hashes.add(pose_hash)
        delta = rp2_geodesic_plane_delta(
            base_normal, base_offset, canonical_normal, canonical_offset
        )
        candidate_payload = {
            "candidate_index": index,
            **delta,
            "normal_rp2": canonical_normal.tolist(),
            "signed_offset_um": canonical_offset,
            "brain_intersection": bool(intersects),
            "intersecting_voxel_count": hit_count,
            "support_projection_sha256": projection["projection_sha256"],
            "plane_pose_sha256": pose_hash,
            "rng": {
                "normal_stream_seed_uint64": _seed_hex(normal_seed),
                "offset_stream_seed_uint64": _seed_hex(offset_seed),
                "rejection_attempts": rejections,
                "attempts_total": rejections + 1,
            },
        }
        candidate_id = _payload_sha256(
            {
                "schema": "arbitrary-plane-candidate/v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "plane_realization_id": sample["plane_realization_id"],
                **candidate_payload,
            }
        )
        candidates.append({**candidate_payload, "candidate_id": candidate_id})

    candidate_set = {
        "schema_version": CANDIDATE_SCHEMA,
        "scope": "unoriented-infinite-plane-only",
        "representation": "coordinate-free RP2 ambient log map in AP/DV/ML radians plus aligned physical signed-offset delta (um)",
        "finite_raster_frame_status": (
            "not implemented; minimally rotate/parallel-transport the verified base frame to each candidate "
            "normal before adding an explicit in-plane roll delta"
        ),
        "excluded_degrees_of_freedom": [
            "in_plane_roll",
            "in_plane_center",
            "in_plane_basis",
            "raster_reflection_state",
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_index": sample_index,
        "plane_realization_id": sample["plane_realization_id"],
        "support_sha256": support["support_sha256"],
        "root_seed": _seed_hex(seed),
        "generator": generator_receipt,
        "rng": {
            "bit_generator": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "seed_encoding": UINT64_SEED_ENCODING,
            "seed_namespace": SAMPLER_ALGORITHM,
            "field_stream_derivation": "sha256(algorithm, root_seed, candidate field plus realization id[, candidate index])",
            "independent_fields": ["candidate-center", "candidate-normal", "candidate-offset"],
            "center_stream_seed_uint64": _seed_hex(center_seed),
        },
        "count": count,
        "include_center": include_center,
        "center_candidate_index": center_candidate_index,
        "max_geodesic_deg": max_geodesic_deg,
        "max_offset_um": max_offset_um,
        "max_rejection_attempts": max_rejection_attempts,
        "posterior_use_status": (
            "proposal set only; normalized candidate scores are not calibrated posterior mass until "
            "proposal density/truncation and the inserted center candidate are accounted for"
        ),
        "candidates": candidates,
    }
    return {**candidate_set, "candidate_set_sha256": _payload_sha256(candidate_set)}


def replay_brain_intersecting_candidates(
    candidate_set: dict[str, object],
    manifest: dict[str, object],
    support: dict[str, object],
    *,
    generator_source_commit: str | None = None,
    manifest_generator_source_commit: str | None = None,
) -> dict[str, object]:
    supplied_hash = candidate_set.get("candidate_set_sha256")
    unhashed = {key: value for key, value in candidate_set.items() if key != "candidate_set_sha256"}
    if supplied_hash != _payload_sha256(unhashed):
        raise ValueError("Candidate content does not match candidate_set_sha256")
    if candidate_set["schema_version"] != CANDIDATE_SCHEMA:
        raise ValueError("Unsupported arbitrary-plane candidate version")
    _validate_generator_receipt(candidate_set["generator"], generator_source_commit)
    if candidate_set["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise ValueError("Candidate set does not match the verified manifest")
    replayed = make_brain_intersecting_candidates(
        manifest,
        int(candidate_set["sample_index"]),
        support,
        int(candidate_set["count"]),
        _parse_seed_hex(candidate_set["root_seed"]),
        max_geodesic_deg=float(candidate_set["max_geodesic_deg"]),
        max_offset_um=float(candidate_set["max_offset_um"]),
        include_center=bool(candidate_set["include_center"]),
        max_rejection_attempts=int(candidate_set["max_rejection_attempts"]),
        generator_source_commit=generator_source_commit,
        manifest_generator_source_commit=manifest_generator_source_commit,
    )
    if replayed["candidate_set_sha256"] != supplied_hash:
        raise ValueError("Candidate replay did not reproduce the recorded hash")
    return replayed
