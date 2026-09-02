"""Content-addressed I:-drive persistence for authenticated v3 prepared sections."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from types import MappingProxyType

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_training_data_v3 as training_data_v3


PREPARED_SECTION_CACHE_V3_SCHEMA = (
    "anatomy-tracker.prepared-training-section-cache-entry/v3"
)
PREPARED_SECTION_CACHE_V3_ALGORITHM = (
    "content-addressed-json-npz-byte-exact-prepared-parent/v3"
)
PREPARED_SECTION_NAMESPACE = "prepared_sections"
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_prepared_section_cache_v3.py",
    "arbitrary_plane_training_data_v3.py",
)
_NODE = "__anatomy_tracker_prepared_section_cache_v3_node__"
_ENTRY_FILES = {"manifest.json", "metadata.json", "arrays.npz"}
_STORAGE_FORMAT = (
    "canonical JSON metadata plus NumPy NPZ arrays; allow_pickle=False"
)


def _source_hashes():
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _i_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("prepared-section caches must be stored only on I:")
    return resolved


def _namespace(cache_directory, *, create=False):
    root = _i_path(cache_directory)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    namespace = _i_path(root / PREPARED_SECTION_NAMESPACE)
    if namespace.parent != root:
        raise ValueError("prepared-section namespace escapes its cache root")
    if create:
        namespace.mkdir(exist_ok=True)
    return namespace


def _section_directory(cache_directory, prepared_receipt_sha256, *, create_namespace=False):
    if not _valid_sha256(prepared_receipt_sha256):
        raise ValueError("prepared-section content address must be one lowercase SHA-256")
    namespace = _namespace(cache_directory, create=create_namespace)
    target = _i_path(namespace / prepared_receipt_sha256)
    if target.parent != namespace:
        raise ValueError("prepared-section content address escapes its namespace")
    return target


def _safe_entry_file(section_directory, relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("prepared-section cache path is empty")
    path = _i_path(section_directory / relative_path)
    if path.parent != section_directory:
        raise ValueError("prepared-section cache path escapes its content entry")
    return path


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez(
            handle,
            **{
                name: np.ascontiguousarray(value)
                for name, value in sorted(arrays.items())
            },
        )
        handle.flush()
        os.fsync(handle.fileno())


def _register_array(value, arrays):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("prepared-section cache never stores object arrays")
    key = f"array_{len(arrays):08d}"
    arrays[key] = array
    return key, acquisition._array_receipt(array)


def _encode_node(value, arrays):
    if isinstance(value, np.ndarray):
        key, receipt = _register_array(value, arrays)
        return {_NODE: "ndarray", "key": key, "array_receipt": receipt}
    if isinstance(value, np.generic):
        key, receipt = _register_array(np.asarray(value), arrays)
        return {_NODE: "numpy-scalar", "key": key, "array_receipt": receipt}
    if isinstance(value, MappingProxyType):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("prepared-section mapping keys must be strings")
        return {
            _NODE: "mappingproxy",
            "items": [
                [key, _encode_node(value[key], arrays)] for key in sorted(value)
            ],
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("prepared-section mapping keys must be strings")
        return {
            _NODE: "dict",
            "items": [
                [key, _encode_node(value[key], arrays)] for key in sorted(value)
            ],
        }
    if isinstance(value, tuple):
        return {_NODE: "tuple", "items": [_encode_node(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode_node(item, arrays) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"prepared-section metadata type {type(value).__name__!r} is unsupported")


def _decode_node(value, arrays, used):
    if isinstance(value, list):
        return [_decode_node(item, arrays, used) for item in value]
    if not isinstance(value, dict) or _NODE not in value:
        return value
    kind = value[_NODE]
    if kind in ("ndarray", "numpy-scalar"):
        if set(value) != {_NODE, "key", "array_receipt"}:
            raise ValueError("prepared-section array marker is malformed")
        key = value["key"]
        if key not in arrays or key in used:
            raise ValueError("prepared-section array marker is missing or duplicated")
        if value["array_receipt"] != acquisition._array_receipt(arrays[key]):
            raise ValueError("prepared-section array marker receipt changed")
        used.add(key)
        return arrays[key] if kind == "ndarray" else arrays[key][()]
    if kind in ("dict", "mappingproxy"):
        if set(value) != {_NODE, "items"} or not isinstance(value["items"], list):
            raise ValueError("prepared-section mapping marker is malformed")
        result = {}
        for item in value["items"]:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or item[0] in result
            ):
                raise ValueError("prepared-section mapping entry is malformed")
            result[item[0]] = _decode_node(item[1], arrays, used)
        return MappingProxyType(result) if kind == "mappingproxy" else result
    if kind == "tuple":
        if set(value) != {_NODE, "items"} or not isinstance(value["items"], list):
            raise ValueError("prepared-section tuple marker is malformed")
        return tuple(_decode_node(item, arrays, used) for item in value["items"])
    raise ValueError("prepared-section metadata contains an unknown node marker")


def _manifest_payload(manifest):
    return {key: value for key, value in manifest.items() if key != "receipt_sha256"}


def _make_manifest(prepared, metadata_path, arrays_path, arrays):
    payload = {
        "schema_version": PREPARED_SECTION_CACHE_V3_SCHEMA,
        "algorithm": PREPARED_SECTION_CACHE_V3_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "storage_format": _STORAGE_FORMAT,
        "prepared_schema_version": prepared["schema_version"],
        "prepared_implementation_source_sha256": prepared[
            "implementation_source_sha256"
        ],
        "prepared_training_section_id": prepared["prepared_training_section_id"],
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "parent_authentication_receipt_sha256": prepared[
            "parent_authentication_v3"
        ]["receipt_sha256"],
        "metadata_relative_path": metadata_path.name,
        "metadata_file_sha256": _file_sha256(metadata_path),
        "arrays_relative_path": arrays_path.name,
        "arrays_file_sha256": _file_sha256(arrays_path),
        "array_count": len(arrays),
        "array_receipts": {
            name: acquisition._array_receipt(array)
            for name, array in sorted(arrays.items())
        },
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _verify_manifest(manifest, prepared_receipt_sha256):
    expected_keys = {
        "schema_version",
        "algorithm",
        "implementation_source_sha256",
        "storage_format",
        "prepared_schema_version",
        "prepared_implementation_source_sha256",
        "prepared_training_section_id",
        "prepared_receipt_sha256",
        "parent_authentication_receipt_sha256",
        "metadata_relative_path",
        "metadata_file_sha256",
        "arrays_relative_path",
        "arrays_file_sha256",
        "array_count",
        "array_receipts",
        "receipt_sha256",
    }
    payload = _manifest_payload(manifest)
    if (
        set(manifest) != expected_keys
        or manifest.get("receipt_sha256") != _hash_json(payload)
        or manifest.get("schema_version") != PREPARED_SECTION_CACHE_V3_SCHEMA
        or manifest.get("algorithm") != PREPARED_SECTION_CACHE_V3_ALGORITHM
        or manifest.get("implementation_source_sha256") != _source_hashes()
        or manifest.get("storage_format") != _STORAGE_FORMAT
        or manifest.get("prepared_schema_version")
        != training_data_v3.PREPARED_TRAINING_SECTION_V3_SCHEMA
        or manifest.get("prepared_receipt_sha256") != prepared_receipt_sha256
        or not _valid_sha256(manifest.get("prepared_training_section_id"))
        or not _valid_sha256(manifest.get("parent_authentication_receipt_sha256"))
        or not _valid_sha256(manifest.get("metadata_file_sha256"))
        or not _valid_sha256(manifest.get("arrays_file_sha256"))
        or manifest.get("metadata_relative_path") != "metadata.json"
        or manifest.get("arrays_relative_path") != "arrays.npz"
        or not isinstance(manifest.get("array_count"), int)
        or isinstance(manifest.get("array_count"), bool)
        or manifest.get("array_count") < 0
        or not isinstance(manifest.get("array_receipts"), dict)
        or manifest.get("array_count") != len(manifest.get("array_receipts", {}))
    ):
        raise ValueError("prepared-section cache manifest failed source/schema/receipt authentication")


def save_prepared_training_section_v3(cache_directory, prepared, prepared_context):
    """Atomically persist one verified prepared parent under its receipt SHA-256."""
    cache_directory = _i_path(cache_directory)
    training_data_v3.verify_prepared_training_section_v3(prepared, prepared_context)
    receipt = prepared["receipt_sha256"]
    target = _section_directory(
        cache_directory, receipt, create_namespace=True
    )
    if target.exists():
        raise FileExistsError("prepared-section content address already exists; overwrite refused")
    arrays = {}
    metadata = _encode_node(prepared, arrays)
    namespace = target.parent
    staging = _i_path(namespace / f".{receipt}.{uuid.uuid4().hex}.tmp")
    if staging.parent != namespace:
        raise ValueError("prepared-section staging path escapes its namespace")
    staging.mkdir(exist_ok=False)
    try:
        metadata_path = staging / "metadata.json"
        arrays_path = staging / "arrays.npz"
        _write_json(metadata_path, metadata)
        _write_npz(arrays_path, arrays)
        manifest = _make_manifest(prepared, metadata_path, arrays_path, arrays)
        _write_json(staging / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(
                "prepared-section content address already exists; overwrite refused"
            )
        staging.rename(target)
    except Exception:
        if staging.exists() and staging.parent == namespace:
            shutil.rmtree(staging)
        raise
    return manifest


def _read_manifest(section_directory, prepared_receipt_sha256):
    if set(path.name for path in section_directory.iterdir()) != _ENTRY_FILES:
        raise ValueError("prepared-section cache entry has missing or extra files")
    try:
        manifest = json.loads(
            (section_directory / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("prepared-section cache manifest is truncated or unreadable") from error
    _verify_manifest(manifest, prepared_receipt_sha256)
    return manifest


def _load_arrays(arrays_path, manifest):
    try:
        with np.load(arrays_path, allow_pickle=False) as stored:
            if len(stored.files) != len(set(stored.files)):
                raise ValueError("prepared-section NPZ contains duplicate arrays")
            arrays = {
                name: np.array(stored[name], copy=True, order="C")
                for name in stored.files
            }
    except (OSError, ValueError, EOFError) as error:
        raise ValueError("prepared-section NPZ is truncated or unsafe") from error
    expected = manifest["array_receipts"]
    if (
        set(arrays) != set(expected)
        or len(arrays) != manifest["array_count"]
        or any(
            acquisition._array_receipt(arrays[name]) != expected[name]
            for name in arrays
        )
    ):
        raise ValueError("prepared-section cached array receipt changed")
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def _load_prepared_section_directory(
    section_directory,
    prepared_receipt_sha256,
    prepared_context,
    *,
    expected_manifest_receipt_sha256=None,
):
    manifest = _read_manifest(section_directory, prepared_receipt_sha256)
    if (
        expected_manifest_receipt_sha256 is not None
        and manifest["receipt_sha256"] != expected_manifest_receipt_sha256
    ):
        raise ValueError("prepared-section cache manifest receipt differs")
    metadata_path = _safe_entry_file(
        section_directory, manifest["metadata_relative_path"]
    )
    arrays_path = _safe_entry_file(
        section_directory, manifest["arrays_relative_path"]
    )
    if (
        _file_sha256(metadata_path) != manifest["metadata_file_sha256"]
        or _file_sha256(arrays_path) != manifest["arrays_file_sha256"]
    ):
        raise ValueError("prepared-section cache file hash differs")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("prepared-section metadata is truncated or unreadable") from error
    arrays = _load_arrays(arrays_path, manifest)
    used = set()
    prepared = _decode_node(metadata, arrays, used)
    if used != set(arrays):
        raise ValueError("prepared-section metadata does not consume every cached array")
    if (
        not isinstance(prepared, dict)
        or prepared.get("schema_version") != manifest["prepared_schema_version"]
        or prepared.get("implementation_source_sha256")
        != manifest["prepared_implementation_source_sha256"]
        or prepared.get("prepared_training_section_id")
        != manifest["prepared_training_section_id"]
        or prepared.get("receipt_sha256") != prepared_receipt_sha256
        or prepared.get("parent_authentication_v3", {}).get("receipt_sha256")
        != manifest["parent_authentication_receipt_sha256"]
    ):
        raise ValueError("prepared-section content differs from its cache manifest")
    training_data_v3.verify_prepared_training_section_v3(prepared, prepared_context)
    return prepared, manifest


def load_prepared_training_section_v3(
    cache_directory,
    prepared_receipt_sha256,
    prepared_context,
    *,
    expected_manifest_receipt_sha256=None,
):
    """Load and authenticate one content-addressed prepared parent."""
    section_directory = _section_directory(
        cache_directory, prepared_receipt_sha256
    )
    if not section_directory.is_dir():
        raise FileNotFoundError("prepared-section content address does not exist")
    prepared, _ = _load_prepared_section_directory(
        section_directory,
        prepared_receipt_sha256,
        prepared_context,
        expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
    )
    return prepared


def audit_prepared_training_section_cache_v3(cache_directory, prepared_context):
    """Authenticate cached parents one at a time without retaining a parent list."""
    namespace = _namespace(cache_directory)
    if not namespace.is_dir():
        raise FileNotFoundError("prepared-section cache namespace does not exist")
    receipts = []
    for section_directory in sorted(namespace.iterdir(), key=lambda path: path.name):
        receipt = section_directory.name
        if not section_directory.is_dir() or not _valid_sha256(receipt):
            raise ValueError("prepared-section namespace contains an invalid entry")
        prepared, _ = _load_prepared_section_directory(
            section_directory, receipt, prepared_context
        )
        receipts.append(prepared["receipt_sha256"])
        del prepared
    return {
        "schema_version": PREPARED_SECTION_CACHE_V3_SCHEMA,
        "implementation_source_sha256": _source_hashes(),
        "prepared_section_count": len(receipts),
        "ordered_prepared_receipts_sha256": _hash_json(receipts),
        "all_prepared_sections_authenticated": True,
    }
