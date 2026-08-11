import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import source.atlas_pose_runtime as runtime
from source.atlas_pose_runtime import automatic_brain_mask


def test_automatic_brain_mask_rejects_dim_textured_slide_background():
    rng = np.random.default_rng(17)
    height, width = 360, 520
    image = np.clip(rng.normal(4.0, 2.0, (height, width)), 0, 255).astype(np.uint8)
    expected = np.zeros((height, width), np.uint8)
    cv2.ellipse(expected, (width // 2, height // 2), (175, 120), 0, 0, 360, 1, -1)
    yy, xx = np.mgrid[:height, :width]
    tissue = 75.0 + 18.0 * np.sin(xx / 18.0) + 12.0 * np.cos(yy / 15.0)
    image[expected > 0] = np.clip(tissue[expected > 0], 0, 255).astype(np.uint8)

    detected = automatic_brain_mask(image)
    intersection = np.logical_and(detected, expected).sum()
    union = np.logical_or(detected, expected).sum()

    assert intersection / union > 0.96
    assert not detected[[0, -1]].any()
    assert not detected[:, [0, -1]].any()


def test_automatic_brain_mask_retains_deep_tissue_in_low_contrast_nissl():
    rng = np.random.default_rng(9)
    height, width = 320, 440
    yy, xx = np.mgrid[:height, :width]
    expected = ((xx - 220.0) / 165.0) ** 2 + ((yy - 165.0) / 115.0) ** 2 <= 1.0
    inner = ((xx - 220.0) / 145.0) ** 2 + ((yy - 165.0) / 95.0) ** 2 <= 1.0
    image = np.clip(rng.normal(22.0, 2.0, (height, width)), 0, 255).astype(np.uint8)
    image[expected] = np.clip(
        42.0 + 5.0 * np.sin(xx[expected] / 13.0) + rng.normal(0.0, 2.0, expected.sum()),
        0,
        255,
    )
    cortex = expected & ~inner
    image[cortex] = np.clip(
        82.0 + 10.0 * np.sin(xx[cortex] / 17.0) + rng.normal(0.0, 4.0, cortex.sum()),
        0,
        255,
    )

    detected = automatic_brain_mask(image)

    assert np.logical_and(detected, expected).sum() / expected.sum() > 0.90
    assert detected[165, 220]


def test_mask_version_is_separate_from_the_onnx_tensor_preprocessing_contract(monkeypatch):
    contract = runtime.atlas_pose_preprocessing_contract_sha256()
    assert runtime.AUTOMATIC_BRAIN_MASK_VERSION == "border-distance-conditional-hull-v6"
    monkeypatch.setattr(runtime, "automatic_brain_mask", lambda image: np.ones(image.shape[:2], bool))
    assert runtime.atlas_pose_preprocessing_contract_sha256() == contract


def test_automatic_brain_mask_ignores_a_dark_slide_edge_during_retry():
    rng = np.random.default_rng(81)
    height, width = 295, 556
    yy, xx = np.mgrid[:height, :width]
    expected = ((xx - 365.0) / 145.0) ** 2 + ((yy - 125.0) / 100.0) ** 2 <= 1.0
    image = np.full((height, width), 248.0, dtype=np.float32)
    image[expected] = np.clip(
        205.0 + 18.0 * np.sin(xx[expected] / 4.0) + rng.normal(0.0, 8.0, expected.sum()),
        0.0,
        255.0,
    )
    image[-6:] = 8.0
    cv2.circle(image, (220, 250), 25, 225.0, -1)

    detected = automatic_brain_mask(image.astype(np.uint8))
    intersection = np.logical_and(detected, expected).sum()
    union = np.logical_or(detected, expected).sum()

    assert intersection / union > 0.90
    assert not detected[-1].any()


def test_automatic_brain_mask_recovers_extremely_pale_brightfield_tissue():
    rng = np.random.default_rng(72)
    height, width = 240, 298
    yy, xx = np.mgrid[:height, :width]
    expected = ((xx - 150.0) / 135.0) ** 2 + ((yy - 115.0) / 105.0) ** 2 <= 1.0
    image = rng.normal(253.0, 0.5, (height, width))
    image[expected] -= 2.0
    cv2.circle(image, (112, 72), 6, 60.0, 1)
    cv2.circle(image, (154, 179), 6, 60.0, 1)

    detected = automatic_brain_mask(np.rint(image).astype(np.uint8))
    intersection = np.logical_and(detected, expected).sum()
    union = np.logical_or(detected, expected).sum()

    assert intersection / union > 0.90


def test_preprocessing_contract_is_source_bound_and_not_cpython_bytecode_bound():
    text = Path(runtime.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    tree = ast.parse(text)
    lines = text.splitlines()
    functions = {
        node.name: "\n".join(lines[node.lineno - 1 : node.end_lineno]).strip()
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in runtime.ATLAS_POSE_PREPROCESSING_CONTRACT_FUNCTIONS
    }
    payload = {
        "version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
        "image_size": runtime.POSE_IMAGE_SIZE,
        "functions": {
            name: functions[name]
            for name in runtime.ATLAS_POSE_PREPROCESSING_CONTRACT_FUNCTIONS
        },
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert runtime.atlas_pose_preprocessing_contract_sha256() == expected
    assert "co_code" not in ast.get_source_segment(
        text,
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "atlas_pose_preprocessing_contract_sha256"),
    )


def test_automatic_brain_mask_rejects_an_uninformative_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[::80, ::80] = 3
    with pytest.raises(ValueError, match="No brain foreground was detected"):
        automatic_brain_mask(image)


def test_automatic_brain_mask_is_equivalent_for_uint8_float_and_uint16():
    height, width = 240, 320
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width), 7, dtype=np.uint8)
    inside = ((xx - 160.0) / 115.0) ** 2 + ((yy - 125.0) / 85.0) ** 2 < 1.0
    image[inside] = np.clip(110.0 + 35.0 * np.sin(xx[inside] / 12.0), 0, 255)
    expected = automatic_brain_mask(image)
    assert np.array_equal(automatic_brain_mask(image.astype(np.float32) / 255.0), expected)
    assert np.array_equal(automatic_brain_mask(image.astype(np.uint16) * 257), expected)
