import numpy as np

from source.dense_registration_preprocessing import (
    FEATHER_RING_VALUES,
    numpy_cosine_mask_feather,
)


def test_cosine_mask_feather_has_exact_outward_rings_and_zero_beyond_radius_three():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    alpha = numpy_cosine_mask_feather(mask)

    assert alpha[5, 5] == 1.0
    assert alpha[5, 6] == FEATHER_RING_VALUES[0]
    assert alpha[5, 7] == FEATHER_RING_VALUES[1]
    assert alpha[5, 8] == FEATHER_RING_VALUES[2]
    assert alpha[5, 9] == 0.0
    assert alpha.dtype == np.float32


def test_cosine_mask_feather_does_not_wrap_at_native_canvas_edges():
    mask = np.zeros((8, 9), dtype=bool)
    mask[0, 0] = True
    alpha = numpy_cosine_mask_feather(mask)

    assert alpha[0, 0] == 1.0
    assert alpha[0, 1] == FEATHER_RING_VALUES[0]
    assert alpha[-1, -1] == 0.0
