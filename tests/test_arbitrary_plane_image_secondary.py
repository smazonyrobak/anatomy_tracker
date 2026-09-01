import numpy as np
from scipy.ndimage import binary_erosion

from training.arbitrary_plane_image_information import rank_candidate_scores
from training.arbitrary_plane_image_secondary import (
    central_gradients,
    hog_blocks,
    hog_boundary_ring_weights,
    hog_complete_block_mask,
    ngf_evaluation_domain,
    score_hog_candidates,
    score_ngf_candidates,
)


def _texture():
    y, x = np.indices((65, 67))
    return ((7 * y + 11 * x) % 23).astype(np.float64) / 22.0


def test_central_gradients_use_physical_xy_and_scalar_padding():
    image = np.arange(24, dtype=np.float64).reshape(4, 6)
    gx, gy = central_gradients(image, 2.0, padding_value=3.0)
    assert gx[2, 3] == (image[2, 4] - image[2, 2]) / 4.0
    assert gy[2, 3] == (image[3, 3] - image[1, 3]) / 4.0
    assert gx[0, 0] == (image[0, 1] - 3.0) / 4.0
    assert gy[0, 0] == (image[1, 0] - 3.0) / 4.0


def test_hog_shapes_complete_blocks_and_chunk_scores_are_exact():
    image = _texture()
    blocks, q = hog_blocks(image, 50.0)
    assert q == 8
    assert blocks.shape == (8, 8, 36)
    assert hog_complete_block_mask(np.ones(image.shape, bool), q).sum() == 36
    candidates = np.stack((image, np.roll(image, 1, 1), 1.0 - image))
    first = score_hog_candidates(image, candidates, np.ones(image.shape, bool), 50.0, chunk_size=1)
    second = score_hog_candidates(image, candidates, np.ones(image.shape, bool), 50.0, chunk_size=3)
    assert first["eligible_block_count"] == 36
    assert first["cell_pixels"] == 8
    assert np.array_equal(first["block_weights"], second["block_weights"])
    assert np.array_equal(first["scores"], second["scores"])
    assert first["scores"][0] == 1.0

    ring = np.zeros(image.shape, bool)
    ring[10, 10] = True
    weights = hog_boundary_ring_weights(ring, q)
    ring_result = score_hog_candidates(
        image, candidates, ring, 50.0, boundary_ring=True
    )
    assert weights.sum() == 1
    assert ring_result["eligible_block_count"] == 1
    assert np.array_equal(ring_result["block_weights"], weights)


def test_ngf_uses_exact_post_smoothing_domain_and_is_chunk_invariant():
    image = _texture()
    domain = np.zeros(image.shape, bool)
    domain[8:57, 8:59] = True
    expected = binary_erosion(
        domain,
        structure=np.ones((15, 15), bool),
        iterations=1,
        border_value=0,
        origin=0,
        brute_force=False,
    )
    assert np.array_equal(ngf_evaluation_domain(domain, 50.0), expected)
    candidates = np.stack((image, np.roll(image, 1, 0), 1.0 - image))
    first = score_ngf_candidates(image, candidates, domain, 50.0, chunk_size=1)
    second = score_ngf_candidates(image, candidates, domain, 50.0, chunk_size=3)
    assert first["effective_domain_count"] == int(expected.sum())
    assert first["gaussian_radius_px"] == 6
    assert first["target_eta"] == second["target_eta"]
    assert np.array_equal(first["candidate_eta"], second["candidate_eta"])
    assert np.array_equal(first["scores"], second["scores"])
    assert first["scores"][0] == 1.0


def test_secondary_descriptors_return_null_for_insufficient_domains():
    image = np.ones((15, 17), dtype=np.float64)
    candidates = np.stack((image, image))
    domain = np.zeros(image.shape, bool)
    assert score_hog_candidates(image, candidates, domain, 100.0) is None
    assert score_ngf_candidates(image, candidates, domain, 100.0) is None


def test_hog_and_ngf_affine_polarity_rank_controls():
    image = _texture()
    candidates = np.stack(
        (image, np.roll(image, 1, 0), np.roll(image, 2, 1), np.roll(image, 3, 0))
    )
    domain = np.zeros(image.shape, bool)
    domain[8:57, 8:59] = True
    ids = ["c0", "c1", "c2", "c3"]
    for scorer in (score_hog_candidates, score_ngf_candidates):
        original = scorer(image, candidates, domain, 50.0)["scores"]
        expected = rank_candidate_scores(original, ids, "c0")
        for scale, offset, padding in ((0.7, 0.1, 0.1), (1.2, 0.0, 0.0), (-1.0, 1.0, 1.0)):
            transformed = scorer(
                scale * image + offset,
                scale * candidates + offset,
                domain,
                50.0,
                padding_value=padding,
            )["scores"]
            observed = rank_candidate_scores(transformed, ids, "c0")
            assert observed["top1"] == expected["top1"]
            assert observed["true_rank"] == expected["true_rank"]
            assert observed["tied_maximum_candidate_ids"] == expected["tied_maximum_candidate_ids"]
