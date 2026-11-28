
import numpy as np
import pytest

from album_similarity.representation import (
    aggregate, aggregate_mean, aggregate_medoid, compute_diversity,
    derive_sampling_seed, resolve_track_count, sample_tracks,
)


def unit_rows(*rows):
    matrix = np.asarray(rows, dtype=np.float64)
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def random_unit(n, d=16, seed=0):
    matrix = np.random.default_rng(seed).normal(size=(n, d))
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


# input validation shared by all aggregations

@pytest.mark.parametrize("bad", [
    np.zeros((0, 4), dtype=np.float32),                         # empty
    np.ones((2, 4), dtype=np.float64) / 2,                      # wrong dtype
    np.array([[1, 0], [np.nan, 0]], dtype=np.float32),          # non-finite
    np.array([[3, 4], [0, 1]], dtype=np.float32),               # not unit norm
    np.zeros((2, 4), dtype=np.float32),                         # zero rows (B's failures)
])
def test_aggregations_reject_invalid_embeddings(bad):
    with pytest.raises(ValueError):
        aggregate_mean(bad)


# mean

def test_mean_is_normalized_average():
    emb = unit_rows([1, 0], [0, 1])
    result = aggregate_mean(emb)
    assert result.dtype == np.float32 and result.shape == (2,)
    np.testing.assert_allclose(result, [2 ** -0.5, 2 ** -0.5], atol=1e-6)


def test_mean_of_single_track_is_that_track():
    emb = random_unit(1)
    np.testing.assert_allclose(aggregate_mean(emb), emb[0], atol=1e-6)


def test_mean_of_opposite_tracks_raises():
    with pytest.raises(ValueError, match="zero norm"):
        aggregate_mean(unit_rows([1, 0], [-1, 0]))


def test_mean_does_not_modify_input():
    emb = random_unit(5)
    before = emb.copy()
    aggregate_mean(emb)
    np.testing.assert_array_equal(emb, before)


# medoid

def test_medoid_picks_central_track_and_returns_its_own_vector():
    # The middle vector is closest to both neighbours.
    emb = unit_rows([1, 0], [1, 1], [0, 1])
    result = aggregate_medoid(emb, ["10", "20", "30"])
    np.testing.assert_array_equal(result, emb[1])


def test_medoid_ignores_row_order():
    emb = unit_rows([1, 0], [1, 1], [0, 1])
    ids = ["10", "20", "30"]
    order = [2, 0, 1]
    shuffled = aggregate_medoid(emb[order], [ids[i] for i in order])
    np.testing.assert_array_equal(shuffled, aggregate_medoid(emb, ids))


def test_medoid_two_tracks_tie_goes_to_numerically_smallest_id():
    emb = unit_rows([1, 0], [0, 1])
    # "9" < "10" numerically but not lexically; the numeric rule must win.
    result = aggregate_medoid(emb, ["10", "9"])
    np.testing.assert_array_equal(result, emb[1])


def test_medoid_result_is_a_copy():
    emb = random_unit(4)
    result = aggregate_medoid(emb, ["1", "2", "3", "4"])
    result[:] = 0
    assert np.linalg.norm(emb, axis=1).min() > 0.99


@pytest.mark.parametrize("ids", [["1", "2"], ["1", "1", "2"], ["1", "", "2"], "123"])
def test_medoid_rejects_misaligned_or_bad_ids(ids):
    with pytest.raises((ValueError, TypeError)):
        aggregate_medoid(random_unit(3), ids)


# aggregate dispatch

def test_aggregate_dispatches_and_validates():
    emb = random_unit(5)
    ids = [str(i) for i in range(5)]
    np.testing.assert_array_equal(aggregate(emb, "mean"), aggregate_mean(emb))
    np.testing.assert_array_equal(aggregate(emb, "medoid", ids), aggregate_medoid(emb, ids))
    with pytest.raises(ValueError, match="requires aligned track_ids"):
        aggregate(emb, "medoid")
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(emb, "median")


# track counts

@pytest.mark.parametrize("count,setting,expected", [
    (1, "one", 1), (10, "one", 1),
    (3, "three", 3), (10, "three", 3),
    (3, "half", 2), (4, "half", 2), (5, "half", 3), (1, "half", 1),
    (7, "full", 7),
])
def test_resolve_track_count(count, setting, expected):
    assert resolve_track_count(count, setting) == expected


@pytest.mark.parametrize("count,setting", [(2, "three"), (0, "one"), (5, "quarter"), (3.0, "one")])
def test_resolve_track_count_rejects(count, setting):
    with pytest.raises(ValueError):
        resolve_track_count(count, setting)


# seeds

def test_seed_is_stable_and_known():
    seed = derive_sampling_seed(42, "1889", "half", 3)
    assert seed == derive_sampling_seed(42, "1889", "half", 3)
    assert 0 <= seed < 2 ** 63
    # Pinned value: if this changes, every saved sampling plan is invalidated.
    assert seed == 3598126916191441497


def test_seed_varies_with_every_component():
    base = derive_sampling_seed(42, "1", "one", 0)
    assert base != derive_sampling_seed(43, "1", "one", 0)
    assert base != derive_sampling_seed(42, "2", "one", 0)
    assert base != derive_sampling_seed(42, "1", "three", 0)
    assert base != derive_sampling_seed(42, "1", "one", 1)


def test_seed_has_no_delimiter_collisions():
    # Without separators, ("1", repeat 23) and ("12", repeat 3) could hash the same key.
    assert derive_sampling_seed(1, "1", "one", 23) != derive_sampling_seed(1, "12", "one", 3)


@pytest.mark.parametrize("args", [(-1, "1", "one", 0), (1, "", "one", 0),
                                  (1, "1", "most", 0), (1, "1", "one", -1), (1, 5, "one", 0)])
def test_seed_rejects_bad_inputs(args):
    with pytest.raises(ValueError):
        derive_sampling_seed(*args)


# sampling

def test_sampling_is_deterministic_and_order_independent():
    ids = [str(i) for i in range(1, 21)]
    a = sample_tracks(ids, "half", seed=7)
    b = sample_tracks(list(reversed(ids)), "half", seed=7)
    assert a == b
    assert len(a) == 10 and len(set(a)) == 10 and set(a) <= set(ids)


def test_sampling_returns_canonical_numeric_order():
    result = sample_tracks(["100", "9", "20"], "full", seed=0)
    assert result == ["9", "20", "100"]


def test_sampling_does_not_modify_input():
    ids = ["3", "1", "2"]
    sample_tracks(ids, "one", seed=1)
    assert ids == ["3", "1", "2"]


def test_sampling_different_seeds_explore_different_subsets():
    ids = [str(i) for i in range(30)]
    draws = {tuple(sample_tracks(ids, "three", seed=s)) for s in range(20)}
    assert len(draws) > 10


def test_sampling_errors():
    with pytest.raises(ValueError):
        sample_tracks(["1", "2"], "three", seed=0)
    with pytest.raises(ValueError):
        sample_tracks(["1", "1", "2"], "one", seed=0)
    with pytest.raises(ValueError):
        sample_tracks(["1", "2"], "one", seed=-1)


# diversity

def test_diversity_of_identical_tracks_is_zero():
    emb = unit_rows([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert compute_diversity(emb) == pytest.approx(0.0, abs=1e-6)


def test_diversity_single_track_is_zero():
    assert compute_diversity(random_unit(1)) == pytest.approx(0.0, abs=1e-6)


def test_diversity_matches_hand_computation():
    # Orthogonal pair: center is the diagonal, each track at cos 1/sqrt(2).
    emb = unit_rows([1, 0], [0, 1])
    assert compute_diversity(emb) == pytest.approx(1 - 2 ** -0.5, abs=1e-6)


def test_diversity_increases_with_spread():
    tight = unit_rows([1, 0.05, 0], [1, -0.05, 0], [1, 0, 0.05])
    loose = unit_rows([1, 0.8, 0], [1, -0.8, 0], [1, 0, 0.8])
    assert compute_diversity(loose) > compute_diversity(tight)


def test_diversity_returns_python_float():
    assert type(compute_diversity(random_unit(4))) is float