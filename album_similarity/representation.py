import hashlib
import math

import numpy as np

from album_similarity.utils import run_module

TRACK_SETTINGS = ("one", "three", "half", "full")

# Tolerance for treating two medoid candidates' total distances as tied. Totals are
# computed in float64 from float32 inputs, so differences below this are rounding noise.
_MEDOID_TIE_TOLERANCE = 1e-6
_UNIT_NORM_TOLERANCE = 1e-3


def _sort_id(value):
    """Order IDs numerically when numeric, otherwise lexically; same rule as data._sort_id."""
    return (0, int(value)) if value.isdigit() else (1, value)


def _check_embeddings(track_embeddings):
    """Validate the shared precondition: nonempty, 2-D, float32, finite, unit-norm rows."""
    if not isinstance(track_embeddings, np.ndarray):
        raise TypeError("track_embeddings must be a NumPy array")
    if track_embeddings.ndim != 2 or track_embeddings.shape[0] == 0 or track_embeddings.shape[1] == 0:
        raise ValueError(f"track_embeddings must be a nonempty [N,D] matrix, got {track_embeddings.shape}")
    if track_embeddings.dtype != np.float32:
        raise ValueError(f"track_embeddings must be float32, got {track_embeddings.dtype}")
    if not np.isfinite(track_embeddings).all():
        raise ValueError("track_embeddings contains non-finite values")
    norms = np.linalg.norm(track_embeddings.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= _UNIT_NORM_TOLERANCE):
        raise ValueError(
            "track_embeddings rows must be L2-normalized "
            f"(norms range {norms.min():.4f}..{norms.max():.4f}); run scripts/adapt_embeddings.py"
        )


def _check_track_ids(track_ids, expected_length=None):
    """Validate a list of unique, nonempty string IDs, optionally of a given length."""
    if isinstance(track_ids, str):
        raise TypeError("track_ids must be a sequence of strings, not a single string")
    track_ids = list(track_ids)
    if not all(isinstance(tid, str) and tid for tid in track_ids):
        raise ValueError("track_ids must be nonempty strings")
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("track_ids must be unique")
    if expected_length is not None and len(track_ids) != expected_length:
        raise ValueError(f"expected {expected_length} track_ids, got {len(track_ids)}")
    return track_ids


def load_representation_inputs(albums_path, embeddings_dir, genre_index_path):
    """Explicitly load albums, successful track embeddings and index, and genre availability;
    return inputs joinable by ID.

    Args:
        albums_path: Path to the prepared albums CSV file.
        embeddings_dir: Directory containing track embeddings, their index, and manifest.
        genre_index_path: Path to the album genre index CSV file.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("representation.load_representation_inputs: not implemented")


def filter_eligible_albums(albums, track_ids, genre_index, min_tracks):
    """Filter albums using the intersection with successful track IDs; return (eligible album
    records, exclusion records).

    Args:
        albums: Album records containing track IDs, artist IDs, and genre metadata.
        track_ids: String track IDs aligned with the relevant records or embedding rows.
        genre_index: Album genre index records with row_index, album_id, and has_genres.
        min_tracks: Agreed minimum number of usable tracks required per album.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("representation.filter_eligible_albums: not implemented")


def aggregate_mean(track_embeddings):
    """Take a nonempty float32 [N,D] matrix of normalized tracks; return an L2-normalized [D]
    mean embedding.

    Args:
        track_embeddings: Nonempty float32 [N,D] matrix of finite, L2-normalized track
            embeddings.

    Returns:
        float32 [D] unit vector.

    Raises:
        ValueError: The input violates the precondition, or the tracks cancel out to a
            zero mean (which has no direction to normalize).
    """
    _check_embeddings(track_embeddings)
    mean = track_embeddings.astype(np.float64).mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 1e-12:
        raise ValueError("mean embedding has zero norm; tracks cancel out")
    return (mean / norm).astype(np.float32)


def aggregate_medoid(track_embeddings, track_ids):
    """Select the track minimizing total cosine distance, breaking ties by track_id; return its
    original [D] embedding.

    Args:
        track_embeddings: Nonempty float32 [N,D] matrix of finite, L2-normalized track
            embeddings.
        track_ids: String track IDs aligned with the relevant records or embedding rows.

    Returns:
        float32 [D] copy of the medoid track's own embedding (already unit-norm).

    Notes:
        With unit-norm rows, cosine distance is 1 - dot product, so the medoid is the row
        whose dot products with all rows sum highest. Totals within _MEDOID_TIE_TOLERANCE
        count as tied; ties go to the smallest track_id under _sort_id. Two-track albums
        always tie (the distance is symmetric), so their medoid is the smaller ID.
    """
    _check_embeddings(track_embeddings)
    track_ids = _check_track_ids(track_ids, expected_length=len(track_embeddings))
    vectors = track_embeddings.astype(np.float64)
    total_distance = (1.0 - vectors @ vectors.T).sum(axis=1)
    best = total_distance.min()
    tied = np.flatnonzero(total_distance <= best + _MEDOID_TIE_TOLERANCE)
    chosen = min(tied, key=lambda row: _sort_id(track_ids[row]))
    return track_embeddings[chosen].copy()


def aggregate(track_embeddings, method="mean", track_ids=None):
    """Return a [D] embedding using the selected aggregation method. Medoid requires aligned
    track_ids to avoid relying on incidental row order.

    Args:
        track_embeddings: Nonempty float32 [N,D] matrix of finite, L2-normalized track
            embeddings.
        method: Aggregation method: mean or medoid.
        track_ids: String track IDs aligned with the relevant records or embedding rows.

    Raises:
        ValueError: Unknown method, or medoid requested without track_ids.
    """
    if method == "mean":
        if track_ids is not None:
            _check_track_ids(track_ids, expected_length=len(track_embeddings))
        return aggregate_mean(track_embeddings)
    if method == "medoid":
        if track_ids is None:
            raise ValueError("medoid aggregation requires aligned track_ids")
        return aggregate_medoid(track_embeddings, track_ids)
    raise ValueError(f"unknown aggregation method: {method!r}")


def resolve_track_count(track_count, setting):
    """Map one/three/half/full to an integer count; round half upward and raise ValueError when
    three tracks are unavailable.

    Args:
        track_count: Number of usable tracks available in the album.
        setting: Requested track setting: one, three, half, or full.

    Returns:
        one -> 1, three -> 3, half -> ceil(track_count / 2), full -> track_count.

    Raises:
        ValueError: Unknown setting, a nonpositive count, or fewer than three tracks for
            the three setting.
    """
    if type(track_count) is not int or track_count < 1:
        raise ValueError(f"track_count must be a positive integer, got {track_count!r}")
    if setting == "one":
        return 1
    if setting == "three":
        if track_count < 3:
            raise ValueError(f"setting 'three' needs at least 3 tracks, album has {track_count}")
        return 3
    if setting == "half":
        return math.ceil(track_count / 2)
    if setting == "full":
        return track_count
    raise ValueError(f"unknown track setting: {setting!r}")


def derive_sampling_seed(global_seed, album_id, setting, repeat_id):
    """Derive a nonnegative integer seed using a stable digest, independent of built-in hash
    and process random state.

    Args:
        global_seed: Nonnegative base seed shared by the experiment.
        album_id: String album ID used to derive a reproducible sampling seed.
        setting: Requested track setting: one, three, half, or full.
        repeat_id: Index of the repeated sampling observation.

    Returns:
        Integer in [0, 2**63): the first 8 bytes of a SHA-256 digest, top bit cleared.

    Notes:
        Python's built-in hash() is salted per process for strings, so it would give
        different seeds on every run. SHA-256 gives the same seed on every machine.
    """
    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a nonnegative integer")
    if not isinstance(album_id, str) or not album_id:
        raise ValueError("album_id must be a nonempty string")
    if setting not in TRACK_SETTINGS:
        raise ValueError(f"unknown track setting: {setting!r}")
    if type(repeat_id) is not int or repeat_id < 0:
        raise ValueError("repeat_id must be a nonnegative integer")
    key = f"{global_seed}|{album_id}|{setting}|{repeat_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sample_tracks(track_ids, setting, seed):
    """Sort by track_id before sampling without replacement; return selected_track_ids without
    modifying the input.

    Args:
        track_ids: String track IDs aligned with the relevant records or embedding rows.
        setting: Requested track setting: one, three, half, or full.
        seed: Nonnegative integer seed for deterministic sampling.

    Returns:
        New list of selected IDs in canonical (_sort_id) order. The full setting returns
        every track without drawing random numbers.

    Raises:
        ValueError: Invalid IDs, setting, or seed, or too few tracks for the setting.
    """
    canonical = sorted(_check_track_ids(track_ids), key=_sort_id)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    count = resolve_track_count(len(canonical), setting)
    if count == len(canonical):
        return canonical
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(canonical), size=count, replace=False))
    return [canonical[i] for i in chosen]


def build_sampling_plan(albums, settings, repeats, seed):
    """Return (sampling_plan records, unavailable_settings records) for all albums; include
    only one full row per album.

    Args:
        albums: Album records containing track IDs, artist IDs, and genre metadata.
        settings: Track settings to include in the sampling plan.
        repeats: Number of random repetitions for partial-track settings.
        seed: Nonnegative integer seed for deterministic sampling.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("representation.build_sampling_plan: not implemented")


def compute_diversity(track_embeddings):
    """Return the mean cosine distance from tracks to the full-album mean center, independent
    of the retrieval aggregation method.

    Args:
        track_embeddings: Nonempty float32 [N,D] matrix of finite, L2-normalized track
            embeddings. Pass ALL of the album's usable tracks, not a sample.

    Returns:
        Python float in [0, 2]; 0.0 for a single-track album. Raw cosine distance, not
        rescaled, so it matches the album_stats column diversity_mean_cosine_distance.
    """
    center = aggregate_mean(track_embeddings).astype(np.float64)
    distances = 1.0 - track_embeddings.astype(np.float64) @ center
    return float(np.clip(distances, 0.0, 2.0).mean())


def build_album_representations(inputs, config):
    """Build mean/medoid matrices, diversity, and sampling plans for a common eligible album
    set; return artifacts and exclusion reasons.

    Args:
        inputs: Representation input dictionary loaded and joinable by explicit IDs.
        config: Configuration dictionary for this workflow or operation.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("representation.build_album_representations: not implemented")


def save_representations(representations, config, output_dir):
    """Write album matrices, shared index, eligible albums, statistics, sampling plan, and
    manifest; return the manifest dictionary.

    Args:
        representations: Representation outputs: shared album index, matrices, statistics,
            and sampling plan.
        config: Configuration dictionary for this workflow or operation.
        output_dir: Destination directory for the artifacts produced by this operation.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("representation.save_representations: not implemented")


def run_pipeline(config, project_root):
    """Run the album workflow: read upstream artifacts, write representation artifacts, and
    return the manifest dictionary.

    Args:
        config: Configuration dictionary for this workflow or operation.
        project_root: Resolved project root Path used to interpret configured artifact
            paths.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
        Implement feature functions first, then coordinate them here.
        main() handles CLI arguments; do not substitute invented research results.
    """
    raise NotImplementedError("representation.run_pipeline: not implemented")


def main(argv=None):
    """The sole CLI entry point; supports --config, --help, and --check-config.

    Args:
        argv: Command-line argument list, or None to use the process arguments.

    Notes:
        The CLI is wired; normal execution still calls an unimplemented workflow.
        --check-config validates configuration without running that workflow.
    """
    return run_module("representation", run_pipeline, argv)


if __name__ == "__main__":
    main()