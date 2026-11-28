from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from album_similarity.utils import write_table, read_table  # noqa: E402
 
# Treat a row as a failed extraction when its norm is at or below this.
# B writes exact zero vectors, so this is a guard against float dust, not a threshold.
ZERO_NORM_TOLERANCE = 1e-8
 
 
def load_source(tracks_csv: Path, matrix_path: Path, manifest_path: Path):
    """Read B's three artifacts and check they are mutually consistent.
 
    Returns:
        (track_rows, vectors, manifest)
 
    Raises:
        ValueError: The matrix and track list disagree, or the matrix is malformed.
    """
    track_rows = read_table(tracks_csv, "tracks")
    vectors = np.load(matrix_path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
 
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
    if vectors.dtype != np.float32:
        raise ValueError(f"expected float32, got {vectors.dtype}")
    if len(track_rows) != len(vectors):
        raise ValueError(
            f"row count mismatch: {tracks_csv.name} has {len(track_rows)} rows, "
            f"matrix has {len(vectors)}. The matrix is aligned to the exact tracks.csv "
            f"B ran against; do not substitute a regenerated one."
        )
 
    track_ids = [row["track_id"] for row in track_rows]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("track_id values are not unique")
    if any(not tid for tid in track_ids):
        raise ValueError("empty track_id encountered")
 
    return track_rows, vectors, manifest
 
 
def classify_rows(track_rows, vectors, manifest):
    """Split rows into successful and failed, cross-checking both failure signals.
 
    B reports failures two ways: zero vectors in the matrix, and a failed_track_details
    list in the manifest. They should name the same tracks. Disagreement means one of
    the two artifacts is stale, which is worth stopping for rather than guessing.
 
    Returns:
        (ok_indices, failed_indices, failure_notes)
    """
    norms = np.linalg.norm(vectors, axis=1)
    zero_rows = {int(i) for i in np.where(norms <= ZERO_NORM_TOLERANCE)[0]}
    nonfinite_rows = {int(i) for i in np.where(~np.isfinite(vectors).all(axis=1))[0]}
    failed = zero_rows | nonfinite_rows
 
    index_by_track_id = {row["track_id"]: i for i, row in enumerate(track_rows)}
    manifest_failures = {}
    for entry in manifest.get("failed_track_details", []):
        track_id = str(entry.get("track_id", ""))
        if track_id not in index_by_track_id:
            raise ValueError(f"manifest names unknown track_id {track_id!r}")
        manifest_failures[index_by_track_id[track_id]] = entry
 
    declared = set(manifest_failures)
    if declared != failed:
        only_matrix = sorted(failed - declared)
        only_manifest = sorted(declared - failed)
        raise ValueError(
            "manifest failures and matrix zero/non-finite rows disagree; "
            f"rows bad only in matrix: {only_matrix}, "
            f"rows failed only in manifest: {only_manifest}"
        )
 
    notes = {}
    for i in sorted(failed):
        entry = manifest_failures.get(i, {})
        message = str(entry.get("error", "") or "")
        notes[i] = (
            "nonfinite" if i in nonfinite_rows else "zero_vector",
            message,
        )
 
    ok_indices = [i for i in range(len(track_rows)) if i not in failed]
    return ok_indices, sorted(failed), notes
 
 
def l2_normalize(vectors):
    """Return an L2-normalized float32 copy. Input must contain no zero rows."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(norms > ZERO_NORM_TOLERANCE):
        raise ValueError("zero-norm row reached normalization; filter failures first")
    normalized = (vectors / norms).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("normalization produced non-finite values")
    return normalized
 
 
def write_outputs(out_dir: Path, track_rows, vectors, ok_indices, failed_indices, notes, manifest):
    """Write the normalized matrix, track index, extraction status, and a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
 
    kept_ids = [track_rows[i]["track_id"] for i in ok_indices]
    normalized = l2_normalize(vectors[ok_indices])
 
    np.save(out_dir / "track_embeddings.npy", normalized, allow_pickle=False)
 
    write_table(
        out_dir / "track_index.csv",
        "track_index",
        [{"row_index": str(row), "track_id": tid} for row, tid in enumerate(kept_ids)],
    )
 
    status_rows = []
    failed_set = set(failed_indices)
    for i, row in enumerate(track_rows):
        if i in failed_set:
            code, message = notes[i]
            status_rows.append({
                "track_id": row["track_id"],
                "status": "failed",
                "error_code": code,
                "error_message": message,
            })
        else:
            status_rows.append({
                "track_id": row["track_id"],
                "status": "ok",
                "error_code": "",
                "error_message": "",
            })
    write_table(out_dir / "extraction_status.csv", "extraction_status", status_rows)
 
    adapted_manifest = {
        "schema_version": "1.0",
        "produced_by": "scripts/adapt_embeddings.py",
        "source_manifest": {
            "model_name": manifest.get("model_name"),
            "sample_rate": manifest.get("sample_rate"),
            "hidden_layer": manifest.get("hidden_layer"),
            "pooling": manifest.get("pooling"),
            "l2_normalization": manifest.get("l2_normalization"),
        },
        "transformations": [
            "removed rows whose embeddings were zero or non-finite",
            "L2-normalized every retained row",
            "emitted an explicit track_index.csv so row order is no longer load-bearing",
            "emitted extraction_status.csv covering every input track",
        ],
        "input_track_count": len(track_rows),
        "retained_track_count": len(kept_ids),
        "failed_track_count": len(failed_indices),
        "failed_track_ids": [track_rows[i]["track_id"] for i in failed_indices],
        "embedding_shape": list(normalized.shape),
        "embedding_dtype": str(normalized.dtype),
        "l2_normalized": True,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(adapted_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return normalized, kept_ids
 
 
def verify(out_dir: Path, expected_ids):
    """Re-read the written artifacts through the shared loader the way C will."""
    from album_similarity.utils import load_indexed_vectors
 
    ids, vectors = load_indexed_vectors(
        out_dir / "track_embeddings.npy", out_dir / "track_index.csv", "track_id"
    )
    if ids != expected_ids:
        raise ValueError("written index does not match the retained track IDs")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"rows are not unit norm: min={norms.min()}, max={norms.max()}")
    read_table(out_dir / "extraction_status.csv", "extraction_status")
    return ids, vectors
 
 
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Adapt Person B's raw embeddings into contract form for Person C."
    )
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--tracks-csv", default="outputs/tracks.csv",
                        help="the exact tracks.csv the embedding matrix is aligned to")
    parser.add_argument("--matrix", default="data/embeddings/track_embeddings.npy")
    parser.add_argument("--manifest", default="data/embeddings/manifest.json")
    parser.add_argument("--out-dir", default="data/embeddings/default",
                        help="where configs/representation.yaml expects the embeddings")
    parser.add_argument("--check-only", action="store_true",
                        help="inspect and report without writing anything")
    args = parser.parse_args(argv)
 
    root = args.project_root.resolve()
    tracks_csv = root / args.tracks_csv
    matrix_path = root / args.matrix
    manifest_path = root / args.manifest
    out_dir = root / args.out_dir
 
    track_rows, vectors, manifest = load_source(tracks_csv, matrix_path, manifest_path)
    ok_indices, failed_indices, notes = classify_rows(track_rows, vectors, manifest)
 
    print(f"source matrix     : {vectors.shape} {vectors.dtype}")
    print(f"source tracks.csv : {len(track_rows)} rows ({tracks_csv.relative_to(root)})")
    print(f"failed extractions: {len(failed_indices)}")
    print(f"retained tracks   : {len(ok_indices)}")
 
    # Albums that lose eligibility once failures are dropped. Reported here because the
    # number belongs in C's write-up, but the authoritative filter is
    # representation.filter_eligible_albums, which works from A's albums.csv.
    from collections import Counter
    before = Counter(row["album_id"] for row in track_rows)
    after = Counter(track_rows[i]["album_id"] for i in ok_indices)
    for threshold in (3,):
        lost = sorted(
            album for album in before
            if before[album] >= threshold and after[album] < threshold
        )
        print(f"albums dropping below min_tracks={threshold}: {len(lost)} {lost}")
 
    if args.check_only:
        print("\n--check-only: nothing written")
        return 0
 
    _, kept_ids = write_outputs(
        out_dir, track_rows, vectors, ok_indices, failed_indices, notes, manifest
    )
    ids, matrix = verify(out_dir, kept_ids)
    print(f"\nwrote {out_dir.relative_to(root)}/")
    print(f"  track_embeddings.npy  {matrix.shape} {matrix.dtype}, unit-norm verified")
    print(f"  track_index.csv       {len(ids)} rows")
    print(f"  extraction_status.csv {len(track_rows)} rows")
    print(f"  manifest.json")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 