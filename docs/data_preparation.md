# FMA data preparation — Person A handoff

This implements the existing `album_similarity.data` module without changing
`contracts.py`, `utils.py`, or other team members' modules.

## 1. Inputs

The repository already uses these paths in `configs/data.yaml`:

- `data/raw/fma_metadata/tracks.csv` — official FMA metadata, two-level CSV header
- `data/raw/fma_metadata/genres.csv` — genre ID, title, parent hierarchy
- `data/raw/fma_medium/<3-digit-prefix>/<6-digit-track-id>.mp3` — MP3 files

Get the **FMA Medium** archive and metadata from the [official FMA repository](https://github.com/mdeff/fma).
The official medium audio archive is approximately 22 GiB, so check disk space.
Keep the large audio archive and extracted audio **outside Git**.

### Setup

From the repository root:

```bash
python -m pip install -e .
python -m pip install pytest  # for the supplied tests
python -m album_similarity.data --config configs/data.yaml --check-config
python -m pytest -q tests/test_data.py
```

`configs/data.yaml` currently has `min_tracks: null` intentionally. Choose this
threshold **after inspecting actual audio availability and album-size distribution**.
It must be a positive integer. A threshold of 3 is useful as a *candidate* because
RQ1 compares one and three tracks, but it is **not** an approved project threshold.
Set `min_tracks` after discussion and then run:

```bash
python -m album_similarity.data --config configs/data.yaml
```

### Inspect the real-data cohort before choosing a threshold

```bash
python - <<'PY'
from collections import Counter
from album_similarity.data import load_metadata, validate_audio_files

raw = load_metadata('data/raw/fma_metadata')
tracks, excluded = validate_audio_files(raw['tracks'], 'data/raw/fma_medium')
counts = Counter(track['album_id'] for track in tracks)
print('Metadata tracks:', len(raw['tracks']))
print('Audio candidates passing basic checks:', len(tracks))
print('Audio exclusions:', len(excluded))
for n in (1, 3, 5, 8, 10):
    n_albums = sum(n_tracks >= n for n_tracks in counts.values())
    n_tracks = sum(n_tracks for n_tracks in counts.values() if n_tracks >= n)
    print(f'min_tracks={n}: albums={n_albums}, tracks={n_tracks}')
PY
```

## 2. Outputs

The data pipeline writes these into `data/prepared/`:

| File | Contract |
|---|---|
| `tracks.csv` | `track_id,album_id,artist_id,audio_path,genre_ids` |
| `albums.csv` | `album_id,title,artist_ids,track_ids,genre_ids,usable_track_count` |
| `genres.csv` | `genre_id,name,parent_id,column_index` |
| `album_genre_index.csv` | `row_index,album_id,has_genres` |
| `album_genres.npy` | `float32` matrix `[num_albums, num_genres]`, row-aligned with `album_genre_index.csv` |
| `quality_report.json` | counts, exclusions, unknown genre IDs, filtering decisions, validation |
| `manifest.json` | schema version, configuration summary, files, shapes |

All IDs are strings. All list-valued CSV cells use JSON arrays via the repository's
existing `write_table` helper. `audio_path` is project-root-relative, so Person B
should resolve it against the same `project_root` before calling `load_audio()`.

## 3. Shared interfaces and responsibilities

`prepare_metadata(raw_metadata_dir, audio_dir, config)` returns a catalog of
cleaned tracks, albums, genre definitions, and a report.

`build_genre_representation(albums, genres)` returns a float32 matrix and a
row-aligned index. `run_pipeline(config, project_root)` writes the artifacts.

**Handoff to B:** `tracks.csv` includes validated candidate audio paths.
The data module only checks file presence, nonzero length, and a lightweight MP3
signature; it cannot guarantee decoding. B must record any subsequent extraction
failures in `extraction_status.csv`, not substitute zero embeddings.

**Handoff to C/D:** Use `album_id` and `track_id` to join tables and vectors.
Do not assume row order unless an explicit index accompanies the matrix.
Album genre vectors are unions of the retained usable tracks' labels.

## 4. Current limitations and outstanding decisions

- A full FMA run has **not** been performed because the raw ~22 GiB audio archive
  and metadata are not included with this deliverable.
- The minimum eligible track count `min_tracks` must be chosen from real data.
- Raw FMA genre IDs are preserved; invalid IDs are removed and reported, and
  parent expansion is optional. Semantic synonym merges are intentionally
  **not** invented without an agreed genre mapping.
- This module does not decode complete MP3s or compute MERT embeddings.
- Confirm the output filename `album_genres.npy` and common path conventions
  with B/C/D before freezing the shared artifact interface.

## 5. Acceptance checklist

- `python -m pytest -q tests/test_data.py` succeeds
- All prepared CSVs load with `album_similarity.utils.read_table`
- `album_genres.npy` is `float32` and aligns with `album_genre_index.csv`
- Every retained track belongs to exactly one retained album
- Missing audio and unknown genres are reported; no data are silently fabricated
- Running twice with identical files/config creates byte-identical outputs
- Actual FMA cohort counts, threshold choice, and full-run validation are recorded
