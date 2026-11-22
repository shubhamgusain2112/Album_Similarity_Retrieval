"""Prepare FMA track, album and hierarchical genre metadata.

FMA metadata uses a two-row CSV header, unlike the flattened CSV artifacts
written by this module. No audio decoding or embedding extraction occurs here.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from album_similarity.contracts import SCHEMA_VERSION
from album_similarity.utils import run_module, write_table


def _id(value):
    value = str(value).strip() if value is not None else ""
    if not value or value.lower() in {"nan", "none", "null"}:
        return ""
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value


def _sort_id(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def _genre_ids(value, track_id):
    """Read FMA's Python-literal list without using eval."""
    if value is None or not str(value).strip() or str(value).strip().lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"track {track_id}: invalid genre list {value!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"track {track_id}: genres must be a list")
    result = [_id(item) for item in parsed]
    if any(not item for item in result):
        raise ValueError(f"track {track_id}: invalid genre ID")
    return sorted(set(result), key=_sort_id)


def load_metadata(raw_metadata_dir):
    """Read FMA tracks.csv (two header rows) and genres.csv into typed records."""
    directory = Path(raw_metadata_dir)
    track_file, genre_file = directory / "tracks.csv", directory / "genres.csv"
    if not track_file.is_file() or not genre_file.is_file():
        raise FileNotFoundError(f"FMA metadata requires {track_file} and {genre_file}")

    with track_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            groups = next(reader)
            fields = next(reader)
        except StopIteration as exc:
            raise ValueError("FMA tracks.csv must have two header rows") from exc
        if len(groups) != len(fields):
            raise ValueError("FMA track header rows have different lengths")
        columns = {(group.strip(), name.strip()): index
                   for index, (group, name) in enumerate(zip(groups, fields))}
        required = [("album", "id"), ("album", "title"),
                    ("artist", "id"), ("track", "genres")]
        missing = [str(key) for key in required if key not in columns]
        if missing:
            raise ValueError(f"FMA tracks.csv missing fields: {', '.join(missing)}")

        records = []
        seen = set()
        ignored_missing_ids = 0
        for line_no, row in enumerate(reader, start=3):
            if not row or not any(row):
                continue
            if len(row) != len(groups):
                raise ValueError(f"FMA tracks.csv line {line_no}: unexpected column count")
            track_id = _id(row[0])
            album_id = _id(row[columns[("album", "id")]])
            artist_id = _id(row[columns[("artist", "id")]])
            if not track_id or not album_id or not artist_id:
                ignored_missing_ids += 1
                continue
            if track_id in seen:
                raise ValueError(f"duplicate FMA track_id: {track_id}")
            seen.add(track_id)
            records.append({
                "track_id": track_id,
                "album_id": album_id,
                "artist_id": artist_id,
                "genre_ids": _genre_ids(row[columns[("track", "genres")]], track_id),
                "album_title": row[columns[("album", "title")]].strip(),
            })

    with genre_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"title", "parent"}.issubset(reader.fieldnames or []):
            raise ValueError("FMA genres.csv requires title and parent columns")
        first_column = reader.fieldnames[0]
        genres = []
        genre_seen = set()
        for row in reader:
            genre_id = _id(row[first_column])
            if not genre_id:
                continue
            if genre_id in genre_seen:
                raise ValueError(f"duplicate genre_id: {genre_id}")
            genre_seen.add(genre_id)
            parent_id = _id(row["parent"])
            genres.append({
                "genre_id": genre_id,
                "name": (row["title"] or "").strip(),
                "parent_id": "" if parent_id == "0" else parent_id,
            })
    if not genres:
        raise ValueError("FMA genres.csv contains no genres")
    genres.sort(key=lambda genre: _sort_id(genre["genre_id"]))
    for index, genre in enumerate(genres):
        genre["column_index"] = index
    return {"tracks": records, "genres": genres,
            "report": {"metadata_track_count": len(records),
                       "metadata_missing_id_rows": ignored_missing_ids,
                       "genre_vocabulary_count": len(genres)}}


def validate_audio_files(tracks, audio_dir):
    """Check expected FMA MP3 paths and basic file headers, without decoding audio."""
    root = Path(audio_dir)
    valid, excluded = [], []
    for track in tracks:
        track_id = track["track_id"]
        if not track_id.isdigit():
            excluded.append({"track_id": track_id, "reason": "non_numeric_track_id"})
            continue
        name = f"{int(track_id):06d}"
        path = root / name[:3] / f"{name}.mp3"
        if not path.is_file():
            excluded.append({"track_id": track_id, "reason": "missing_audio"})
            continue
        try:
            if path.stat().st_size < 4:
                excluded.append({"track_id": track_id, "reason": "empty_or_short_audio"})
                continue
            with path.open("rb") as handle:
                prefix = handle.read(4096)
        except OSError:
            excluded.append({"track_id": track_id, "reason": "unreadable_audio"})
            continue
        # Lightweight signature check only; MERT preprocessing must still decode each file
        has_mpeg_frame = any(
            prefix[i] == 0xFF and (prefix[i + 1] & 0xE0) == 0xE0
            and (prefix[i + 1] & 0x06) != 0
            for i in range(len(prefix) - 1)
        )
        if not (prefix.startswith(b"ID3") or has_mpeg_frame):
            excluded.append({"track_id": track_id, "reason": "invalid_mp3_header"})
            continue
        valid.append({
            "track_id": track_id,
            "album_id": track["album_id"],
            "artist_id": track["artist_id"],
            "audio_path": str(path),
            "genre_ids": list(track["genre_ids"]),
            "album_title": track.get("album_title", ""),
        })
    return valid, excluded


def clean_genres(tracks, genres, include_parents=False):
    """Preserve known fine-grained labels and optionally add valid ancestors."""
    vocabulary = {genre["genre_id"]: genre for genre in genres}
    for genre in genres:
        parent = genre["parent_id"]
        if parent and parent not in vocabulary:
            raise ValueError(f"genre {genre['genre_id']}: unknown parent {parent}")
    def with_ancestors(genre_id):
        result, visited = [], set()
        current = genre_id
        while current:
            if current in visited:
                raise ValueError(f"cyclic genre parent chain at {current}")
            visited.add(current)
            result.append(current)
            current = vocabulary[current]["parent_id"] if include_parents else ""
        return result

    cleaned = []
    for track in tracks:
        labels = set()
        for genre_id in track["genre_ids"]:
            if genre_id in vocabulary:
                labels.update(with_ancestors(genre_id))
        cleaned.append({**track, "genre_ids": sorted(labels, key=_sort_id)})
    return cleaned, [dict(item) for item in genres]


def build_album_catalog(tracks, min_tracks):
    """Group validated tracks into albums, excluding groups below min_tracks."""
    if type(min_tracks) is not int or min_tracks < 1:
        raise ValueError("min_tracks must be a positive integer; inspect the dataset and set configs/data.yaml")
    groups = defaultdict(list)
    for track in tracks:
        groups[track["album_id"]].append(track)
    albums, excluded = [], []
    for album_id in sorted(groups, key=_sort_id):
        members = groups[album_id]
        if len(members) < min_tracks:
            excluded.append({"album_id": album_id, "reason": "too_few_usable_tracks",
                             "usable_track_count": len(members)})
            continue
        names = sorted({member.get("album_title", "") for member in members
                        if member.get("album_title", "")})
        albums.append({
            "album_id": album_id,
            "title": names[0] if names else "",
            "artist_ids": sorted({member["artist_id"] for member in members}, key=_sort_id),
            "track_ids": sorted([member["track_id"] for member in members], key=_sort_id),
            "genre_ids": sorted({tag for member in members for tag in member["genre_ids"]},
                                key=_sort_id),
            "usable_track_count": len(members),
        })
    return albums, excluded


def prepare_metadata(raw_metadata_dir, audio_dir, config):
    """Prepare an eligible FMA cohort and record filtering decisions."""
    min_tracks = config.get("min_tracks")
    if type(min_tracks) is not int or min_tracks < 1:
        raise ValueError("Set configs/data.yaml min_tracks to an agreed positive integer before preparation")
    raw = load_metadata(raw_metadata_dir)
    valid_tracks, excluded_audio = validate_audio_files(raw["tracks"], audio_dir)
    known_genres = {genre["genre_id"] for genre in raw["genres"]}
    unknown_assignments = Counter(genre_id for track in valid_tracks
                                  for genre_id in track["genre_ids"]
                                  if genre_id not in known_genres)
    cleaned_tracks, genres = clean_genres(valid_tracks, raw["genres"],
                                          config.get("genres", {}).get("include_parents", False))
    albums, excluded_albums = build_album_catalog(cleaned_tracks, min_tracks)
    retained_album_ids = {album["album_id"] for album in albums}
    tracks = sorted((track for track in cleaned_tracks
                     if track["album_id"] in retained_album_ids),
                    key=lambda track: _sort_id(track["track_id"]))
    audio_failure_counts = dict(sorted(Counter(item["reason"] for item in excluded_audio).items()))
    report = {
        **raw["report"],
        "audio_valid_track_count": len(valid_tracks),
        "audio_exclusions": audio_failure_counts,
        "albums_below_min_tracks": len(excluded_albums),
        "prepared_album_count": len(albums),
        "prepared_track_count": len(tracks),
        "tracks_without_genres": sum(not item["genre_ids"] for item in tracks),
        "unknown_genre_assignments_removed": sum(unknown_assignments.values()),
        "unknown_genre_ids": sorted(unknown_assignments, key=_sort_id),
        "min_tracks": min_tracks,
        "include_parent_genres": bool(config.get("genres", {}).get("include_parents", False)),
        "audio_validation": "existence + nonempty file + MP3 signature; not full decoding",
        "decisions": ["Only tracks with an existing readable MP3 candidate are retained",
                      "Album genres are the union of usable track genre labels",
                      "Unknown genre IDs are removed without inventing replacements"],
    }
    catalog = {"tracks": tracks, "albums": albums, "genres": genres, "report": report}
    report["validation"] = validate_catalog(tracks, albums)
    return catalog


def build_genre_representation(albums, genres):
    """Build a float32 [albums, genres] union-label matrix and row-aligned index."""
    columns = {genre["genre_id"]: int(genre["column_index"]) for genre in genres}
    if set(columns.values()) != set(range(len(genres))):
        raise ValueError("genre column_index must be contiguous and unique")
    matrix = np.zeros((len(albums), len(genres)), dtype=np.float32)
    index = []
    for row_index, album in enumerate(albums):
        for genre_id in album["genre_ids"]:
            if genre_id not in columns:
                raise ValueError(f"album {album['album_id']}: unknown genre {genre_id}")
            matrix[row_index, columns[genre_id]] = 1.0
        index.append({"row_index": row_index, "album_id": album["album_id"],
                      "has_genres": bool(album["genre_ids"])})
    return matrix, index


def validate_catalog(tracks, albums):
    """Reject duplicate IDs, inconsistent membership or incorrect union labels."""
    track_ids = [row["track_id"] for row in tracks]
    album_ids = [row["album_id"] for row in albums]
    if len(track_ids) != len(set(track_ids)) or len(album_ids) != len(set(album_ids)):
        raise ValueError("track and album IDs must be unique")
    track_map = {row["track_id"]: row for row in tracks}
    represented = []
    for album in albums:
        ids = album["track_ids"]
        if len(ids) != len(set(ids)) or len(ids) != int(album["usable_track_count"]):
            raise ValueError(f"album {album['album_id']}: incorrect track count")
        if any(track_id not in track_map or
               track_map[track_id]["album_id"] != album["album_id"] for track_id in ids):
            raise ValueError(f"album {album['album_id']}: invalid track membership")
        expected_artists = {track_map[track_id]["artist_id"] for track_id in ids}
        expected_genres = {genre_id for track_id in ids
                           for genre_id in track_map[track_id]["genre_ids"]}
        if set(album["artist_ids"]) != expected_artists:
            raise ValueError(f"album {album['album_id']}: artist IDs disagree with tracks")
        if set(album["genre_ids"]) != expected_genres:
            raise ValueError(f"album {album['album_id']}: genre IDs disagree with tracks")
        represented.extend(ids)
    if set(represented) != set(track_ids) or len(represented) != len(track_ids):
        raise ValueError("every prepared track must belong to exactly one album")
    return {"valid": True, "track_count": len(tracks), "album_count": len(albums)}


def save_prepared_data(catalog, genre_matrix, genre_index, config, output_dir):
    """Persist all prepared tables, an aligned genre matrix and an audit manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tracks = []
    raw_root = Path(config["paths"]["audio_dir"])
    project_root = output.resolve()
    for _ in Path(config["paths"]["output_dir"]).parts:
        project_root = project_root.parent
    for track in catalog["tracks"]:
        row = {key: track[key] for key in
               ("track_id", "album_id", "artist_id", "audio_path", "genre_ids")}
        path = Path(row["audio_path"])
        if path.is_absolute():
            # Persist project-relative paths so artifacts work on other machines
            try:
                relative_audio = path.relative_to((project_root / raw_root).resolve())
            except ValueError as exc:
                raise ValueError(f"audio path is outside configured audio directory: {path}") from exc
            row["audio_path"] = (raw_root / relative_audio).as_posix()
        tracks.append(row)
    write_table(output / "tracks.csv", "tracks", tracks)
    write_table(output / "albums.csv", "albums", catalog["albums"])
    write_table(output / "genres.csv", "genres", catalog["genres"])
    write_table(output / "album_genre_index.csv", "album_genre_index", genre_index)
    if (genre_matrix.dtype != np.float32 or genre_matrix.ndim != 2 or
            genre_matrix.shape != (len(catalog["albums"]), len(catalog["genres"]))):
        raise ValueError("genre matrix shape or dtype violates data contract")
    np.save(output / "album_genres.npy", genre_matrix, allow_pickle=False)
    (output / "quality_report.json").write_text(
        json.dumps(catalog["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "module": "data",
        "min_tracks": config["min_tracks"],
        "include_parent_genres": bool(config.get("genres", {}).get("include_parents", False)),
        "num_tracks": len(tracks),
        "num_albums": len(catalog["albums"]),
        "num_genres": len(catalog["genres"]),
        "genre_matrix_shape": list(genre_matrix.shape),
        "files": {name: name for name in (
            "tracks.csv", "albums.csv", "genres.csv", "album_genre_index.csv",
            "album_genres.npy", "quality_report.json")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_pipeline(config, project_root):
    """Run the data module using relative paths from the configuration."""
    root = Path(project_root).resolve()
    paths = config["paths"]
    catalog = prepare_metadata(root / paths["metadata_dir"], root / paths["audio_dir"], config)
    matrix, index = build_genre_representation(catalog["albums"], catalog["genres"])
    return save_prepared_data(catalog, matrix, index, config, root / paths["output_dir"])


def main(argv=None):
    """Run the existing shared CLI contract."""
    return run_module("data", run_pipeline, argv)


if __name__ == "__main__":
    main()
