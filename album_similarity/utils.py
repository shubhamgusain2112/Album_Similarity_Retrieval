"""Shared configuration, CLI and CSV/vector I/O. No import-time file or model loading."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .contracts import TABLE_COLUMNS, LIST_COLUMNS

MODULES = ("data", "audio", "representation", "retrieval", "evaluation", "ui")

def load_config(path, module):
    """Take a YAML path and module name; validate the schema and relative paths, then return
    (config dictionary, project root Path).

    Args:
        path: Path to the input or output file.
        module: Business module name: data, audio, representation, retrieval, evaluation, or
            ui.

    Notes:
        Implemented shared helper; no business algorithm is run implicitly.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required. Install project dependencies with: python -m pip install -e .") from exc
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("module") != module:
        raise ValueError(f"configuration must declare module: {module}")
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported schema_version")
    root = (config_path.parent / config["project_root"]).resolve()
    for key, value in config.get("paths", {}).items():
        relative = Path(value)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"paths.{key} must be relative to the project root")
    return config, root


def write_table(path, table, rows):
    """Write records as CSV using TABLE_COLUMNS, encoding list cells as JSON; return None.

    Args:
        path: Path to the input or output file.
        table: Table name defined in contracts.TABLE_COLUMNS.
        rows: Record dictionaries whose fields match the selected table contract.

    Notes:
        Implemented shared helper; no business algorithm is run implicitly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = TABLE_COLUMNS[table]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ValueError(f"{table}: row columns must match {columns}")
            encoded = dict(row)
            for key in LIST_COLUMNS.intersection(columns):
                value = row[key]
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError(f"{key}: expected list[str]")
                encoded[key] = json.dumps(value, ensure_ascii=False)
            for key in ("has_genres",):
                if key in encoded:
                    if type(encoded[key]) is not bool:
                        raise ValueError(f"{key}: expected boolean")
                    encoded[key] = "true" if encoded[key] else "false"
            writer.writerow(encoded)


def read_table(path, table):
    """Validate CSV against the table contract and decode list columns; return records with
    scalar cells kept as strings.

    Args:
        path: Path to the input or output file.
        table: Table name defined in contracts.TABLE_COLUMNS.

    Notes:
        Implemented shared helper; no business algorithm is run implicitly.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS[table]:
            raise ValueError(f"{table}: unexpected CSV header")
        rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed CSV row")
        for key in LIST_COLUMNS.intersection(row):
            value = json.loads(row[key])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{key}: expected JSON array of strings")
            row[key] = value
    # Scalars deliberately remain strings. Each module casts its numeric fields explicitly.
    return rows


def load_indexed_vectors(matrix_path, index_path, id_column):
    """Validate a float32 matrix and its independent index; return (ordered IDs, two-
    dimensional NumPy matrix).

    Args:
        matrix_path: Path to a float32 two-dimensional .npy embedding matrix.
        index_path: Path to the independent CSV row index for that matrix.
        id_column: ID column identifying track_id or album_id.

    Notes:
        Implemented shared helper; no business algorithm is run implicitly.
    """
    table = {"track_id": "track_index", "album_id": "album_index"}[id_column]
    rows = read_table(index_path, table)
    vectors = np.load(matrix_path, allow_pickle=False)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite float32 matrix")
    if len(rows) != len(vectors):
        raise ValueError("matrix/index row count mismatch")
    if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("row_index must be contiguous and in matrix order")
    ids = [row[id_column] for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("IDs must be nonempty and unique")
    return ids, vectors


def run_module(module, runner, argv=None):
    """Parse CLI arguments, validate config, then invoke runner(config, project_root). --check-
    config never invokes runner. The internal UI --render branch is reserved for Streamlit;
    normal UI invocation delegates server launch to runner.

    Args:
        module: Business module name: data, audio, representation, retrieval, evaluation, or
            ui.
        runner: Callable taking (config, project_root) that executes the module workflow.
        argv: Command-line argument list, or None to use the process arguments.

    Notes:
        Implemented shared helper; no business algorithm is run implicitly.
    """
    parser = argparse.ArgumentParser(description=f"Album similarity: {module}")
    parser.add_argument("--config", default=f"configs/{module}.yaml")
    parser.add_argument("--check-config", action="store_true", help="validate configuration without running the pipeline")
    if module == "ui":
        parser.add_argument("--render", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config, root = load_config(args.config, module)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.check_config:
        print(f"Configuration OK: {module}; project root: {root}")
        return
    try:
        if module == "ui" and args.render:
            from album_similarity.ui import build_app
            return build_app(config, root)
        return runner(config, root)
    except NotImplementedError as exc:
        parser.exit(2, f"{module}: {exc}\n")
