"""Ui functions. Business algorithms are scaffolds."""
from album_similarity.utils import run_module


def load_demo_data(config, project_root):
    """Read precomputed artifacts and optional experiment summaries; return
    artifacts/album_metadata/experiment_results without running MERT.

    Args:
        config: Configuration dictionary for this workflow or operation.
        project_root: Resolved project root Path used to interpret configured artifact
            paths.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.load_demo_data: not implemented")


def render_query_controls(album_metadata, config):
    """Render query, aggregation, track setting, seed, weight, and artist-exclusion controls;
    return a validated RetrievalRequest.

    Args:
        album_metadata: Album metadata keyed by album_id for display joins.
        config: Configuration dictionary for this workflow or operation.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_query_controls: not implemented")


def render_query_details(request, response, album_metadata):
    """Display the query album, actual tracks, seed, diversity, and missing-label warnings;
    update the page and return None.

    Args:
        request: Validated RetrievalRequest containing the query and retrieval settings.
        response: RetrievalResponse containing the selected tracks, diversity, results, and
            warnings.
        album_metadata: Album metadata keyed by album_id for display joins.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_query_details: not implemented")


def render_results(response, album_metadata):
    """Display Top-K, audio/genre/fusion scores, and shared genres; join metadata only by
    album_id and return None.

    Args:
        response: RetrievalResponse containing the selected tracks, diversity, results, and
            warnings.
        album_metadata: Album metadata keyed by album_id for display joins.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_results: not implemented")


def render_surprise_controls(config):
    """Render the Surprise button and strategy selector; return None before a click, otherwise
    surprise_a or surprise_b. Do not silently choose an undecided default.

    Args:
        config: Configuration dictionary for this workflow or operation.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_surprise_controls: not implemented")


def render_surprise_result(response, album_metadata):
    """Display the Surprise recommendation and cross-modal explanation, or explain an empty
    result; return None.

    Args:
        response: RetrievalResponse containing the selected tracks, diversity, results, and
            warnings.
        album_metadata: Album metadata keyed by album_id for display joins.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_surprise_result: not implemented")


def render_experiment_summary(experiment_results):
    """Display existing experiment plots and summaries, reporting missing data without
    rerunning experiments; return None.

    Args:
        experiment_results: Experiment tables, analysis, summaries, and recorded skipped
            cases.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.render_experiment_summary: not implemented")


def build_app(config, project_root):
    """Build the Streamlit page using precomputed data, controls, and shared retrieve; return
    None without launching a nested server.

    Args:
        config: Configuration dictionary for this workflow or operation.
        project_root: Resolved project root Path used to interpret configured artifact
            paths.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.build_app: not implemented")


def launch(config, project_root):
    """Explicitly launch Streamlit to execute this file via build_app; use the internal
    --render branch to prevent repeated launches and return None.

    Args:
        config: Configuration dictionary for this workflow or operation.
        project_root: Resolved project root Path used to interpret configured artifact
            paths.

    Raises:
        NotImplementedError: This function is a scaffold and has not been implemented.

    Notes:
        Outputs described above are the intended contract, not current results.
    """
    raise NotImplementedError("ui.launch: not implemented")


def run_pipeline(config, project_root):
    """Run the UI CLI workflow via launch; the internal render branch calls build_app directly.
    Return None.

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
    raise NotImplementedError("ui.run_pipeline: not implemented")


def main(argv=None):
    """The sole CLI entry point; supports --config, --help, and --check-config.

    Args:
        argv: Command-line argument list, or None to use the process arguments.

    Notes:
        The CLI is wired; normal execution still calls an unimplemented workflow.
        --check-config validates configuration without running that workflow.
    """
    return run_module("ui", run_pipeline, argv)


if __name__ == "__main__":
    main()
