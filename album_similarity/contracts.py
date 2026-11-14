"""Versioned public contracts, with no command-line entry point or business algorithms.

RetrievalRequest, RetrievalResult, and RetrievalResponse define the shared request
and response format. SCHEMA_VERSION, type aliases, TABLE_COLUMNS, and LIST_COLUMNS
define the storage contract. IDs are strings; vectors are float32 NumPy arrays.
Business modules depend on these contracts, never the other way around.
"""
from dataclasses import dataclass, field
from typing import Literal
import math

SCHEMA_VERSION = "1.0"
TrackSetting = Literal["one", "three", "half", "full"]
Aggregation = Literal["mean", "medoid"]
RetrievalMode = Literal["standard", "surprise_a", "surprise_b"]

@dataclass(frozen=True)
class RetrievalRequest:
    query_album_id: str
    aggregation: Aggregation = "mean"
    track_setting: TrackSetting = "full"
    seed: int = 42
    genre_weight: float = 0.5
    top_k: int = 5
    exclude_same_artist: bool = False
    mode: RetrievalMode = "standard"

    def __post_init__(self):
        """Validate a newly constructed request; this validation is implemented.

        Raises:
            ValueError: An ID, method, mode, weight, count, seed, or exclusion flag
                violates the public request contract.
        """
        if not isinstance(self.query_album_id, str) or not self.query_album_id:
            raise ValueError("query_album_id must be a nonempty string")
        if self.aggregation not in ("mean", "medoid"):
            raise ValueError("unknown aggregation")
        if self.track_setting not in ("one", "three", "half", "full"):
            raise ValueError("unknown track_setting")
        if self.mode not in ("standard", "surprise_a", "surprise_b"):
            raise ValueError("unknown retrieval mode")
        if not math.isfinite(self.genre_weight) or not 0 <= self.genre_weight <= 1:
            raise ValueError("genre_weight must be finite and in [0, 1]")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.exclude_same_artist) is not bool:
            raise ValueError("exclude_same_artist must be boolean")

@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    album_id: str
    audio_similarity_raw: float
    audio_score: float
    genre_score: float | None
    final_score: float
    shared_genre_ids: list[str] = field(default_factory=list)
    surprise_score: float | None = None

@dataclass(frozen=True)
class RetrievalResponse:
    query_album_id: str
    selected_track_ids: list[str]
    actual_track_count: int
    diversity: float
    results: list[RetrievalResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

# Every column is required unless its value is explicitly allowed to be empty.
TABLE_COLUMNS = {
    "tracks": ("track_id", "album_id", "artist_id", "audio_path", "genre_ids"),
    "albums": ("album_id", "title", "artist_ids", "track_ids", "genre_ids", "usable_track_count"),
    "genres": ("genre_id", "name", "parent_id", "column_index"),
    "album_genre_index": ("row_index", "album_id", "has_genres"),
    "track_index": ("row_index", "track_id"),
    "extraction_status": ("track_id", "status", "error_code", "error_message"),
    "eligible_albums": ("album_id", "valid_track_ids", "valid_track_count", "has_genres"),
    "album_index": ("row_index", "album_id"),
    "album_stats": ("album_id", "valid_track_count", "diversity_mean_cosine_distance"),
    "sampling_plan": ("album_id", "setting", "repeat_id", "seed", "selected_track_ids", "actual_track_count"),
    "stability": ("album_id", "aggregation", "track_setting", "actual_track_count", "repeat_id", "seed", "top5_overlap", "embedding_cosine", "diversity"),
    "weight_sensitivity": ("album_id", "aggregation", "genre_weight", "previous_genre_weight", "mean_audio_similarity_raw", "mean_genre_score", "top5_overlap_previous", "retrieved_album_ids"),
    "surprise_comparison": ("album_id", "strategy", "candidate_count", "recommended_album_id", "audio_similarity_raw", "genre_score", "surprise_score", "evaluation_note"),
    "summary": ("experiment", "metric", "group", "count", "mean", "std"),
}
LIST_COLUMNS = {"genre_ids", "artist_ids", "track_ids", "valid_track_ids", "selected_track_ids", "retrieved_album_ids"}
