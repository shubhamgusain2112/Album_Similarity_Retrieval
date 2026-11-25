
"""
MERT audio embedding extraction.

Input:
    A's prepared tracks.csv

Output:
    data/embeddings/track_embeddings.npy
    data/embeddings/manifest.json

Each row in track_embeddings.npy corresponds exactly to the
same row in the input tracks.csv.
"""

import argparse
import json
import re
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "m-a-p/MERT-v1-95M"

# Your actual FMA data location on the server.
# The relative path extracted from tracks.csv starts with "raw/".
AUDIO_ROOT = Path("/local/s4854756/fma")

# Project root: this file is assumed to be in album_similarity/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default input and output paths.
TRACKS_CSV = PROJECT_ROOT / "outputs" / "tracks.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

EMBEDDING_DIM = 768

# Print progress every N tracks.
LOG_INTERVAL = 100


# ============================================================
# Path resolution
# ============================================================

def resolve_audio_path(audio_path):
    """
    Extract 'raw/...' from A's relative audio_path and
    prepend AUDIO_ROOT.

    Example:
        data/raw/fma_medium/000/000002.mp3

    Becomes:
        /local/s4854756/fma/raw/fma_medium/000/000002.mp3
    """

    audio_path = str(audio_path).replace("\\", "/")

    match = re.search(r"raw/.*$", audio_path)

    if match is None:
        raise ValueError(
            f"Cannot extract raw/ path from: {audio_path}"
        )

    relative_path = Path(match.group(0))
    real_path = AUDIO_ROOT / relative_path

    if not real_path.is_file():
        raise FileNotFoundError(
            f"Audio file does not exist: {real_path}"
        )

    return real_path


# ============================================================
# Model loading
# ============================================================

def load_mert():
    """Load the pretrained MERT model and feature extractor."""

    print(f"Loading model: {MODEL_NAME}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    sample_rate = feature_extractor.sampling_rate

    if sample_rate != 24000:
        raise ValueError(
            f"Unexpected MERT sampling rate: {sample_rate}. "
            "Please verify the model configuration."
        )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model.eval()
    model.to(device)

    hidden_size = model.config.hidden_size

    if hidden_size != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIM}, "
            f"but model has hidden size {hidden_size}."
        )

    print(f"Sample rate: {sample_rate}")
    print(f"Hidden size: {hidden_size}")
    print("MERT model loaded successfully.\n")

    return model, feature_extractor, device, sample_rate


# ============================================================
# Single-track embedding extraction
# ============================================================

def extract_embedding(
    audio_path,
    model,
    feature_extractor,
    device,
    sample_rate
):
    """
    Extract one embedding for an entire audio track.

    Steps:
        1. Load the entire audio.
        2. Convert to mono and resample to MERT's sample rate.
        3. Extract MERT hidden states.
        4. Apply temporal mean pooling.
        5. Return a 768-dimensional float32 vector.
    """

    audio, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        # Shape before pooling: [1, T, 768]
        hidden_states = outputs.last_hidden_state

        # Mean pooling over the temporal dimension.
        # Shape after pooling: [1, 768]
        embedding = hidden_states.mean(dim=1)

    embedding = embedding.squeeze(0).cpu().numpy()
    embedding = embedding.astype(np.float32)

    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or Inf.")

    return embedding


# ============================================================
# Main extraction pipeline
# ============================================================

def extract_all_embeddings(tracks_csv, output_dir):

    tracks_csv = Path(tracks_csv)
    output_dir = Path(output_dir)

    if not tracks_csv.is_file():
        raise FileNotFoundError(
            f"tracks.csv not found: {tracks_csv}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read A's prepared tracks.
    tracks = pd.read_csv(tracks_csv)

    required_columns = {
        "track_id",
        "album_id",
        "audio_path"
    }

    missing_columns = required_columns - set(tracks.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in tracks.csv: {missing_columns}"
        )

    num_tracks = len(tracks)

    if num_tracks == 0:
        raise ValueError("tracks.csv contains no tracks.")

    print(f"Number of tracks: {num_tracks}")
    print(f"Input: {tracks_csv}")
    print(f"Output: {output_dir}\n")

    # Load MERT.
    model, feature_extractor, device, sample_rate = load_mert()

    # Create the output matrix directly on disk.
    embedding_path = output_dir / "track_embeddings.npy"

    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_tracks, EMBEDDING_DIM)
    )

    # Initialize every row with zeros.
    # Failed tracks will keep their zero vectors.
    embeddings[:] = 0.0

    failed_tracks = []
    successful_count = 0

    start_time = time.time()

    # IMPORTANT:
    # Do not sort or reset the DataFrame index.
    # Output row i corresponds to tracks.csv row i.
    for row_index, row in enumerate(
        tracks.itertuples(index=False)
    ):

        track_id = row.track_id
        album_id = row.album_id
        audio_path = row.audio_path

        try:
            real_audio_path = resolve_audio_path(audio_path)

            embedding = extract_embedding(
                real_audio_path,
                model,
                feature_extractor,
                device,
                sample_rate
            )

            embeddings[row_index] = embedding
            successful_count += 1

        except Exception as exc:

            # Keep the zero vector for this track.
            failed_tracks.append({
                "row_index": row_index,
                "track_id": str(track_id),
                "album_id": str(album_id),
                "audio_path": str(audio_path),
                "error": str(exc)
            })

        # Periodically flush results to disk.
        if (row_index + 1) % LOG_INTERVAL == 0:
            embeddings.flush()

            elapsed = time.time() - start_time

            print(
                f"Processed {row_index + 1}/{num_tracks} | "
                f"Success: {successful_count} | "
                f"Failed: {len(failed_tracks)} | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

    # Flush and release the memory-mapped array.
    embeddings.flush()
    del embeddings

    elapsed = time.time() - start_time

    # ========================================================
    # Save manifest
    # ========================================================

    manifest = {
        "model_name": MODEL_NAME,
        "input_tracks_csv": str(tracks_csv),
        "output_embedding_file": str(embedding_path),

        "num_tracks": num_tracks,
        "successful_tracks": successful_count,
        "failed_tracks": len(failed_tracks),

        "embedding_shape": [
            num_tracks,
            EMBEDDING_DIM
        ],
        "embedding_dtype": "float32",

        "sample_rate": sample_rate,
        "audio_channels": 1,
        "audio_input": "entire_track",

        "hidden_layer": "last_hidden_state",
        "pooling": "temporal_mean_pooling",
        "l2_normalization": False,

        "row_alignment": (
            "Row i in track_embeddings.npy corresponds "
            "exactly to row i in the input tracks.csv."
        ),

        "failure_handling": (
            "Failed tracks are represented by zero vectors. "
            "Their details are recorded below."
        ),

        "failed_track_details": failed_tracks,

        "extraction_time_seconds": round(elapsed, 2)
    }

    manifest_path = output_dir / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 55)
    print("MERT EMBEDDING EXTRACTION COMPLETED")
    print("=" * 55)

    print(f"Total tracks:       {num_tracks}")
    print(f"Successful tracks:  {successful_count}")
    print(f"Failed tracks:      {len(failed_tracks)}")

    print(
        f"Embedding shape:   ({num_tracks}, {EMBEDDING_DIM})"
    )

    print(f"Elapsed time:       {elapsed / 60:.2f} minutes")

    print(f"\nSaved embeddings: {embedding_path}")
    print(f"Saved manifest:   {manifest_path}")

    if failed_tracks:
        print("\nFailed track IDs:")
        print([
            item["track_id"]
            for item in failed_tracks[:20]
        ])


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract MERT embeddings for FMA tracks."
    )

    parser.add_argument(
        "--tracks-csv",
        type=str,
        default=str(TRACKS_CSV),
        help="Path to A's prepared tracks.csv"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for embeddings and manifest"
    )

    args = parser.parse_args()

    extract_all_embeddings(
        tracks_csv=args.tracks_csv,
        output_dir=args.output_dir
    )
