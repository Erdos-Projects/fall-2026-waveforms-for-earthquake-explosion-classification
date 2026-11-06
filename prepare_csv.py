"""Run with: python data_processing/prepare_csv.py"""
from pathlib import Path
import pandas as pd

from clean_data import (
    FEATURES, clean_labels, clean_context, select_recordings,
    aggregate_features, make_event_table,
)


def main():
    # Locate data/ beside data_processing/, regardless of the terminal's folder.
    script_folder = Path(__file__).resolve().parent
    project_folder = script_folder.parent
    data_folder = project_folder / "data"
    raw = pd.read_csv(data_folder / "comcat_metadata.csv.gz", dtype={
        "event_id": "string", "station_network_code": "string", "station_code": "string",
    })
    labeled = clean_labels(raw)
    candidates = clean_context(labeled)

    waveform_results = pd.read_csv(
        data_folder / "waveform_measurements.csv.gz", usecols=["source_row", "status"]
    )
    saved_groups = pd.read_csv(data_folder / "event_groups_and_folds.csv.gz")
    recordings = select_recordings(candidates, waveform_results, saved_groups)
    measurements = pd.read_csv(
        data_folder / "recording_features.csv.gz",
        usecols=["source_row", "event_id", "station_id"] + FEATURES,
        float_precision="round_trip",  # Read the saved measurements at their stored precision.
    )
    medians, station_counts = aggregate_features(recordings, measurements)

    # Read the checkpoint as text to retain its saved context and two baseline features.
    checkpoint = pd.read_csv(data_folder / "event_checkpoint.csv.gz", dtype=str, keep_default_na=False)
    cleaned = make_event_table(checkpoint, recordings, medians, station_counts, saved_groups)
    exclusions = pd.read_csv(data_folder / "excluded_events.csv.gz")
    event_is_excluded = cleaned["event_id"].isin(exclusions["event_id"])
    cleaned = cleaned.loc[~event_is_excluded]
    assert cleaned["event_id"].is_unique

    output_folder = project_folder / "deliverables"
    output_folder.mkdir(exist_ok=True)
    output = output_folder / "pnw_training_corrected.csv"
    cleaned.to_csv(output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
