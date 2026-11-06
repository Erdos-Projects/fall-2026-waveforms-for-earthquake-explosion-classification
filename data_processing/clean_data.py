"""Clean station metadata, select recordings, and calculate event features."""
import numpy as np
import pandas as pd

PNSN_LABELS = {"eq": 0, "px": 1, "ex": 1, "qb": 1}
GENERIC_LABELS = {"earthquake": 0, "explosion": 1}
CONTEXT_COLUMNS = [
    "source_latitude_deg", "source_longitude_deg",
    "preferred_source_magnitude", "source_depth_km",
]
FEATURE_NAMES = {
    "wf_p_s_peak_rms_ratio": "nb_log10_pz_sh_rms_ratio_4_8hz",
    "wf_s_horizontal_vertical_rms_ratio": "nb_log10_sh_sz_rms_ratio_4_8hz",
    "wf_p_rise_10_90_s": "nb_envelope_rise_10_90_s_2_10hz",
    "wf_excess_kurtosis": "nb_envelope_excess_kurtosis_2_10hz",
    "wf_low_high_power_ratio": "nb_log10_psd_power_1_5_over_5_15hz",
}
FEATURES = list(FEATURE_NAMES.values())


def clean_labels(raw):
    """Return station rows with complete identifiers and consistent event labels."""
    data = raw.copy()
    # Saved measurements use the original zero-based data row number as a key.
    data["source_row"] = np.arange(len(data))

    # Treat missing codes as empty text, then ignore case and surrounding spaces.
    pnsn_codes = data["source_type_pnsn_label"].fillna("")
    pnsn_codes = pnsn_codes.str.lower().str.strip()
    generic_codes = data["source_type"].fillna("")
    generic_codes = generic_codes.str.lower().str.strip()
    pnsn_labels = pnsn_codes.map(PNSN_LABELS)
    generic_labels = generic_codes.map(GENERIC_LABELS)

    # map() leaves unknown codes missing. Only an empty PNSN code permits fallback.
    data["label"] = pnsn_labels
    pnsn_is_empty = pnsn_codes == ""
    data.loc[pnsn_is_empty, "label"] = generic_labels.loc[pnsn_is_empty]
    label_is_missing = data["label"].isna()
    both_labels_known = pnsn_labels.notna() & generic_labels.notna()
    labels_disagree = both_labels_known & (pnsn_labels != generic_labels)

    # Collect event IDs first so one bad label excludes all rows for that event.
    bad_label = label_is_missing | labels_disagree
    excluded_event_ids = set(data.loc[bad_label, "event_id"])
    labels_per_event = data.groupby("event_id")["label"].nunique()
    conflicting_events = labels_per_event[labels_per_event > 1]
    excluded_event_ids.update(conflicting_events.index)

    identity_columns = ["event_id", "station_network_code", "station_code", "trace_name"]
    # axis=1 checks across columns: every identifier in a row must be present.
    has_identity = data[identity_columns].notna().all(axis=1)
    for column in identity_columns:
        text = data[column].astype("string").str.strip()
        has_text = text.ne("").fillna(False)
        has_identity = has_identity & has_text

    event_is_excluded = data["event_id"].isin(excluded_event_ids)
    # ~ reverses True/False. loc[keep] selects rows whose keep value is True.
    keep = has_identity & ~event_is_excluded
    data = data.loc[keep].copy()
    data["label"] = data["label"].astype(int)
    data["station_id"] = data["station_network_code"] + "." + data["station_code"]
    return data


def clean_context(data):
    """Check required event context, then remove publisher-flagged recordings."""
    data = data.copy()
    # coerce turns unreadable numbers or dates into missing values.
    for column in CONTEXT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    origin_times = pd.to_datetime(
        data["source_origin_time"], utc=True, errors="coerce", format="mixed"
    )
    finite_context = np.isfinite(data[CONTEXT_COLUMNS]).all(axis=1)
    valid_latitude = data["source_latitude_deg"].between(-90, 90)
    valid_longitude = data["source_longitude_deg"].between(-180, 180)
    valid_context = finite_context & origin_times.notna() & valid_latitude & valid_longitude
    excluded_event_ids = set(data.loc[~valid_context, "event_id"])

    # nunique counts distinct values within each event. Required values must agree.
    # Keep original time strings for this comparison, even if two strings parse alike.
    for column in CONTEXT_COLUMNS + ["source_origin_time"]:
        values_per_event = data.groupby("event_id")[column].nunique(dropna=False)
        inconsistent_events = values_per_event[values_per_event > 1]
        excluded_event_ids.update(inconsistent_events.index)
    event_is_excluded = data["event_id"].isin(excluded_event_ids)
    data = data.loc[~event_is_excluded].copy()

    # Negative depths and missing optional uncertainties remain in the dataset.
    # These publisher flags remove only the affected station row.
    for column in ["trace_missing_channel", "trace_has_offset"]:
        flag_values = pd.to_numeric(data[column], errors="coerce")
        recording_is_flagged = flag_values > 0
        data = data.loc[~recording_is_flagged].copy()
    return data


def select_recordings(candidates, waveform_results, saved_groups):
    """Keep QC passes, one stream per station, and the saved reviewed event list."""
    outcomes = waveform_results[["source_row", "status"]]
    # merge matches row IDs; one_to_one requires a unique ID on each side.
    recordings = candidates.merge(outcomes, on="source_row", validate="one_to_one")
    passed_qc = recordings["status"] == "pass"
    recordings = recordings.loc[passed_qc].copy()

    # Keep the first trace name in sorted order for each event/station pair.
    # Select after QC so a failed stream cannot hide a valid one at its station.
    recordings = recordings.sort_values(["event_id", "station_id", "trace_name"], kind="stable")
    recordings = recordings.drop_duplicates(["event_id", "station_id"])
    reviewed_event_ids = saved_groups["event_id"]
    event_was_retained = recordings["event_id"].isin(reviewed_event_ids)
    return recordings.loc[event_was_retained].copy()


def aggregate_features(recordings, measurements):
    """Return two event-indexed tables: feature medians and station counts."""
    keys = ["source_row", "event_id", "station_id"]
    selected_keys = recordings[keys]
    measurements = selected_keys.merge(measurements, on=keys, validate="one_to_one")
    assert len(measurements) == len(recordings), "Some selected recordings have no saved measurements."
    # pandas skips missing values in both median() and count(). Zero stays valid.
    measurements[FEATURES] = measurements[FEATURES].replace([np.inf, -np.inf], np.nan)
    station_values_by_event = measurements.groupby("event_id")[FEATURES]
    medians = station_values_by_event.median()
    station_counts = station_values_by_event.count()
    return medians, station_counts


def make_event_table(checkpoint, recordings, medians, counts, saved_groups):
    """Combine selected events with saved context, baseline features, and folds.

    medians and counts have event IDs as their index. Context and the two
    baseline features come from checkpoint; changing raw values does not update them.
    """
    has_measurements = checkpoint["event_id"].isin(medians.index)
    events = checkpoint.loc[has_measurements].copy()
    # An event-ID index lets pandas match values by ID when row orders differ.
    events = events.set_index("event_id")
    events = events.rename(columns=FEATURE_NAMES)

    one_row_per_event = recordings.drop_duplicates("event_id")
    labels_by_event = one_row_per_event.set_index("event_id")["label"]
    folds_by_event = saved_groups.set_index("event_id")["geographic_fold"]
    # reindex puts each table in the checkpoint's event order.
    event_labels = labels_by_event.reindex(events.index)
    event_folds = folds_by_event.reindex(events.index)
    events["label"] = event_labels.astype(int).astype(str)
    events["geographic_fold"] = event_folds.astype(str)

    # Retain the checkpoint's saved text. Replace its five old feature columns.
    for feature in FEATURES:
        feature_medians = medians[feature].reindex(events.index)
        feature_text = feature_medians.map(str)
        # Write missing measurements as empty CSV fields.
        feature_text = feature_text.where(feature_medians.notna(), "")
        events[feature] = feature_text
        events[feature + "__n_stations"] = counts[feature].astype(str)
    # Restore event_id as an ordinary CSV column.
    return events.reset_index()
