"""Reconstruct the geographic groups and folds for the original reviewed cohort.

This is an optional explanation of the saved assignments. prepare_csv.py keeps
reading the supplied folds and does not call these functions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def distance_km(first, second):
    """Great-circle distance between two event rows with latitude and longitude."""
    latitude_1, latitude_2 = np.deg2rad([
        first["source_latitude_deg"], second["source_latitude_deg"],
    ])
    longitude_difference = np.deg2rad(
        second["source_longitude_deg"] - first["source_longitude_deg"]
    )
    latitude_term = np.sin((latitude_2 - latitude_1) / 2) ** 2
    longitude_term = (
        np.cos(latitude_1) * np.cos(latitude_2)
        * np.sin(longitude_difference / 2) ** 2
    )
    # Clip roundoff so sqrt and arcsin receive values in their valid range.
    haversine = np.clip(latitude_term + longitude_term, 0, 1)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(haversine))


def geographic_groups(events):
    """Return grouped event rows and the event pairs that connect their cells.

    events has one row per reviewed event, with event_id, label,
    source_origin_time, source_latitude_deg, and source_longitude_deg.
    The returned events are sorted by event_id with a fresh row index.
    """
    events = events.sort_values("event_id").reset_index(drop=True).copy()
    assert events["event_id"].is_unique

    # Approximate 20 km cells. Longitude is scaled at latitude 47 degrees.
    # floor rounds down (-0.2 becomes -1), keeping negatives in the correct cell.
    longitude_cell_index = np.floor(
        events["source_longitude_deg"] * 111.195 * np.cos(np.deg2rad(47)) / 20
    ).astype(int)
    latitude_cell_index = np.floor(
        events["source_latitude_deg"] * 111.195 / 20
    ).astype(int)
    events["source_cell"] = (
        longitude_cell_index.astype(str) + ":" + latitude_cell_index.astype(str)
    )

    # Each cell begins in its own group. Connected cells will share a name.
    group_for_cell = {}
    for cell in events["source_cell"].unique():
        group_for_cell[cell] = cell

    events_by_time = events.copy()
    events_by_time["origin"] = pd.to_datetime(
        events_by_time["source_origin_time"], utc=True, format="mixed"
    )
    events_by_time = events_by_time.sort_values("origin")
    rows = events_by_time.to_dict("records")
    pairs = []
    # A pair within 30 seconds and 25 km connects the cells containing its events.
    for first_index, first in enumerate(rows):
        second_index = first_index + 1
        while second_index < len(rows):
            second = rows[second_index]
            seconds = (second["origin"] - first["origin"]).total_seconds()
            if seconds > 30:
                # Later rows are farther away in time, so stop this search.
                break
            distance = distance_km(first, second)
            if distance <= 25:
                pairs.append({
                    "event_a": first["event_id"], "event_b": second["event_id"],
                    "seconds": seconds, "km": float(distance),
                })
                first_group = group_for_cell[first["source_cell"]]
                second_group = group_for_cell[second["source_cell"]]
                # Compare names as strings, not as numeric cell coordinates.
                smaller_name = min(first_group, second_group)
                larger_name = max(first_group, second_group)
                # Merge whole groups: A linked to B and B to C keeps all three together.
                for cell, group in group_for_cell.items():
                    if group == larger_name:
                        group_for_cell[cell] = smaller_name
            second_index += 1

    events["spatial_group"] = events["source_cell"].map(group_for_cell)
    pairs = pd.DataFrame(pairs, columns=["event_a", "event_b", "seconds", "km"])
    return events, pairs


def geographic_folds(events):
    """Assign five folds while keeping every geographic group together.

    Return a copy sorted by event_id, with a fresh row index and fold numbers 1–5.
    """
    events = events.sort_values("event_id").reset_index(drop=True).copy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261022)
    splits = splitter.split(events, events["label"], groups=events["spatial_group"])
    events["geographic_fold"] = 0
    for fold, split_rows in enumerate(splits, start=1):
        training_rows, held_out_rows = split_rows
        # Only the held-out rows receive this fold number; the rest are training rows.
        # These positions refer to the sorted events above.
        # reset_index makes the .loc row labels match those positions.
        events.loc[held_out_rows, "geographic_fold"] = fold
    return events
