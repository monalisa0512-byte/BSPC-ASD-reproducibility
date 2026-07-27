"""
Process raw eye-tracking data into a clean 57-subject dataset.

Rows marked as Unidentified(Pos) or Unidentified(Neg) are excluded rather than
mapped to pseudo-participant IDs. The final cohort keeps only verified numeric
participants present in the raw recordings: PIDs 1-59 excluding 12 and 16.
"""
import pandas as pd
from pathlib import Path
import argparse


VALID_PIDS = set(range(1, 60)) - {12, 16}


def process_eyesdata(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = raw_dir / "Metadata_Participants.csv"
    meta = pd.read_csv(meta_path)
    # Build lookup dict: PID -> row
    meta_dict = {}
    for _, row in meta.iterrows():
        meta_dict[int(row["ParticipantID"])] = row

    # Target column structure used by the modeling scripts
    target_cols = [
        "Unnamed: 0",
        "RecordingTime [ms]",
        "Time of Day [h:m:s:ms]",
        "Trial",
        "Stimulus",
        "Export Start Trial Time [ms]",
        "Export End Trial Time [ms]",
        "Participant",
        "Color",
        "Tracking Ratio [%]",
        "Category Group",
        "Category Right",
        "Category Left",
        "Index Right",
        "Index Left",
        "Pupil Diameter Right [mm]",
        "Pupil Diameter Left [mm]",
        "Point of Regard Right X [px]",
        "Point of Regard Right Y [px]",
        "Point of Regard Left X [px]",
        "Point of Regard Left Y [px]",
        "AOI Name Right",
        "AOI Name Left",
        "Gaze Vector Right X",
        "Gaze Vector Right Y",
        "Gaze Vector Right Z",
        "Gaze Vector Left X",
        "Gaze Vector Left Y",
        "Gaze Vector Left Z",
        "Annotation Name",
        "Annotation Description",
        "Annotation Tags",
        "Mouse Position X [px]",
        "Mouse Position Y [px]",
        "Scroll Direction X",
        "Scroll Direction Y",
        "Content",
        "ParticipantID",
        "Gender",
        "Age",
        "Class",
        "CARS Score",
    ]

    # Get raw CSV files (scene 1-25)
    csv_files = sorted(
        [f for f in raw_dir.iterdir() if f.suffix == ".csv" and f.name != "Metadata_Participants.csv"],
        key=lambda x: int(x.stem),
    )

    print(f"Found {len(csv_files)} raw CSV files")
    all_pids = set()
    summary_records = []

    for csv_file in csv_files:
        scene_num = int(csv_file.stem)
        print(f"\nProcessing scene {scene_num}...")

        # Read raw data
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"  Original shape: {df.shape}, columns: {len(df.columns)}")
        original_rows = len(df)

        # --- Step 1: Standardize columns to match target structure ---
        # Add missing columns with NaN
        for col in target_cols:
            if col not in df.columns and col not in ("ParticipantID", "Gender", "Age", "Class", "CARS Score"):
                df[col] = pd.NA

        # --- Step 2: Keep only verified numeric participant IDs ---
        participant_raw = df["Participant"].astype(str).str.strip()
        unidentified_mask = participant_raw.isin(["Unidentified(Pos)", "Unidentified(Neg)"])
        unidentified_rows = int(unidentified_mask.sum())

        participant_numeric = pd.to_numeric(participant_raw, errors="coerce")
        non_numeric_rows = int(participant_numeric.isna().sum())
        df["Participant"] = participant_numeric.astype("Int64")

        invalid_pid_mask = ~df["Participant"].isin(VALID_PIDS)
        invalid_pid_rows = int((invalid_pid_mask & ~participant_numeric.isna()).sum())
        keep_mask = df["Participant"].isin(VALID_PIDS)
        df = df.loc[keep_mask].copy()

        # --- Step 3: Add label columns ---
        def get_meta(pid):
            if pd.isna(pid):
                return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
            pid = int(pid)
            if pid in meta_dict:
                row = meta_dict[pid]
                return pd.Series([pid, row["Gender"], row["Age"], row["Class"], row["CARS Score"]])
            return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA])

        labels = df["Participant"].apply(get_meta)
        labels.columns = ["ParticipantID", "Gender", "Age", "Class", "CARS Score"]

        df = pd.concat([df, labels], axis=1)

        # --- Step 4: Reorder to target columns ---
        # Only keep columns that exist in target_cols
        existing_cols = [c for c in target_cols if c in df.columns]
        df = df[existing_cols]

        # Fill any remaining missing target columns
        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[target_cols]

        # Reset index to create Unnamed: 0
        df = df.reset_index(drop=True)
        df["Unnamed: 0"] = df.index

        # Collect PIDs in this file
        pids = set(df["ParticipantID"].dropna().astype(int).unique())
        all_pids.update(pids)
        print(f"  PIDs in this file: {sorted(pids)}")
        print(f"  Output shape: {df.shape}")
        print(
            f"  Removed rows: unidentified={unidentified_rows}, "
            f"non_numeric={non_numeric_rows}, invalid_pid={invalid_pid_rows}, "
            f"total_removed={original_rows - len(df)}"
        )
        summary_records.append({
            "scene": scene_num,
            "input_file": csv_file.name,
            "input_rows": original_rows,
            "output_rows": len(df),
            "removed_rows": original_rows - len(df),
            "unidentified_rows": unidentified_rows,
            "non_numeric_rows": non_numeric_rows,
            "invalid_pid_rows": invalid_pid_rows,
            "pid_count": len(pids),
            "pids": " ".join(str(pid) for pid in sorted(pids)),
        })

        # Save
        output_file = output_dir / f"labeled_{scene_num}.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

    print(f"\n{'='*60}")
    print(f"All unique PIDs: {sorted(all_pids)} ({len(all_pids)} people)")
    print(f"ASD count: {len([p for p in all_pids if p <= 29])}")
    print(f"TD count: {len([p for p in all_pids if p >= 30])}")
    print(f"Missing expected PIDs: {sorted(VALID_PIDS - all_pids)}")
    print(f"Unexpected PIDs: {sorted(all_pids - VALID_PIDS)}")

    summary_df = pd.DataFrame(summary_records).sort_values("scene")
    summary_path = output_dir / "processing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Processing summary saved to {summary_path}")


if __name__ == "__main__":
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the verified 57-participant processed dataset.")
    parser.add_argument("--raw-dir", type=Path, default=package_root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=package_root / "data" / "eyesdata_processed_57")
    args = parser.parse_args()
    process_eyesdata(args.raw_dir, args.output_dir)
