# Data directory

Raw eye-tracking data are not redistributed in this repository. Obtain the public dataset from Figshare at <https://doi.org/10.6084/m9.figshare.20113592>.

Expected layout after download and preprocessing:

```text
data/
├── raw/
│   ├── Metadata_Participants.csv
│   └── [scene CSV files]
└── eyesdata_processed_57/
    ├── labeled_*.csv
    └── processing_summary.csv
```

The processing script excludes records that cannot be linked to verified numeric participants and retains PIDs 1–59 except 12 and 16, which are absent from the raw recordings.

