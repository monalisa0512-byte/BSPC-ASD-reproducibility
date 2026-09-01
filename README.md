# BSPC eye-tracking ASD protocol-sensitivity release

This repository accompanies the manuscript **“Preprocessing and Protocol Sensitivity in Eye-Tracking-Based ASD Classification: A Multi-Seed Participant-Level Evaluation.”** The study is a methodological and reproducibility analysis of how preprocessing, model choice, training stochasticity, and the window-selection protocol affect participant-level classification estimates.

Repository: <https://github.com/monalisa0512-byte/BSPC-ASD-reproducibility>.

## Authors and affiliations

1. **Zizheng Liu**<br>
   School of Software, Yunnan University, Kunming, China<br>
   E-mail: <liumiao0512@gmail.com><br>
   ORCID: <https://orcid.org/0009-0000-9152-0228>

2. **Huiyong Li**<br>
   Research Institute for Information Technology, Kyushu University, Japan<br>
   E-mail: <li.huiyong.194@m.kyushu-u.ac.jp><br>
   ORCID: <https://orcid.org/0000-0001-9916-7908>

3. **Na Zhao**<br>
   School of Software, Yunnan University, Kunming, China<br>
   E-mail: <zhaonayx@126.com><br>
   ORCID: <https://orcid.org/0000-0002-6166-2118>

4. **Wudao Yang** (corresponding author)<br>
   CS Department, School of Mathematics and Computer Science, Yunnan Minzu University, No. 2929 Yuehua Street, Chenggong District, Kunming 650504, China<br>
   Faculty of Computer Science & Information Technology, Universiti Malaya, Kuala Lumpur 50603, Malaysia<br>
   E-mail: <wudaoyang@ymu.edu.cn><br>
   ORCID: <https://orcid.org/0000-0001-8411-1450>

## Citation requirement

Use of this repository, including its source code, processed outputs, result tables, figures, or substantial portions thereof, must be accompanied by an appropriate citation to the following paper. This requirement also applies to derivative analyses, adaptations, benchmarks, and redistributed copies.

Until the final bibliographic record and DOI are available, cite the paper as:

> Z. Liu, H. Li, N. Zhao, and W. Yang, “Preprocessing and protocol sensitivity in eye-tracking-based ASD classification: A multi-seed participant-level evaluation,” *Biomedical Signal Processing and Control*, in press, 2026.

BibTeX:

```bibtex
@article{liu2026_protocol_sensitivity,
  author  = {Liu, Zizheng and Li, Huiyong and Zhao, Na and Yang, Wudao},
  title   = {Preprocessing and Protocol Sensitivity in Eye-Tracking-Based ASD Classification: A Multi-Seed Participant-Level Evaluation},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {In press}
}
```

After publication, replace this provisional citation with the journal's final citation and DOI. When citing the software release separately, also include the repository URL and the release version or commit hash used.

## Copyright and permitted use

Copyright © 2026 Zizheng Liu, Huiyong Li, Na Zhao, and Wudao Yang. All rights reserved.

The authors retain copyright in the original source code, documentation, result tables, and repository-generated figures, except for third-party materials that remain subject to their respective owners' terms. The public Cilia et al. eye-tracking dataset is not redistributed by this repository and is governed by the terms of its original Figshare record.

Access to this repository does not waive the authors' copyright. Any use, reproduction, modification, redistribution, public display, or preparation of derivative work must:

1. preserve this copyright notice and the author attribution;
2. cite the accompanying paper using the citation above (or the final published citation when available);
3. clearly identify any modifications made to the original materials; and
4. comply with the applicable terms for all third-party data, software, and other dependencies.

For questions or permissions beyond these terms, contact the corresponding author, Wudao Yang, at <wudaoyang@ymu.edu.cn>. A separate `LICENSE` file should be added to define the legally operative software licence; if it conflicts with this README, the `LICENSE` file controls.

## Evidence lock

The archived results use 57 independent participants (27 ASD and 30 TD), participant-level leave-one-subject-out (LOSO) evaluation, and global seeds 42–46. No best-seed selection is performed.

- **Fixed-window analysis:** 75 independent tasks = 3 models × 5 preprocessing conditions × 5 seeds; 4,275 outer folds; 1000/500-ms window/stride; zero failed folds. This analysis provides a controlled comparison of preprocessing conditions.
- **Nested-window analysis:** 20 independent tasks = three Full Preprocessing model runs plus AttentionNet without filtering, each at five seeds; 1,140 outer folds and 4,560 candidate fits; zero failed folds. Candidate windows are selected using training/validation participants only.
- **Inference:** crossed bootstrap resamples global seeds and participants. Global seeds are repeated computational runs, not additional independent participants.

The principal results are intentionally non-superiority findings. All reported 95% confidence intervals for Full-minus-ablation, pairwise model, and fixed-versus-nested comparisons include zero.

The separate subject-balanced-versus-ordinary-shuffle result is retained only because it was requested as a dedicated reviewer diagnostic. It is a single-seed exploratory analysis that predates the strict batches, its paired tests were not significant, and it is not pooled with the multi-seed inference.

| Protocol / condition | Accuracy, mean ± SD | F1, mean ± SD |
|---|---:|---:|
| Fixed, AttentionNet, Full | 76.49 ± 4.23% | 75.51 ± 3.98% |
| Fixed, CNNLSTM, Full | 75.79 ± 5.46% | 74.77 ± 5.30% |
| Fixed, PureLSTM, Full | 74.39 ± 4.74% | 73.97 ± 5.11% |
| Nested, AttentionNet, Full | 76.84 ± 4.54% | 76.76 ± 4.41% |
| Nested, CNNLSTM, Full | 75.79 ± 5.32% | 75.72 ± 5.42% |
| Nested, PureLSTM, Full | 75.09 ± 6.25% | 75.07 ± 6.30% |
| Nested, AttentionNet, No Filtering | 77.89 ± 5.90% | 77.66 ± 5.48% |

For AttentionNet, Full minus No Filtering was +2.81 percentage points under the fixed protocol (95% CI −5.61 to 10.88; Full higher in 4/5 seeds) and −1.05 percentage points under nested selection (95% CI −9.82 to 7.37; Full higher in 3/5 seeds). The 1000/500-ms candidate was selected in 76–80% of nested runs, but more than 80% of folds had ties on validation Youden J, F1, and accuracy; its frequency therefore partly reflects the prespecified smaller-window tie-break rule and is not evidence that it is uniquely optimal.

## Data access

The raw recordings are third-party public data and are not redistributed. Download:

- Federica Cilia et al. (2022), *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder*.
- DOI: <https://doi.org/10.6084/m9.figshare.20113592>

Place the scene CSV files and `Metadata_Participants.csv` in `data/raw/`, then run:

```bash
python data_processing/process_eyesdata.py \
  --raw-dir data/raw \
  --output-dir data/eyesdata_processed_57
```

Alternatively, set `BSPC_DATA_DIR` to an existing verified processed-data directory.

## Environment and audit

The GPU experiments used Ubuntu 22.04, PyTorch 2.8.0+cu128, and one NVIDIA GeForce RTX 4090 D. Install a PyTorch build compatible with the local CUDA runtime and then install the remaining dependencies:

```bash
pip install -r requirements.txt
python smoke_check.py
```

## Strict experiment entry points

One fixed task runs exactly one model, preprocessing condition, and global seed and can resume from completed fold files:

```bash
python experiments/run_fixed_multiseed_fold_strict.py \
  --model AttentionNet \
  --config Full_Preprocessing \
  --global-seed 42 \
  --data-folder data/eyesdata_processed_57 \
  --output-root run_outputs/fixed_multiseed_strict \
  --epochs 50
```

The full fixed batch and nested batch can be launched on Linux with:

```bash
bash experiments/run_strict_reproducibility_gate.sh
bash experiments/launch_fixed_multiseed_strict_75.sh
bash experiments/launch_nested_multiseed_strict.sh
```

The reproducibility gate repeats one complete task twice before formal batch execution. Batch scripts retain fold-level checkpoints and task progress files.

## Seed and leakage controls

The fixed analysis uses:

```text
fold_seed = global_seed * 100000 + one_based_fold_id
```

The nested analysis additionally uses:

```text
candidate_seed = fold_seed * 10 + one_based_candidate_index
```

Within the same model, global seed, and LOSO fold, preprocessing conditions reuse the same participant-level training/validation split and fold seed. The seed controls participant partitioning, weight initialization, weighted sampling, mini-batch order, dropout, and stochastic optimization; this is therefore **pipeline-level stochastic sensitivity**. Preprocessing is applied according to the assigned condition to training, validation, and test data. The held-out test participant is never used for normalization fitting, early stopping, threshold selection, window selection, model selection, or seed selection.

Nested candidates are ranked lexicographically by validation Youden J, F1, and accuracy. If all three metrics tie, the smaller window and then the smaller stride are selected. Each chosen candidate is applied once to the untouched outer test participant.

## Rebuild aggregate tables and the protocol-sensitivity figure

```bash
python experiments/aggregate_fixed_multiseed_strict.py \
  --input-root results/fixed_multiseed_strict \
  --output-dir regenerated/fixed_aggregate

python experiments/aggregate_nested_multiseed_strict.py \
  --input-root results/nested_multiseed_strict \
  --fixed-root results/fixed_multiseed_strict \
  --output-dir regenerated/nested_aggregate

python figures/generate_protocol_sensitivity_figure.py \
  --fixed-aggregate regenerated/fixed_aggregate \
  --nested-aggregate regenerated/nested_aggregate \
  --output-dir regenerated/figure
```

The exploratory sampling-control diagnostic can be rerun separately with `experiments/run_subject_balancing_ablation_nested.py`; it must not be interpreted as part of the strict five-seed evidence lock.

## Result map

| Manuscript evidence | Archived file |
|---|---|
| Fixed per-seed task metrics | `results/fixed_multiseed_strict/aggregate/multiseed_task_metrics.csv` |
| Fixed preprocessing summaries | `results/fixed_multiseed_strict/aggregate/multiseed_metric_summary.csv` |
| Fixed Full-minus-ablation intervals | `results/fixed_multiseed_strict/aggregate/multiseed_paired_differences.csv` |
| Fixed pairwise model intervals | `results/fixed_multiseed_strict/aggregate/multiseed_full_model_paired_differences.csv` |
| Nested per-seed task metrics | `results/nested_multiseed_strict/aggregate/nested_task_metrics.csv` |
| Nested model/filtering intervals | `results/nested_multiseed_strict/aggregate/nested_paired_comparisons.csv` |
| Fixed-versus-nested intervals | `results/nested_multiseed_strict/aggregate/fixed_nested_reference_comparisons.csv` |
| Window frequencies and per-fold ties | `results/nested_multiseed_strict/aggregate/nested_window_selection_frequency.csv`; `nested_selected_window_by_fold.csv` |
| Participant-level predictions and fold metadata | task directories under `results/fixed_multiseed_strict/` and `results/nested_multiseed_strict/` |
| Exploratory single-seed sampling diagnostic | `results/subject_balancing_ablation_nested/subject_balancing_ablation_nested_summary.csv` |
| Figure source data and exports | `figures/source_data/`; `figures/protocol_sensitivity.*` |

Archived `task_contract.json` files retain original execution-host paths and source hashes as provenance metadata. The release scripts themselves use repository-relative defaults or explicit command-line paths.

## Repository layout

```text
data_processing/  raw-to-processed data preparation
experiments/      strict runners, launchers, progress tools, and aggregators
figures/          protocol-sensitivity figure script, exports, and source data
results/          archived fixed and nested task records and aggregate tables
data/             data-access instructions; raw data are not redistributed
```

No Zenodo DOI is assigned at the review stage. A public tagged release, a legally operative `LICENSE` file, and an archival DOI should be added to the final public release.
