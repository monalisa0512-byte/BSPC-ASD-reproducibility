# BSPC eye-tracking ASD reproducibility release

This repository accompanies the manuscript **“Physiology-Aware Preprocessing for Eye-Tracking-Based ASD Classification: A Diagnostic Study of Window Selection and Subject-Level Evaluation.”** It separates fixed-window reference analyses, pipeline-level seed sensitivity, nested window selection, controlled architecture comparisons, preprocessing ablations, participant-balanced sampling, paired statistics, and figure generation.

Repository: <https://github.com/monalisa0512-byte/BSPC-ASD-reproducibility> (private review repository; access is granted separately to invited reviewers).

## Data access

The raw eye-tracking recordings are third-party public data and are not duplicated here. Download the Cilia et al. dataset from Figshare:

- Federica Cilia et al. (2022), *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder*.
- DOI: <https://doi.org/10.6084/m9.figshare.20113592>

Place the downloaded CSV files and `Metadata_Participants.csv` in `data/raw/`, then build the verified 57-participant dataset:

```bash
python data_processing/process_eyesdata.py \
  --raw-dir data/raw \
  --output-dir data/eyesdata_processed_57
```

Alternatively, point all experiment scripts to an existing processed dataset:

```bash
export BSPC_DATA_DIR=/absolute/path/to/eyesdata_processed_57
```

On Windows PowerShell:

```powershell
$env:BSPC_DATA_DIR = 'D:\path\to\eyesdata_processed_57'
```

## Environment

Python 3.9 was used for the checked plotting environment. The five-seed GPU run used PyTorch 2.8.0+cu128 on Ubuntu 22.04 with an NVIDIA GeForce RTX 4090 D. Install the matching CUDA-enabled PyTorch wheel for the target system, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Run the compact release audit before launching long GPU experiments:

```bash
python smoke_check.py
```

## Experiment entry points

### Fixed-window and five-seed AttentionNet

```bash
python experiments/run_fixed_window_seed.py \
  --seed 42 \
  --data-folder data/eyesdata_processed_57 \
  --window-size 1000 \
  --stride 500 \
  --models AttentionNet
```

Run seeds 42–46 with `experiments/run_seed_sensitivity_server.sh`. These repetitions hold the outer LOSO participants, preprocessing specification, fixed window/stride, architecture, and training budget constant. The run seed jointly controls the participant-level training/validation split, model initialization, weighted sampling, mini-batch sequence, dropout, and stochastic optimization. Each run repeats early stopping and validation-only Youden’s J threshold selection. The experiment is therefore a **pipeline-level stochastic sensitivity analysis**, not an initialization-only test.

### Nested window selection

`experiments/run_attentionnet_loso.py` and `experiments/run_model_comparison.py` enable all four candidate windows by default. In each outer fold, the test participant is isolated before candidate training and validation. The selected validation candidate is applied once to the outer test participant.

```bash
python experiments/run_attentionnet_loso.py
python experiments/run_model_comparison.py
```

`experiments/run_attentionnet_loso_nested_seed_control.py` is the candidate-level random-state control used to diagnose the fixed-to-nested instability.

### Preprocessing and sampling ablations

```bash
python experiments/run_attentionnet_preprocessing_ablation_pipeline_fixed.py
python experiments/run_cnnlstm_preprocessing_ablation_pipeline_fixed.py
python experiments/run_purelstm_preprocessing_ablation_pipeline_fixed.py
python experiments/run_subject_balancing_ablation_nested.py
```

### Paired statistics

```bash
python experiments/compute_statistical_significance.py
python experiments/compute_preprocessing_ablation_paired_tests.py
```

## Manuscript result map

| Manuscript result | Script / analysis | Archived verification output |
|---|---|---|
| Fixed-window AttentionNet reference, 85.96% accuracy | `run_fixed_window_seed.py` with one 1000/500-ms candidate | `results/attention/fold_level_metrics.csv` |
| Five-seed mean, 78.60 ± 4.87% | `run_seed_sensitivity_server.sh` | `results/seed_sensitivity_fixed_1000_500_server_v2/seed_summary_all.csv`; `seed_summary_stats.csv` |
| Nested AttentionNet, 77.19% | `run_attentionnet_loso.py` | `results/attention_nested_window/fold_level_metrics.csv` |
| Nested CNNLSTM, 82.46% | `run_model_comparison.py` | `results/model_comparison_nested_window/cnnlstm_subject_predictions.csv` |
| Table 2, three-model preprocessing ablation | three fixed preprocessing-ablation scripts | `results/*preprocessing_ablation_pipeline_fixed/*.csv` |
| Table 3 and Supplementary paired ablations | `compute_preprocessing_ablation_paired_tests.py` | `results/preprocessing_ablation_paired_tests/preprocessing_ablation_paired_tests.csv` |
| Table 4, fixed architecture comparison | fixed AttentionNet predictions + `run_model_comparison.py` with one candidate | `results/attention/fold_level_metrics.csv`; `results/model_comparison/cnnlstm_subject_predictions.csv` |
| Table 5, fixed paired model test | `compute_statistical_significance.py` | `results/statistical_tests/paired_significance_tests.csv` |
| Table 6, seed sensitivity | `run_seed_sensitivity_server.sh` | five-seed summary files above |
| Table 7, non-nested window comparison | fixed single-candidate runs | `results/window_comparison.csv` |
| Table 8, nested window sensitivity | nested AttentionNet and CNNLSTM scripts | `results/attention_nested_window/fold_level_metrics.csv`; `results/model_comparison_nested_window/cnnlstm_subject_predictions.csv` |
| Table 9, participant balancing | `run_subject_balancing_ablation_nested.py` | `results/subject_balancing_ablation_nested/subject_balancing_ablation_nested_summary.csv` |
| Figure 11, LOSO correctness heatmap | `figures/generate_evaluation_figures.py` | `results/attention/fold_level_metrics.csv` |

The archived fixed reference prediction file is the authoritative participant-level record for the 85.96% result. Seed 46 in the reported five-seed server run reaches the same aggregate accuracy but is retained as part of the separate stochastic-sensitivity experiment.

## Test-participant isolation

For every reported outer LOSO fold, the held-out test participant is excluded from normalization fitting, training, early stopping, validation-threshold selection, window selection, hyperparameter selection, seed selection, and checkpoint selection. Candidate-level test-oracle calculations in the diagnostic scripts are explicitly post hoc diagnostics and are not used to produce a reported performance estimate.

## Repository layout

```text
data_processing/   raw-to-processed data preparation
experiments/       training, ablation, nested-selection and statistical scripts
figures/           figure-generation scripts
results/           compact archived predictions and result tables
data/              data-access instructions; raw data are not redistributed
```

## Release and citation

Before public release, the authors must choose a software licence, create a tagged GitHub release, and archive that release in Zenodo to obtain a DOI. Replace the repository placeholder in the manuscript only after the DOI or stable repository URL resolves publicly or through a tested reviewer link.
