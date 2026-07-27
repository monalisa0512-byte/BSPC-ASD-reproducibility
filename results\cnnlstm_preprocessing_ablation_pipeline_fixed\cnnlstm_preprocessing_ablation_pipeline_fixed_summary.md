# CNNLSTM pipeline-level preprocessing ablation

Date: 2026-06-30 02:14:35

Protocol: fixed 1000/500 ms; training, validation, and test data are all processed with the same preprocessing configuration.

| Setting | Accuracy | 95% CI | F1 | 95% CI |
|---|---:|---:|---:|---:|
| Full_Preprocessing | 78.95% | [68.42%, 87.76%] | 77.78% | [63.41%, 88.46%] |
| Linear_Interpolation | 78.95% | [68.42%, 89.47%] | 78.57% | [64.15%, 89.29%] |
| Without_Blink_Expansion | 75.44% | [64.91%, 85.96%] | 75.00% | [61.53%, 86.57%] |
| No_Filtering | 75.44% | [63.16%, 85.96%] | 76.67% | [61.81%, 87.10%] |

This is the pipeline-level counterpart to the previous locked-test ablation. It is intended to check whether CNNLSTM changes when the same preprocessing ablation is applied consistently to train/validation/test data.
