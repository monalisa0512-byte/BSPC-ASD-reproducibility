# CNNLSTM fixed-window preprocessing ablation and model-ranking summary

Date: 2026-06-29

## Protocol

- Model: CNNLSTM
- Window/stride: 1000/500 ms
- Evaluation: LOSO, participant-level median aggregation
- Training setup: batch size 32, learning rate 0.001, 50 epochs
- Test data handling: locked to the full preprocessing standard, matching the existing AttentionNet ablation protocol

## CNNLSTM preprocessing ablation

| Preprocessing setting | Accuracy | 95% CI | F1 | 95% CI |
|---|---:|---:|---:|---:|
| Full preprocessing | 78.95% | [68.42%, 87.76%] | 77.78% | [63.41%, 88.46%] |
| Linear interpolation | 78.95% | [68.42%, 89.47%] | 78.57% | [64.15%, 89.29%] |
| Without blink expansion | 78.95% | [68.42%, 89.47%] | 77.78% | [63.83%, 88.89%] |
| No filtering | 78.95% | [68.42%, 87.72%] | 78.57% | [65.30%, 88.53%] |

## Preprocessing x model ranking

| Preprocessing setting | AttentionNet accuracy | CNNLSTM accuracy | Best point estimate | Accuracy gap |
|---|---:|---:|---|---:|
| Full preprocessing | 85.96% | 78.95% | AttentionNet | AttentionNet +7.01 |
| Linear interpolation | 71.93% | 78.95% | CNNLSTM | CNNLSTM +7.02 |
| Without blink expansion | 73.68% | 78.95% | CNNLSTM | CNNLSTM +5.27 |
| No filtering | 71.93% | 78.95% | CNNLSTM | CNNLSTM +7.02 |

## Interpretation

Under the fixed 1000/500 ms setting, CNNLSTM showed nearly unchanged subject-level accuracy across the four preprocessing variants, whereas AttentionNet showed large drops under linear interpolation, no blink expansion, and no filtering. Therefore, the clearest conclusion is not that preprocessing improves every model equally. Instead, the result supports a more precise statement: preprocessing can affect both the achievable performance level and the apparent model ranking, with AttentionNet being more sensitive to these preprocessing choices in this dataset.

This result is consistent with the revised manuscript framing: AttentionNet should not be presented as a universally superior architecture. Its fixed-window advantage depends on the full physiology-aware preprocessing pipeline and does not remain stable under all preprocessing or nested-selection conditions.
