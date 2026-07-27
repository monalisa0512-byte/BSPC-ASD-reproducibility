# Preprocessing Ablation Paired Statistical Tests

Exact McNemar is computed on paired participant-level correctness. Paired bootstrap reports Full-minus-ablation effect-size intervals for Accuracy and F1. With B=5,000 resamples, an empirical two-sided p value at the finite-resolution bound is shown as <0.0004 rather than p=0.

| Model | Comparison | n | Full Acc | Ablation Acc | Delta Acc | Full F1 | Ablation F1 | Delta F1 | b | c | McNemar p | Delta Acc 95% CI | Bootstrap Acc p | Delta F1 95% CI | Bootstrap F1 p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AttentionNet | Full_Preprocessing vs Linear_Interpolation | 57 | 85.96% | 80.70% | 5.26% | 85.19% | 78.43% | 6.75% | 5 | 2 | 0.4531 | [-3.51%, 14.04%] | 0.3420 | [-2.52%, 17.31%] | 0.1752 |
| AttentionNet | Full_Preprocessing vs Without_Blink_Expansion | 57 | 85.96% | 80.70% | 5.26% | 85.19% | 80.70% | 4.48% | 5 | 2 | 0.4531 | [-3.51%, 14.04%] | 0.3240 | [-4.89%, 14.87%] | 0.3676 |
| AttentionNet | Full_Preprocessing vs No_Filtering | 57 | 85.96% | 63.16% | 22.81% | 85.19% | 67.69% | 17.49% | 15 | 2 | 0.0023 | [10.53%, 35.09%] | <0.0004 | [5.67%, 30.65%] | 0.0040 |
| AttentionNet | Full_Preprocessing vs Without_Mask_Features | 57 | 85.96% | 80.70% | 5.26% | 85.19% | 79.25% | 5.94% | 3 | 0 | 0.2500 | [0.00%, 12.28%] | 0.0880 | [0.00%, 13.97%] | 0.0880 |
| CNNLSTM | Full_Preprocessing vs Linear_Interpolation | 57 | 78.95% | 78.95% | 0.00% | 77.78% | 78.57% | -0.79% | 4 | 4 | 1.0000 | [-8.77%, 10.53%] | 1.0000 | [-11.01%, 9.05%] | 0.9196 |
| CNNLSTM | Full_Preprocessing vs Without_Blink_Expansion | 57 | 78.95% | 75.44% | 3.51% | 77.78% | 75.00% | 2.78% | 6 | 4 | 0.7539 | [-7.02%, 14.04%] | 0.6460 | [-8.57%, 14.29%] | 0.6600 |
| CNNLSTM | Full_Preprocessing vs No_Filtering | 57 | 78.95% | 75.44% | 3.51% | 77.78% | 76.67% | 1.11% | 8 | 6 | 0.7905 | [-8.77%, 17.54%] | 0.6892 | [-11.30%, 13.68%] | 0.8692 |
| CNNLSTM | Full_Preprocessing vs Without_Mask_Features | 57 | 78.95% | 78.95% | 0.00% | 77.78% | 76.92% | 0.85% | 3 | 3 | 1.0000 | [-8.77%, 8.77%] | 1.0000 | [-8.81%, 10.71%] | 0.9064 |
| PureLSTM | Full_Preprocessing vs Linear_Interpolation | 57 | 77.19% | 78.95% | -1.75% | 76.36% | 77.78% | -1.41% | 4 | 5 | 1.0000 | [-12.28%, 8.77%] | 0.8852 | [-12.33%, 9.26%] | 0.8408 |
| PureLSTM | Full_Preprocessing vs Without_Blink_Expansion | 57 | 77.19% | 78.95% | -1.75% | 76.36% | 77.78% | -1.41% | 4 | 5 | 1.0000 | [-12.28%, 8.77%] | 0.8808 | [-12.02%, 9.45%] | 0.8520 |
| PureLSTM | Full_Preprocessing vs No_Filtering | 57 | 77.19% | 71.93% | 5.26% | 76.36% | 73.33% | 3.03% | 8 | 5 | 0.5811 | [-7.02%, 17.54%] | 0.4956 | [-8.32%, 14.27%] | 0.6172 |
| PureLSTM | Full_Preprocessing vs Without_Mask_Features | 57 | 77.19% | 78.95% | -1.75% | 76.36% | 76.92% | -0.56% | 4 | 5 | 1.0000 | [-12.28%, 8.77%] | 0.8812 | [-11.94%, 11.20%] | 0.9608 |

Here b is the number of subjects correctly classified by Full preprocessing and incorrectly classified by the ablation; c is the opposite discordant-pair count.
