# Paired Statistical Tests

Exact McNemar is the primary test for paired participant-level correctness. Paired bootstrap reports effect-size intervals for metric differences.

| Comparison | n | Acc A | Acc B | Delta Acc | b | c | McNemar p | Bootstrap Delta Acc 95% CI | Bootstrap p | Delta F1 | Bootstrap Delta F1 95% CI | Bootstrap p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AttentionNet vs CNNLSTM | 57 | 0.860 | 0.789 | 0.070 | 6 | 2 | 0.2891 | [-0.018, 0.175] | 0.2020 | 0.074 | [-0.027, 0.188] | 0.1710 |

Here b is the number of subjects correctly classified by model A and incorrectly classified by model B; c is the opposite discordant-pair count.
