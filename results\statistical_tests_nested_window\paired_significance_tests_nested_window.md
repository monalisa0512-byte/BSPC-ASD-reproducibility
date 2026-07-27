# Nested-Window Paired Statistical Tests

Exact McNemar is computed on paired participant-level correctness. Paired bootstrap reports effect-size intervals for Accuracy/F1 differences.

| Comparison | n | Acc A | Acc B | Delta Acc | F1 A | F1 B | Delta F1 | b | c | McNemar p | Delta Acc 95% CI | Bootstrap p | Delta F1 95% CI | Bootstrap p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AttentionNet nested vs CNNLSTM nested | 57 | 0.772 | 0.825 | -0.053 | 0.764 | 0.821 | -0.058 | 4 | 7 | 0.5488 | [-0.158, 0.053] | 0.4480 | [-0.165, 0.042] | 0.2856 |

## Window selections

AttentionNet:

window_size_ms  stride_ms
1000            500          49
1500            750           6
2000            500           1
                1000          1


CNNLSTM:

window_size_ms  stride_ms
1000            500          52
1500            750           3
2000            500           1
                1000          1

