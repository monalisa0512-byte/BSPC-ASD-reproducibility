# Subject-balanced sampling ablation with nested window selection

| mode | n_subjects | accuracy | f1 | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 57 | 0.7719 | 0.7636 | 23 | 7 | 6 | 21 |
| unbalanced | 57 | 0.7544 | 0.7083 | 26 | 4 | 10 | 17 |

## Paired comparison: balanced - unbalanced

- McNemar discordant counts: b=7, c=6, p=1.0000
- Accuracy delta: 0.0175, 95% bootstrap CI [-0.1053, 0.1404], p=0.8840
- F1 delta: 0.0553, 95% bootstrap CI [-0.0830, 0.1967], p=0.4208

## Window selections

### balanced

| window_size_ms | stride_ms | count |
| --- | --- | --- |
| 1000 | 500 | 49 |
| 1500 | 750 | 6 |
| 2000 | 500 | 1 |
| 2000 | 1000 | 1 |

### unbalanced

| window_size_ms | stride_ms | count |
| --- | --- | --- |
| 1000 | 500 | 41 |
| 1500 | 750 | 5 |
| 2000 | 500 | 8 |
| 2000 | 1000 | 3 |

