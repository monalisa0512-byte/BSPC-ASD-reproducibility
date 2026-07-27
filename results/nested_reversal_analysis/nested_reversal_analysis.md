# Nested-window reversal analysis

## Model-level metrics

| model_setting | n | accuracy | f1 | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| att_nested | 57 | 0.7719 | 0.7636 | 23 | 7 | 6 | 21 |
| cnn_nested | 57 | 0.8246 | 0.8214 | 24 | 6 | 4 | 23 |
| att_fixed | 57 | 0.8596 | 0.8519 | 26 | 4 | 4 | 23 |
| cnn_fixed | 57 | 0.7895 | 0.7778 | 24 | 6 | 6 | 21 |

## Same vs different selected windows

| subset | n | attention_acc | cnnlstm_acc | attention_f1 | cnnlstm_f1 |
| --- | --- | --- | --- | --- | --- |
| same_selected_window | 46 | 0.7391 | 0.8043 | 0.7391 | 0.8085 |
| different_selected_window | 11 | 0.9091 | 0.9091 | 0.8889 | 0.8889 |

## Outcome groups

| outcome_group | n | mean_att_margin | mean_cnn_margin | mean_att_abs_margin | mean_cnn_abs_margin | mean_missing_rate | mean_severe_frame_rate | mean_selected_windows_att | mean_selected_windows_cnn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_only_correct | 4 | 0.0271 | -0.0643 | 0.0271 | 0.0643 | 0.2345 | 0.1551 | 554.5000 | 525.2500 |
| both_correct | 40 | 0.2375 | 0.2250 | 0.2375 | 0.2250 | 0.3112 | 0.3564 | 466.6000 | 474.3250 |
| both_wrong | 6 | -0.0769 | -0.0532 | 0.0769 | 0.0532 | 0.3122 | 0.2373 | 446.5000 | 446.5000 |
| cnnlstm_only_correct | 7 | -0.0736 | 0.0971 | 0.0736 | 0.0971 | 0.2843 | 0.2469 | 595.8571 | 609.5714 |

## Quality/margin correlations with correctness

| model | x | y | spearman_r | p |
| --- | --- | --- | --- | --- |
| AttentionNet_nested | mask_missing_rate | correct | -0.0407 | 0.7639 |
| AttentionNet_nested | severe_frame_rate | correct | 0.2008 | 0.1343 |
| AttentionNet_nested | selected_valid_windows | correct | -0.0508 | 0.7073 |
| AttentionNet_nested | abs_margin | correct | 0.4168 | 0.0013 |
| AttentionNet_nested | signed_margin | correct | 0.7269 | <0.0001 |
| CNNLSTM_nested | mask_missing_rate | correct | 0.0729 | 0.5900 |
| CNNLSTM_nested | severe_frame_rate | correct | 0.3280 | 0.0127 |
| CNNLSTM_nested | selected_valid_windows | correct | -0.0743 | 0.5828 |
| CNNLSTM_nested | abs_margin | correct | 0.3841 | 0.0032 |
| CNNLSTM_nested | signed_margin | correct | 0.6589 | <0.0001 |

## Discordant subjects

| pid | true_label_att | pred_att | pred_cnn | median_prob_att | threshold_att | signed_margin_att | median_prob_cnn | threshold_cnn | signed_margin_cnn | window_size_ms_att | stride_ms_att | window_size_ms_cnn | stride_ms_cnn | mask_missing_rate_att | severe_frame_rate_att | selected_valid_windows_att | selected_valid_windows_cnn | outcome_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 0 | 1 | 0.3206 | 0.3800 | -0.0594 | 0.4561 | 0.3400 | 0.1161 | 1000 | 500 | 1000 | 500 | 0.2966 | 0.2513 | 232 | 232 | cnnlstm_only_correct |
| 22 | 1 | 0 | 1 | 0.4210 | 0.5300 | -0.1090 | 0.5049 | 0.3000 | 0.2049 | 1000 | 500 | 1000 | 500 | 0.3789 | 0.2739 | 342 | 342 | cnnlstm_only_correct |
| 32 | 0 | 1 | 0 | 0.4067 | 0.4000 | -0.0067 | 0.4194 | 0.4200 | 0.0006 | 1000 | 500 | 1000 | 500 | 0.2287 | 0.2187 | 265 | 265 | cnnlstm_only_correct |
| 35 | 0 | 0 | 1 | 0.4985 | 0.5000 | 0.0015 | 0.5371 | 0.4900 | -0.0471 | 1000 | 500 | 1000 | 500 | 0.2102 | 0.1515 | 332 | 332 | attention_only_correct |
| 36 | 0 | 0 | 1 | 0.3481 | 0.3500 | 0.0019 | 0.4268 | 0.4100 | -0.0168 | 1500 | 750 | 1500 | 750 | 0.2652 | 0.1236 | 693 | 693 | attention_only_correct |
| 38 | 0 | 1 | 0 | 0.5210 | 0.3700 | -0.1510 | 0.3767 | 0.3800 | 0.0033 | 1500 | 750 | 1000 | 500 | 0.2695 | 0.2555 | 155 | 251 | cnnlstm_only_correct |
| 45 | 0 | 1 | 0 | 0.4087 | 0.3000 | -0.1087 | 0.4199 | 0.6300 | 0.2101 | 1000 | 500 | 1000 | 500 | 0.1312 | 0.1589 | 444 | 444 | cnnlstm_only_correct |
| 48 | 0 | 0 | 1 | 0.4482 | 0.5100 | 0.0618 | 0.4985 | 0.3900 | -0.1085 | 1000 | 500 | 1000 | 500 | 0.1977 | 0.2549 | 344 | 344 | attention_only_correct |
| 49 | 0 | 1 | 0 | 0.4336 | 0.4200 | -0.0136 | 0.4045 | 0.4200 | 0.0155 | 1000 | 500 | 1000 | 500 | 0.3631 | 0.3203 | 1353 | 1353 | cnnlstm_only_correct |
| 57 | 0 | 0 | 1 | 0.3667 | 0.4100 | 0.0433 | 0.3848 | 0.3000 | -0.0848 | 1000 | 500 | 2000 | 500 | 0.2648 | 0.0904 | 849 | 732 | attention_only_correct |
| 58 | 0 | 1 | 0 | 0.5070 | 0.4400 | -0.0670 | 0.3303 | 0.4600 | 0.1297 | 1000 | 500 | 1000 | 500 | 0.3219 | 0.2497 | 1380 | 1380 | cnnlstm_only_correct |
