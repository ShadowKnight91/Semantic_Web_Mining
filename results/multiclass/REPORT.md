# Results report

Extraction run used for the KG: `rule`


## Balanced test split

| experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 |
|---|---|---|---|---|---|---|---|---|
| E1 | lr | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E1 | xgb | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E2 | lr | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E2 | xgb | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E2G | lr | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E2G | xgb | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |


## Imbalanced test split (~5% scams, mean over draws)

| experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 | recall_std | pr_auc_std |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | lr | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E1 | xgb | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E2 | lr | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E2 | xgb | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E2G | lr | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E2G | xgb | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |


## Cross-dataset generalization (X2)

Note: the LLM zero-shot rows use a stratified sample of each external set (see `n`); the same-subset table below scores every model on exactly those calls.

| eval_set | experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 |
|---|---|---|---|---|---|---|---|---|---|
| pranks_control | E1 | lr | 36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pranks_control | E1 | xgb | 36 | 1.000 | 0.375 | 0.545 | 0.997 | 0.993 | 1.000 |
| pranks_control | E2 | lr | 36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pranks_control | E2 | xgb | 36 | 1.000 | 0.958 | 0.979 | 1.000 | 1.000 | 1.000 |
| pranks_control | E2G | lr | 36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pranks_control | E2G | xgb | 36 | 1.000 | 0.958 | 0.979 | 1.000 | 1.000 | 1.000 |


### Recall / PR-AUC drop vs. in-distribution test

| experiment | model | eval_set | recall | recall_drop | pr_auc | pr_auc_drop |
|---|---|---|---|---|---|---|
| E1 | lr | pranks_control | 1.000 | 0.000 | 1.000 | 0.000 |
| E1 | xgb | pranks_control | 0.375 | 0.625 | 0.997 | 0.003 |
| E2 | lr | pranks_control | 1.000 | 0.000 | 1.000 | 0.000 |
| E2 | xgb | pranks_control | 0.958 | 0.042 | 1.000 | 0.000 |
| E2G | lr | pranks_control | 1.000 | 0.000 | 1.000 | -0.000 |
| E2G | xgb | pranks_control | 0.958 | 0.042 | 1.000 | 0.000 |


## Caller-type classification (fraudster / salesperson / prankster / other legitimate)

Training calls (train + val): fraudster 679, other_legitimate 346, salesperson 333, prankster 102. Main model: E2/lr; KG run: `rule`.

| experiment | model | eval_set | n | accuracy | macro_f1 | f1_fraudster | f1_salesperson | f1_prankster | f1_other_legitimate | predicted_prankster |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | lr | test | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.070 |
| E1 | lr | pranks_control | 36 | 0.778 | 0.812 | 0.957 | 0.667 |  |  | 0.222 |
| E1 | rf | test | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.070 |
| E1 | rf | pranks_control | 36 | 0.472 | 0.527 | 0.769 | 0.286 |  |  | 0.528 |
| E2 | lr | test | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.070 |
| E2 | lr | pranks_control | 36 | 1.000 | 1.000 | 1.000 | 1.000 |  |  | 0.000 |
| E2 | rf | test | 258 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.070 |
| E2 | rf | pranks_control | 36 | 0.722 | 0.689 | 0.979 | 0.400 |  |  | 0.278 |
| E2G | lr | test | 258 | 0.864 | 0.776 | 1.000 | 0.921 | 0.493 | 0.691 | 0.198 |
| E2G | lr | pranks_control | 36 | 0.972 | 0.969 | 0.979 | 0.960 |  |  | 0.000 |
| E2G | rf | test | 258 | 0.930 | 0.834 | 0.996 | 0.947 | 0.519 | 0.874 | 0.035 |
| E2G | rf | pranks_control | 36 | 0.972 | 0.969 | 0.979 | 0.960 |  |  | 0.000 |


Confusion matrix, test split (E2/lr; rows = true type, columns = predicted):

| true | fraudster | salesperson | prankster | other_legitimate |
|---|---|---|---|---|
| fraudster | 120 | 0 | 0 | 0 |
| salesperson | 0 | 67 | 0 | 0 |
| prankster | 0 | 0 | 18 | 0 |
| other_legitimate | 0 | 0 | 0 | 53 |


Style check: 36 fraud / sales calls written by the same model and prompt template as the prank calls; share predicted `prankster`: E1/lr 22%, E1/rf 53%, E2/lr 0%, E2/rf 28%, E2G/lr 0%, E2G/rf 0%. A model that labels them `prankster` recognizes the generator's style rather than prank content.
