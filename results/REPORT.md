# Results report

Extraction run used for the KG: `llm_voyager_gemma4-31b-it_v1_full_fb-qwen3-30b-a3b`


## Balanced test split

| experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 |
|---|---|---|---|---|---|---|---|---|
| E1 | lr | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E1 | rf | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E1 | svm | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E1 | xgb | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2 | lr | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2 | rf | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2 | svm | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2 | xgb | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2G | lr | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2G | rf | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2G | svm | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E2G | xgb | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| E3 | distilbert | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E3 | sbert_mlp | 240 | 0.984 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 |
| E4 | graphsage | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E4 | graphsage_text | 240 | 0.992 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 |
| E4 | graphsage_topo | 240 | 1.000 | 0.992 | 0.996 | 1.000 | 1.000 | 1.000 |
| E4 | graphsage_xgb | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LLM | zeroshot_gemma4-31b-it | 240 | 0.992 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 |


## Imbalanced test split (~5% scams, mean over draws)

| experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 | recall_std | pr_auc_std |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | lr | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E1 | rf | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E1 | svm | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E1 | xgb | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2 | lr | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2 | rf | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2 | svm | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2 | xgb | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2G | lr | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2G | rf | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2G | svm | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E2G | xgb | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| E3 | distilbert | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E3 | sbert_mlp | 126 | 0.750 | 1.000 | 0.857 | 0.999 | 1.000 | 1.000 | 0.000 | 0.005 |
| E4 | graphsage | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E4 | graphsage_text | 126 | 0.857 | 1.000 | 0.923 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| E4 | graphsage_topo | 126 | 1.000 | 0.992 | 0.995 | 1.000 | 1.000 | 1.000 | 0.037 | 0.000 |
| E4 | graphsage_xgb | 126 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| LLM | zeroshot_gemma4-31b-it | 126 | 0.857 | 1.000 | 0.923 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |


## Cross-dataset generalization (X2)

| eval_set | experiment | model | n | precision | recall | f1 | pr_auc | roc_auc | recall_at_fpr10 |
|---|---|---|---|---|---|---|---|---|---|
| kaggle | E1 | lr | 793 | 0.938 | 0.458 | 0.615 | 0.846 | 0.818 |  |
| kaggle | E1 | rf | 793 | 1.000 | 0.033 | 0.064 | 0.896 | 0.890 |  |
| kaggle | E1 | svm | 793 | 0.930 | 0.405 | 0.564 | 0.845 | 0.815 |  |
| kaggle | E1 | xgb | 793 | 0.933 | 0.071 | 0.132 | 0.830 | 0.846 |  |
| kaggle | E2 | lr | 793 | 0.840 | 0.588 | 0.692 | 0.888 | 0.904 |  |
| kaggle | E2 | rf | 793 | 0.899 | 0.407 | 0.560 | 0.882 | 0.893 |  |
| kaggle | E2 | svm | 793 | 0.866 | 0.593 | 0.704 | 0.893 | 0.912 |  |
| kaggle | E2 | xgb | 793 | 0.771 | 0.412 | 0.537 | 0.709 | 0.776 |  |
| kaggle | E2G | lr | 793 | 0.846 | 0.588 | 0.694 | 0.889 | 0.905 |  |
| kaggle | E2G | rf | 793 | 0.811 | 0.547 | 0.653 | 0.852 | 0.864 |  |
| kaggle | E2G | svm | 793 | 0.904 | 0.601 | 0.722 | 0.898 | 0.915 |  |
| kaggle | E2G | xgb | 793 | 0.771 | 0.412 | 0.537 | 0.698 | 0.772 |  |
| kaggle | E3 | distilbert | 793 | 0.693 | 0.758 | 0.724 | 0.805 | 0.809 | 0.461 |
| kaggle | E3 | sbert_mlp | 793 | 0.534 | 0.677 | 0.597 | 0.717 | 0.644 | 0.422 |
| kaggle | E4 | graphsage | 793 | 0.922 | 0.598 | 0.725 | 0.917 | 0.926 | 0.776 |
| kaggle | E4 | graphsage_text | 793 | 0.891 | 0.684 | 0.774 | 0.904 | 0.913 | 0.723 |
| kaggle | E4 | graphsage_topo | 793 | 0.970 | 0.417 | 0.584 | 0.952 | 0.960 | 0.913 |
| kaggle | E4 | graphsage_xgb | 793 | 0.923 | 0.491 | 0.641 | 0.791 | 0.814 | 0.692 |
| kaggle | LLM | zeroshot_gemma4-31b-it | 367 | 0.962 | 0.852 | 0.904 | 0.953 | 0.970 | 0.913 |
| ncsu | E1 | lr | 713 | 1.000 | 0.432 | 0.603 |  |  |  |
| ncsu | E1 | rf | 713 | 1.000 | 0.013 | 0.025 |  |  |  |
| ncsu | E1 | svm | 713 | 1.000 | 0.424 | 0.595 |  |  |  |
| ncsu | E1 | xgb | 713 | 1.000 | 0.013 | 0.025 |  |  |  |
| ncsu | E2 | lr | 713 | 1.000 | 0.811 | 0.895 |  |  |  |
| ncsu | E2 | rf | 713 | 1.000 | 0.403 | 0.574 |  |  |  |
| ncsu | E2 | svm | 713 | 1.000 | 0.811 | 0.895 |  |  |  |
| ncsu | E2 | xgb | 713 | 1.000 | 0.760 | 0.864 |  |  |  |
| ncsu | E2G | lr | 713 | 1.000 | 0.808 | 0.894 |  |  |  |
| ncsu | E2G | rf | 713 | 1.000 | 0.790 | 0.882 |  |  |  |
| ncsu | E2G | svm | 713 | 1.000 | 0.804 | 0.891 |  |  |  |
| ncsu | E2G | xgb | 713 | 1.000 | 0.760 | 0.864 |  |  |  |
| ncsu | E3 | distilbert | 713 | 1.000 | 0.788 | 0.882 |  |  |  |
| ncsu | E3 | sbert_mlp | 713 | 1.000 | 0.917 | 0.957 |  |  |  |
| ncsu | E4 | graphsage | 713 | 1.000 | 0.727 | 0.842 |  |  |  |
| ncsu | E4 | graphsage_text | 713 | 1.000 | 0.842 | 0.914 |  |  |  |
| ncsu | E4 | graphsage_topo | 713 | 1.000 | 0.137 | 0.242 |  |  |  |
| ncsu | E4 | graphsage_xgb | 713 | 1.000 | 0.633 | 0.775 |  |  |  |
| ncsu | LLM | zeroshot_gemma4-31b-it | 414 | 1.000 | 0.920 | 0.958 |  |  |  |
| ncsu+kaggle | E1 | lr | 1113 | 0.963 | 0.432 | 0.596 | 0.922 | 0.867 |  |
| ncsu+kaggle | E1 | rf | 1113 | 1.000 | 0.013 | 0.025 | 0.933 | 0.882 |  |
| ncsu+kaggle | E1 | svm | 1113 | 0.962 | 0.424 | 0.588 | 0.926 | 0.873 |  |
| ncsu+kaggle | E1 | xgb | 1113 | 0.818 | 0.013 | 0.025 | 0.835 | 0.782 |  |
| ncsu+kaggle | E2 | lr | 1113 | 0.929 | 0.811 | 0.866 | 0.958 | 0.942 |  |
| ncsu+kaggle | E2 | rf | 1113 | 0.941 | 0.403 | 0.564 | 0.939 | 0.916 |  |
| ncsu+kaggle | E2 | svm | 1113 | 0.941 | 0.811 | 0.871 | 0.958 | 0.947 |  |
| ncsu+kaggle | E2 | xgb | 1113 | 0.919 | 0.760 | 0.832 | 0.842 | 0.825 |  |
| ncsu+kaggle | E2G | lr | 1113 | 0.932 | 0.808 | 0.866 | 0.959 | 0.943 |  |
| ncsu+kaggle | E2G | rf | 1113 | 0.918 | 0.790 | 0.849 | 0.914 | 0.911 |  |
| ncsu+kaggle | E2G | svm | 1113 | 0.958 | 0.804 | 0.874 | 0.956 | 0.946 |  |
| ncsu+kaggle | E2G | xgb | 1113 | 0.919 | 0.760 | 0.832 | 0.816 | 0.812 |  |
| ncsu+kaggle | E3 | distilbert | 1113 | 0.810 | 0.788 | 0.799 | 0.894 | 0.806 | 0.589 |
| ncsu+kaggle | E3 | sbert_mlp | 1113 | 0.738 | 0.917 | 0.818 | 0.904 | 0.838 | 0.579 |
| ncsu+kaggle | E4 | graphsage | 1113 | 0.963 | 0.727 | 0.828 | 0.970 | 0.958 | 0.910 |
| ncsu+kaggle | E4 | graphsage_text | 1113 | 0.948 | 0.842 | 0.892 | 0.966 | 0.950 | 0.857 |
| ncsu+kaggle | E4 | graphsage_topo | 1113 | 0.951 | 0.137 | 0.240 | 0.942 | 0.927 | 0.802 |
| ncsu+kaggle | E4 | graphsage_xgb | 1113 | 0.966 | 0.633 | 0.764 | 0.913 | 0.890 | 0.837 |
| ncsu+kaggle | LLM | zeroshot_gemma4-31b-it | 632 | 0.987 | 0.920 | 0.953 | 0.989 | 0.983 | 0.969 |


### Recall / PR-AUC drop vs. in-distribution test

| experiment | model | eval_set | recall | recall_drop | pr_auc | pr_auc_drop |
|---|---|---|---|---|---|---|
| E1 | lr | kaggle | 0.458 | 0.542 | 0.846 | 0.154 |
| E1 | lr | ncsu | 0.432 | 0.568 |  |  |
| E1 | lr | ncsu+kaggle | 0.432 | 0.568 | 0.922 | 0.078 |
| E1 | svm | kaggle | 0.405 | 0.595 | 0.845 | 0.155 |
| E1 | svm | ncsu | 0.424 | 0.576 |  |  |
| E1 | svm | ncsu+kaggle | 0.424 | 0.576 | 0.926 | 0.074 |
| E1 | rf | kaggle | 0.033 | 0.967 | 0.896 | 0.104 |
| E1 | rf | ncsu | 0.013 | 0.987 |  |  |
| E1 | rf | ncsu+kaggle | 0.013 | 0.987 | 0.933 | 0.067 |
| E1 | xgb | kaggle | 0.071 | 0.929 | 0.830 | 0.170 |
| E1 | xgb | ncsu | 0.013 | 0.987 |  |  |
| E1 | xgb | ncsu+kaggle | 0.013 | 0.987 | 0.835 | 0.165 |
| E2 | lr | kaggle | 0.588 | 0.412 | 0.888 | 0.112 |
| E2 | lr | ncsu | 0.811 | 0.189 |  |  |
| E2 | lr | ncsu+kaggle | 0.811 | 0.189 | 0.958 | 0.042 |
| E2 | svm | kaggle | 0.593 | 0.407 | 0.893 | 0.107 |
| E2 | svm | ncsu | 0.811 | 0.189 |  |  |
| E2 | svm | ncsu+kaggle | 0.811 | 0.189 | 0.958 | 0.042 |
| E2 | rf | kaggle | 0.407 | 0.593 | 0.882 | 0.118 |
| E2 | rf | ncsu | 0.403 | 0.597 |  |  |
| E2 | rf | ncsu+kaggle | 0.403 | 0.597 | 0.939 | 0.061 |
| E2 | xgb | kaggle | 0.412 | 0.588 | 0.709 | 0.291 |
| E2 | xgb | ncsu | 0.760 | 0.240 |  |  |
| E2 | xgb | ncsu+kaggle | 0.760 | 0.240 | 0.842 | 0.158 |
| E2G | lr | kaggle | 0.588 | 0.412 | 0.889 | 0.111 |
| E2G | lr | ncsu | 0.808 | 0.192 |  |  |
| E2G | lr | ncsu+kaggle | 0.808 | 0.192 | 0.959 | 0.041 |
| E2G | svm | kaggle | 0.601 | 0.399 | 0.898 | 0.102 |
| E2G | svm | ncsu | 0.804 | 0.196 |  |  |
| E2G | svm | ncsu+kaggle | 0.804 | 0.196 | 0.956 | 0.044 |
| E2G | rf | kaggle | 0.547 | 0.453 | 0.852 | 0.148 |
| E2G | rf | ncsu | 0.790 | 0.210 |  |  |
| E2G | rf | ncsu+kaggle | 0.790 | 0.210 | 0.914 | 0.086 |
| E2G | xgb | kaggle | 0.412 | 0.588 | 0.698 | 0.302 |
| E2G | xgb | ncsu | 0.760 | 0.240 |  |  |
| E2G | xgb | ncsu+kaggle | 0.760 | 0.240 | 0.816 | 0.184 |
| E3 | sbert_mlp | kaggle | 0.677 | 0.323 | 0.717 | 0.283 |
| E3 | sbert_mlp | ncsu | 0.917 | 0.083 |  |  |
| E3 | sbert_mlp | ncsu+kaggle | 0.917 | 0.083 | 0.904 | 0.095 |
| E3 | distilbert | kaggle | 0.758 | 0.242 | 0.805 | 0.195 |
| E3 | distilbert | ncsu | 0.788 | 0.212 |  |  |
| E3 | distilbert | ncsu+kaggle | 0.788 | 0.212 | 0.894 | 0.106 |
| E4 | graphsage | kaggle | 0.598 | 0.402 | 0.917 | 0.083 |
| E4 | graphsage | ncsu | 0.727 | 0.273 |  |  |
| E4 | graphsage | ncsu+kaggle | 0.727 | 0.273 | 0.970 | 0.030 |
| E4 | graphsage_text | kaggle | 0.684 | 0.316 | 0.904 | 0.096 |
| E4 | graphsage_text | ncsu | 0.842 | 0.158 |  |  |
| E4 | graphsage_text | ncsu+kaggle | 0.842 | 0.158 | 0.966 | 0.034 |
| E4 | graphsage_topo | kaggle | 0.417 | 0.574 | 0.952 | 0.048 |
| E4 | graphsage_topo | ncsu | 0.137 | 0.854 |  |  |
| E4 | graphsage_topo | ncsu+kaggle | 0.137 | 0.854 | 0.942 | 0.058 |
| E4 | graphsage_xgb | kaggle | 0.491 | 0.509 | 0.791 | 0.209 |
| E4 | graphsage_xgb | ncsu | 0.633 | 0.367 |  |  |
| E4 | graphsage_xgb | ncsu+kaggle | 0.633 | 0.367 | 0.913 | 0.087 |
| LLM | zeroshot_gemma4-31b-it | kaggle | 0.852 | 0.148 | 0.953 | 0.047 |
| LLM | zeroshot_gemma4-31b-it | ncsu | 0.920 | 0.080 |  |  |
| LLM | zeroshot_gemma4-31b-it | ncsu+kaggle | 0.920 | 0.080 | 0.989 | 0.011 |


## 5-fold stratified cross-validation (mean / std)

| experiment | model | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | pr_auc_mean | pr_auc_std | roc_auc_mean | roc_auc_std | recall_at_fpr10_mean | recall_at_fpr10_std | accuracy_mean | accuracy_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | lr | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| E1 | rf | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| E1 | svm | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| E1 | xgb | 0.999 | 0.003 | 0.998 | 0.006 | 0.998 | 0.003 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.998 | 0.003 |
| E2 | lr | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2 | rf | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| E2 | svm | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2 | xgb | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2G | lr | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2G | rf | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2G | svm | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |
| E2G | xgb | 0.999 | 0.003 | 1.000 | 0.000 | 0.999 | 0.001 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.999 | 0.001 |


## McNemar tests (E1 vs graph-enhanced)

| model | baseline | challenger | eval_set | n | b_only_correct | a_only_correct | statistic | p_value | method | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | E1 | E2 | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| lr | E1 | E2 | kaggle | 793 | 92 | 73 | 1.964 | 0.161 | chi2_cc | False |
| lr | E1 | E2 | ncsu | 713 | 281 | 11 | 247.812 | 0.000 | chi2_cc | True |
| lr | E1 | E2 | ncsu+kaggle | 1113 | 287 | 49 | 167.170 | 0.000 | chi2_cc | True |
| lr | E1 | E2G | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| lr | E1 | E2G | kaggle | 793 | 92 | 71 | 2.454 | 0.117 | chi2_cc | False |
| lr | E1 | E2G | ncsu | 713 | 279 | 11 | 245.824 | 0.000 | chi2_cc | True |
| lr | E1 | E2G | ncsu+kaggle | 1113 | 285 | 47 | 169.184 | 0.000 | chi2_cc | True |
| rf | E1 | E2 | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| rf | E1 | E2 | kaggle | 793 | 151 | 22 | 94.705 | 0.000 | chi2_cc | True |
| rf | E1 | E2 | ncsu | 713 | 278 | 0 | 276.004 | 0.000 | chi2_cc | True |
| rf | E1 | E2 | ncsu+kaggle | 1113 | 278 | 18 | 226.625 | 0.000 | chi2_cc | True |
| rf | E1 | E2G | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| rf | E1 | E2G | kaggle | 793 | 206 | 54 | 87.696 | 0.000 | chi2_cc | True |
| rf | E1 | E2G | ncsu | 713 | 554 | 0 | 552.002 | 0.000 | chi2_cc | True |
| rf | E1 | E2G | ncsu+kaggle | 1113 | 554 | 50 | 418.889 | 0.000 | chi2_cc | True |
| svm | E1 | E2 | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| svm | E1 | E2 | kaggle | 793 | 110 | 60 | 14.124 | 0.000 | chi2_cc | True |
| svm | E1 | E2 | ncsu | 713 | 284 | 8 | 258.990 | 0.000 | chi2_cc | True |
| svm | E1 | E2 | ncsu+kaggle | 1113 | 290 | 38 | 192.076 | 0.000 | chi2_cc | True |
| svm | E1 | E2G | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| svm | E1 | E2G | kaggle | 793 | 114 | 50 | 24.201 | 0.000 | chi2_cc | True |
| svm | E1 | E2G | ncsu | 713 | 280 | 9 | 252.249 | 0.000 | chi2_cc | True |
| svm | E1 | E2G | ncsu+kaggle | 1113 | 287 | 29 | 209.016 | 0.000 | chi2_cc | True |
| xgb | E1 | E2 | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| xgb | E1 | E2 | kaggle | 793 | 150 | 62 | 35.703 | 0.000 | chi2_cc | True |
| xgb | E1 | E2 | ncsu | 713 | 534 | 1 | 529.017 | 0.000 | chi2_cc | True |
| xgb | E1 | E2 | ncsu+kaggle | 1113 | 535 | 48 | 405.139 | 0.000 | chi2_cc | True |
| xgb | E1 | E2G | test | 240 | 0 | 0 | 0.000 | 1.000 | none | False |
| xgb | E1 | E2G | kaggle | 793 | 150 | 62 | 35.703 | 0.000 | chi2_cc | True |
| xgb | E1 | E2G | ncsu | 713 | 534 | 1 | 529.017 | 0.000 | chi2_cc | True |
| xgb | E1 | E2G | ncsu+kaggle | 1113 | 535 | 48 | 405.139 | 0.000 | chi2_cc | True |


## Extraction quality (X1)

| run | model | prompt_version | n_calls | json_validity_rate | schema_conformance_rate | mean_triples_per_call | repaired_rate | inferred_rate | retry_rate | salvage_rate | fallback_rate | failed_rate | mean_latency_s | triple_precision | triple_recall | triple_f1 | macro_precision | macro_recall | entity_type_accuracy | gold_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | rule |  | 3104 | 1.000 | 1.000 | 2.688 | 0.000 | 0.081 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.773 | 0.589 | 0.669 | 0.674 | 0.542 | 1.000 | 50 |
| llm_voyager_gemma4-31b-it_v1_full_fb-qwen3-30b-a3b | gemma4-31b-it | v1_full | 3104 | 1.000 | 0.984 | 4.008 | 0.000 | 0.108 | 0.000 | 0.000 | 0.000 | 0.000 | 8.664 | 0.897 | 0.906 | 0.901 | 0.794 | 0.870 | 1.000 | 50 |
| llm_voyager_qwen3-30b-a3b_v1_full | qwen3-30b-a3b | v1_full | 287 | 0.983 | 0.617 | 5.829 | 0.003 | 0.129 | 0.000 | 0.017 | 0.000 | 0.000 | 10.191 | 0.617 | 0.837 | 0.710 | 0.579 | 0.834 | 1.000 | 50 |
| llm_voyager_gemma4-31b-it_v2_no_descriptors | gemma4-31b-it | v2_no_descriptors | 287 | 1.000 | 0.958 | 4.589 | 0.000 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 10.060 | 0.917 | 0.926 | 0.921 | 0.852 | 0.893 | 1.000 | 50 |
| llm_voyager_gemma4-31b-it_v3_no_example | gemma4-31b-it | v3_no_example | 287 | 1.000 | 0.944 | 4.436 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 9.449 | 0.929 | 0.901 | 0.915 | 0.862 | 0.892 | 1.000 | 50 |
| llm_voyager_gemma4-31b-it_v0_open | gemma4-31b-it | v0_open | 287 | 1.000 | 0.129 | 2.836 | 0.010 | 0.095 | 0.941 | 0.000 | 0.000 | 0.000 | 11.424 | 0.798 | 0.510 | 0.622 | 0.708 | 0.508 | 1.000 | 50 |


## GraphSAGE input fusion (E4)

Text block -> 64-d, topology block (19 label-free features) -> 16-d, node type -> 8-d; concatenated, LayerNorm, then 2 mean-aggregation SAGE layers. Training graph: inductive (no test / external calls).

Topology features: `het_degree`, `het_pagerank`, `het_betweenness`, `cc_degree`, `cc_weighted_degree`, `cc_pagerank`, `entity_mean_popularity`, `entity_max_popularity`, `community_size`, `n_CLAIMS_TO_BE`, `n_MENTIONS_ORG`, `n_USES_TACTIC`, `n_REQUESTS`, `n_PAY_VIA`, `n_PROVIDES_NUMBER`, `n_inferred`, `has_entities`, `entity_log_degree`, `entity_pagerank`

| variant | model | val_pr_auc | epochs |
|---|---|---|---|
| text+topo | graphsage | 1.000 | 39 |
| text | graphsage_text | 1.000 | 46 |
| topo | graphsage_topo | 1.000 | 61 |


## Clustering (E5)

| method | k | purity | nmi | ari | n_clusters | noise_frac | nmi_vs_subtype | eps | eps_percentile | silhouette |
|---|---|---|---|---|---|---|---|---|---|---|
| kmeans_best | 11.000 | 0.986 | 0.448 | 0.235 | 11 | 0.000 | 0.758 |  |  |  |
| kmeans_k2 | 2.000 | 0.751 | 0.345 | 0.252 | 2 | 0.000 |  |  |  |  |
| dbscan |  | 0.864 | 0.240 | 0.088 | 63 | 0.459 |  | 1.000 | 50.000 | 0.701 |


### Discovered scripts

| cluster | size | scam_rate | signature | dominant_subtype |
|---|---|---|---|---|
| 0 | 291 | 0.000 | delivery_service | delivery (63%) |
| 3 | 228 | 0.000 | insurance -> reward+urgency | insurance (83%) |
| 2 | 199 | 1.000 | government_agent -> fear+urgency -> ssn | ssn (98%) |
| 10 | 189 | 1.000 | tech_support -> fear+urgency -> remote_access | support (92%) |
| 4 | 174 | 0.977 | prize_or_lottery -> reward -> personal_info+ssn | reward (82%) |
| 7 | 158 | 1.000 | retailer_customer_service -> reward+urgency -> personal_info+bank_account | refund (83%) |
| 9 | 158 | 0.019 | telemarketer_sales -> reward | telemarketing (97%) |
| 6 | 100 | 0.000 | (no dominant entities) | wrong (96%) |
| 1 | 65 | 0.908 | retailer_customer_service -> reward+urgency -> money+personal_info -> credit_card_payment+gift_card | reward (43%) |
| 8 | 31 | 0.677 | tech_support+delivery_service -> fear+urgency -> remote_access | support (61%) |
| 5 | 5 | 0.000 | relative_or_friend -> personal_info | wrong (80%) |


## Ablations (cross-dataset sets, delta vs. full model)

The BothBosu test split is saturated (all deltas 0); negative values mean the removed part helped.

| eval_set | ablation | removed | experiment | model | pr_auc | delta_pr_auc | delta_recall |
|---|---|---|---|---|---|---|---|
| kaggle | entity_type | PressureTactic | E2 | lr | 0.770 | -0.118 | -0.186 |
| kaggle | entity_type | RequestedAsset | E2 | lr | 0.825 | -0.063 | 0.076 |
| kaggle | entity_type | Organization | E2 | lr | 0.836 | -0.052 | -0.064 |
| kaggle | feature_group | relation_counts | E2 | lr | 0.862 | -0.026 | 0.033 |
| kaggle | feature_group | structural | E2 | lr | 0.863 | -0.025 | -0.043 |
| kaggle | entity_type | PaymentChannel | E2 | lr | 0.882 | -0.006 | 0.059 |
| kaggle | feature_group | similarity | E2 | lr | 0.882 | -0.006 | -0.046 |
| kaggle | feature_group | entity_risk | E2 | lr | 0.887 | -0.001 | 0.033 |
| kaggle | feature_group | ontology_onehot | E2 | lr | 0.890 | 0.002 | 0.010 |
| kaggle | feature_group | community | E2 | lr | 0.894 | 0.006 | 0.053 |
| kaggle | entity_type | CallbackNumber | E2 | lr | 0.896 | 0.008 | 0.031 |
| kaggle | entity_type | ClaimedIdentity | E2 | lr | 0.962 | 0.074 | 0.155 |
| kaggle | entity_type | RequestedAsset | E2 | xgb | 0.666 | -0.043 | -0.249 |
| kaggle | entity_type | Organization | E2 | xgb | 0.688 | -0.020 | -0.010 |
| kaggle | feature_group | structural | E2 | xgb | 0.697 | -0.011 | 0.000 |
| kaggle | feature_group | community | E2 | xgb | 0.700 | -0.009 | 0.000 |
| kaggle | feature_group | relation_counts | E2 | xgb | 0.709 | 0.000 | 0.000 |
| kaggle | feature_group | entity_risk | E2 | xgb | 0.710 | 0.001 | 0.000 |
| kaggle | entity_type | PressureTactic | E2 | xgb | 0.714 | 0.005 | -0.041 |
| kaggle | feature_group | ontology_onehot | E2 | xgb | 0.717 | 0.009 | 0.000 |
| kaggle | entity_type | CallbackNumber | E2 | xgb | 0.724 | 0.015 | -0.038 |
| kaggle | entity_type | PaymentChannel | E2 | xgb | 0.756 | 0.048 | 0.066 |
| kaggle | feature_group | similarity | E2 | xgb | 0.784 | 0.075 | 0.089 |
| kaggle | entity_type | ClaimedIdentity | E2 | xgb | 0.953 | 0.244 | 0.165 |
| kaggle | entity_type | PressureTactic | E2G | lr | 0.770 | -0.119 | -0.186 |
| kaggle | entity_type | RequestedAsset | E2G | lr | 0.827 | -0.062 | 0.069 |
| kaggle | entity_type | Organization | E2G | lr | 0.837 | -0.053 | -0.064 |
| kaggle | feature_group | relation_counts | E2G | lr | 0.862 | -0.028 | 0.041 |
| kaggle | feature_group | structural | E2G | lr | 0.866 | -0.024 | -0.056 |
| kaggle | entity_type | PaymentChannel | E2G | lr | 0.884 | -0.006 | 0.053 |
| kaggle | feature_group | similarity | E2G | lr | 0.884 | -0.006 | -0.053 |
| kaggle | feature_group | entity_risk | E2G | lr | 0.889 | -0.001 | 0.028 |
| kaggle | feature_group | ontology_onehot | E2G | lr | 0.891 | 0.002 | 0.005 |
| kaggle | feature_group | community | E2G | lr | 0.896 | 0.007 | 0.051 |
| kaggle | entity_type | CallbackNumber | E2G | lr | 0.897 | 0.008 | 0.028 |
| kaggle | entity_type | ClaimedIdentity | E2G | lr | 0.961 | 0.072 | 0.155 |
| kaggle | entity_type | PressureTactic | E2G | xgb | 0.631 | -0.067 | -0.018 |
| kaggle | entity_type | Organization | E2G | xgb | 0.678 | -0.020 | -0.010 |
| kaggle | entity_type | RequestedAsset | E2G | xgb | 0.691 | -0.006 | -0.249 |
| kaggle | feature_group | community | E2G | xgb | 0.700 | 0.003 | 0.000 |
| kaggle | feature_group | ontology_onehot | E2G | xgb | 0.700 | 0.003 | 0.000 |
| kaggle | feature_group | entity_risk | E2G | xgb | 0.708 | 0.010 | 0.000 |
| kaggle | feature_group | structural | E2G | xgb | 0.710 | 0.012 | 0.000 |
| kaggle | feature_group | relation_counts | E2G | xgb | 0.717 | 0.019 | 0.000 |
| kaggle | entity_type | CallbackNumber | E2G | xgb | 0.725 | 0.027 | -0.038 |
| kaggle | entity_type | PaymentChannel | E2G | xgb | 0.794 | 0.097 | 0.066 |
| kaggle | feature_group | similarity | E2G | xgb | 0.827 | 0.130 | 0.084 |
| kaggle | entity_type | ClaimedIdentity | E2G | xgb | 0.967 | 0.269 | 0.249 |
| ncsu | entity_type | ClaimedIdentity | E2 | lr |  |  | -0.137 |
| ncsu | entity_type | Organization | E2 | lr |  |  | -0.013 |
| ncsu | entity_type | PressureTactic | E2 | lr |  |  | -0.258 |
| ncsu | entity_type | RequestedAsset | E2 | lr |  |  | 0.049 |
| ncsu | entity_type | PaymentChannel | E2 | lr |  |  | -0.036 |
| ncsu | entity_type | CallbackNumber | E2 | lr |  |  | -0.020 |
| ncsu | feature_group | structural | E2 | lr |  |  | -0.060 |
| ncsu | feature_group | community | E2 | lr |  |  | 0.017 |
| ncsu | feature_group | similarity | E2 | lr |  |  | -0.080 |
| ncsu | feature_group | entity_risk | E2 | lr |  |  | 0.011 |
| ncsu | feature_group | relation_counts | E2 | lr |  |  | 0.000 |
| ncsu | feature_group | ontology_onehot | E2 | lr |  |  | 0.004 |
| ncsu | entity_type | ClaimedIdentity | E2 | xgb |  |  | -0.108 |
| ncsu | entity_type | Organization | E2 | xgb |  |  | -0.137 |
| ncsu | entity_type | PressureTactic | E2 | xgb |  |  | -0.212 |
| ncsu | entity_type | RequestedAsset | E2 | xgb |  |  | -0.537 |
| ncsu | entity_type | PaymentChannel | E2 | xgb |  |  | -0.007 |
| ncsu | entity_type | CallbackNumber | E2 | xgb |  |  | -0.013 |
| ncsu | feature_group | structural | E2 | xgb |  |  | 0.000 |
| ncsu | feature_group | community | E2 | xgb |  |  | 0.000 |
| ncsu | feature_group | similarity | E2 | xgb |  |  | -0.032 |
| ncsu | feature_group | entity_risk | E2 | xgb |  |  | 0.000 |
| ncsu | feature_group | relation_counts | E2 | xgb |  |  | 0.000 |
| ncsu | feature_group | ontology_onehot | E2 | xgb |  |  | 0.000 |
| ncsu | entity_type | ClaimedIdentity | E2G | lr |  |  | -0.135 |
| ncsu | entity_type | Organization | E2G | lr |  |  | -0.011 |
| ncsu | entity_type | PressureTactic | E2G | lr |  |  | -0.257 |
| ncsu | entity_type | RequestedAsset | E2G | lr |  |  | 0.052 |
| ncsu | entity_type | PaymentChannel | E2G | lr |  |  | -0.035 |
| ncsu | entity_type | CallbackNumber | E2G | lr |  |  | -0.022 |
| ncsu | feature_group | structural | E2G | lr |  |  | -0.058 |
| ncsu | feature_group | community | E2G | lr |  |  | 0.018 |
| ncsu | feature_group | similarity | E2G | lr |  |  | -0.080 |
| ncsu | feature_group | entity_risk | E2G | lr |  |  | 0.013 |
| ncsu | feature_group | relation_counts | E2G | lr |  |  | 0.001 |
| ncsu | feature_group | ontology_onehot | E2G | lr |  |  | 0.006 |
| ncsu | entity_type | ClaimedIdentity | E2G | xgb |  |  | -0.080 |
| ncsu | entity_type | Organization | E2G | xgb |  |  | -0.137 |
| ncsu | entity_type | PressureTactic | E2G | xgb |  |  | -0.160 |
| ncsu | entity_type | RequestedAsset | E2G | xgb |  |  | -0.537 |
| ncsu | entity_type | PaymentChannel | E2G | xgb |  |  | -0.007 |
| ncsu | entity_type | CallbackNumber | E2G | xgb |  |  | -0.013 |
| ncsu | feature_group | structural | E2G | xgb |  |  | 0.000 |
| ncsu | feature_group | community | E2G | xgb |  |  | 0.000 |
| ncsu | feature_group | similarity | E2G | xgb |  |  | -0.066 |
| ncsu | feature_group | entity_risk | E2G | xgb |  |  | 0.000 |
| ncsu | feature_group | relation_counts | E2G | xgb |  |  | 0.000 |
| ncsu | feature_group | ontology_onehot | E2G | xgb |  |  | 0.000 |
| ncsu+kaggle | entity_type | PressureTactic | E2 | lr | 0.867 | -0.090 | -0.258 |
| ncsu+kaggle | entity_type | Organization | E2 | lr | 0.912 | -0.046 | -0.013 |
| ncsu+kaggle | feature_group | similarity | E2 | lr | 0.925 | -0.033 | -0.080 |
| ncsu+kaggle | feature_group | structural | E2 | lr | 0.928 | -0.030 | -0.060 |
| ncsu+kaggle | entity_type | RequestedAsset | E2 | lr | 0.938 | -0.019 | 0.049 |
| ncsu+kaggle | feature_group | relation_counts | E2 | lr | 0.945 | -0.012 | 0.000 |
| ncsu+kaggle | entity_type | CallbackNumber | E2 | lr | 0.946 | -0.011 | -0.020 |
| ncsu+kaggle | entity_type | PaymentChannel | E2 | lr | 0.949 | -0.008 | -0.036 |
| ncsu+kaggle | feature_group | entity_risk | E2 | lr | 0.961 | 0.003 | 0.011 |
| ncsu+kaggle | feature_group | ontology_onehot | E2 | lr | 0.961 | 0.004 | 0.004 |
| ncsu+kaggle | feature_group | community | E2 | lr | 0.968 | 0.010 | 0.017 |
| ncsu+kaggle | entity_type | ClaimedIdentity | E2 | lr | 0.969 | 0.011 | -0.137 |
| ncsu+kaggle | feature_group | structural | E2 | xgb | 0.816 | -0.027 | 0.000 |
| ncsu+kaggle | entity_type | RequestedAsset | E2 | xgb | 0.822 | -0.020 | -0.537 |
| ncsu+kaggle | feature_group | similarity | E2 | xgb | 0.833 | -0.010 | -0.032 |
| ncsu+kaggle | entity_type | Organization | E2 | xgb | 0.840 | -0.003 | -0.137 |
| ncsu+kaggle | feature_group | entity_risk | E2 | xgb | 0.842 | -0.001 | 0.000 |
| ncsu+kaggle | feature_group | relation_counts | E2 | xgb | 0.842 | 0.000 | 0.000 |
| ncsu+kaggle | entity_type | CallbackNumber | E2 | xgb | 0.847 | 0.004 | -0.013 |
| ncsu+kaggle | feature_group | community | E2 | xgb | 0.851 | 0.008 | 0.000 |
| ncsu+kaggle | feature_group | ontology_onehot | E2 | xgb | 0.860 | 0.018 | 0.000 |
| ncsu+kaggle | entity_type | PressureTactic | E2 | xgb | 0.869 | 0.026 | -0.212 |
| ncsu+kaggle | entity_type | PaymentChannel | E2 | xgb | 0.872 | 0.030 | -0.007 |
| ncsu+kaggle | entity_type | ClaimedIdentity | E2 | xgb | 0.933 | 0.091 | -0.108 |
| ncsu+kaggle | entity_type | PressureTactic | E2G | lr | 0.867 | -0.092 | -0.257 |
| ncsu+kaggle | entity_type | Organization | E2G | lr | 0.913 | -0.045 | -0.011 |
| ncsu+kaggle | feature_group | similarity | E2G | lr | 0.926 | -0.032 | -0.080 |
| ncsu+kaggle | feature_group | structural | E2G | lr | 0.930 | -0.028 | -0.058 |
| ncsu+kaggle | entity_type | RequestedAsset | E2G | lr | 0.938 | -0.020 | 0.052 |
| ncsu+kaggle | feature_group | relation_counts | E2G | lr | 0.945 | -0.013 | 0.001 |
| ncsu+kaggle | entity_type | CallbackNumber | E2G | lr | 0.947 | -0.012 | -0.022 |
| ncsu+kaggle | entity_type | PaymentChannel | E2G | lr | 0.951 | -0.008 | -0.035 |
| ncsu+kaggle | feature_group | entity_risk | E2G | lr | 0.961 | 0.003 | 0.013 |
| ncsu+kaggle | feature_group | ontology_onehot | E2G | lr | 0.962 | 0.004 | 0.006 |
| ncsu+kaggle | entity_type | ClaimedIdentity | E2G | lr | 0.968 | 0.009 | -0.135 |
| ncsu+kaggle | feature_group | community | E2G | lr | 0.969 | 0.010 | 0.018 |
| ncsu+kaggle | entity_type | PressureTactic | E2G | xgb | 0.818 | 0.002 | -0.160 |
| ncsu+kaggle | entity_type | Organization | E2G | xgb | 0.830 | 0.015 | -0.137 |
| ncsu+kaggle | feature_group | structural | E2G | xgb | 0.842 | 0.026 | 0.000 |
| ncsu+kaggle | feature_group | entity_risk | E2G | xgb | 0.843 | 0.028 | 0.000 |
| ncsu+kaggle | feature_group | relation_counts | E2G | xgb | 0.845 | 0.029 | 0.000 |
| ncsu+kaggle | entity_type | CallbackNumber | E2G | xgb | 0.848 | 0.033 | -0.013 |
| ncsu+kaggle | feature_group | ontology_onehot | E2G | xgb | 0.850 | 0.035 | 0.000 |
| ncsu+kaggle | feature_group | similarity | E2G | xgb | 0.855 | 0.040 | -0.066 |
| ncsu+kaggle | feature_group | community | E2G | xgb | 0.859 | 0.043 | 0.000 |
| ncsu+kaggle | entity_type | RequestedAsset | E2G | xgb | 0.860 | 0.045 | -0.537 |
| ncsu+kaggle | entity_type | PaymentChannel | E2G | xgb | 0.894 | 0.078 | -0.007 |
| ncsu+kaggle | entity_type | ClaimedIdentity | E2G | xgb | 0.956 | 0.140 | -0.080 |
