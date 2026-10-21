# Exploratory data analysis

## Data sets

| dataset | n | n_scam | scam_rate | mean_words | median_words | mean_turns |
|---|---|---|---|---|---|---|
| bothbosu | 1598 | 799 | 0.500 | 229.960 | 238.000 | 13.685 |
| kaggle | 793 | 393 | 0.496 | 46.272 | 36.000 | 1.000 |
| ncsu | 713 | 713 | 1.000 | 62.049 | 53.000 | 1.000 |


## Splits

| dataset | split | n |
|---|---|---|
| bothbosu | test | 240 |
| bothbosu | train | 1118 |
| bothbosu | val | 240 |
| kaggle | external | 793 |
| ncsu | external | 713 |


## Vocabulary overlap (hard cross-dataset transfer = low Jaccard / high OOV)

| dataset_a | dataset_b | vocab_a | vocab_b | jaccard | oov_rate_b_given_a |
|---|---|---|---|---|---|
| bothbosu | kaggle | 2618 | 1993 | 0.317 | 0.131 |
| bothbosu | ncsu | 2618 | 989 | 0.237 | 0.079 |
| kaggle | ncsu | 1993 | 989 | 0.248 | 0.118 |
