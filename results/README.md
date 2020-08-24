# FairRepair/results

The experiments are conducted on two datasets in the `data` folder. The experimental parameters contain:

- `fairness threshold`
- `alpha`
- `sensitive attribute`
- `random seed`

For adult dataset, there is an additional parameter:

- `dataset size`

For random forest, there is one more parameter:

- `forest size`

For each dataset, we ran experiments on different parameters to test various properties of our tool. In particular, the folders in this directory contains the following results.

- `gt`, German, decision tree
- `gf1`, German, random forest, fixed forest size
- `gf2`, German, random forest, varying forest size
- `at1`, adult, decision tree, fixed dataset size
- `at2`, adult, decision tree, varying dataset size
- `af1`, adult, random forest, fixed dataset size and forest size
- `af2`, adult, random forest, fixed dataset size, varying forest size
- `af3`, adult, random forest, fixed forest size, varying dataset size
