# TDA6323 Part 2: Filter-Based Feature Selection Experiment

This project compares three filter-based feature selection algorithms on the
Gisette dataset:

1. mRMR
2. ReliefF
3. CFS-Greedy

The analysis is focused on filter methods because their runtime can be explained
more directly using dataset shape and algorithm steps. Unlike wrapper
metaheuristics, these algorithms do not repeatedly train a classifier during
feature selection.

## Main Workflow

Run the experiment from:

```text
time_complexity_analysis.ipynb
```

The old `main.py` script has been retired and only points to the notebook.

## Dataset

The experiment uses the local Gisette ARFF file:

```text
dataset/gisette.arff
```

The local ARFF contains the labeled part of Gisette: 7000 rows and 5000
features. The original challenge also included an unlabeled test set, but this
project uses labeled rows only.

## Input Sizes

The notebook creates nested stratified subsamples:

```text
0.1, 0.2, 0.3, ..., 1.0
```

All 5000 features are kept before feature selection. Input size is reported as:

```text
rows * features
```

The notebook also reports nonzero count and density because Gisette is stored in
sparse format but has very high density.

## Algorithms

The filter algorithms are implemented under `algorithms/`. Each algorithm
accepts the training data and labels, then returns a selected feature mask plus
metadata.

- `mrmr.py`: correlation-based minimum redundancy maximum relevance.
- `relieff.py`: nearest-hit / nearest-miss ReliefF feature weighting.
- `cfs_greedy.py`: greedy correlation-based feature subset search.

mRMR and ReliefF select the top 10% of Gisette features. CFS-Greedy stops when
the subset merit no longer improves or when it reaches the same top-10% cap.

## Evaluation

For each subsample size:

1. Split the subsample into stratified train/test sets.
2. Run each filter algorithm on the training split only.
3. Train a Linear SVM using the selected feature subset.
4. Evaluate accuracy on the held-out test split.

The notebook also trains one full-feature baseline Linear SVM on the full
Gisette train/test split.

## Outputs

Notebook outputs are saved under:

```text
results/filter_analysis/
```

Expected outputs include:

- `gisette_subsample_summary.csv`
- `filter_feature_selection_results.csv`
- `baseline_result.csv`
- execution-time vs input-size graph
- selected-feature-count vs input-size graph
- best-accuracy comparison bar chart

## Dependencies

Install the required packages:

```powershell
pip install -r requirements.txt
```
