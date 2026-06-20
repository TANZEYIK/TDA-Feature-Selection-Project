# TDA6323 Part 2: Metaheuristic Feature Selection Experiment

This folder contains the Python implementation for Part 2 of the TDA6323 project.

## Topic

A Comparative Study of Metaheuristic Algorithms for Feature Selection

## Algorithms Implemented

The experiment compares four binary metaheuristic feature selection algorithms:

1. Binary Genetic Algorithm (BGA)
2. Binary Particle Swarm Optimization (BPSO)
3. Binary Grey Wolf Optimizer (BGWO)
4. Binary Whale Optimization Algorithm (BWOA)

Each algorithm searches for a binary feature subset:

- `1` means the feature is selected.
- `0` means the feature is excluded.

The selected feature subset is evaluated using a K-Nearest Neighbors classifier.
KNN is not the main algorithm being compared. It is only the evaluation model
used to judge whether a selected feature subset gives good classification
performance.

## Project Structure

```text
part2_feature_selection/
|-- main.py
|-- requirements.txt
|-- README.md
|-- algorithms/
|   |-- bga.py
|   |-- bpso.py
|   |-- bgwo.py
|   `-- bwoa.py
|-- utils/
|   |-- data_loader.py
|   |-- fitness.py
|   |-- helpers.py
|   `-- plotting.py
|-- dataset/
|   |-- arcene.arff
|   |-- gisette.arff
|   |-- Dexter.arff
|   |-- Dorothea.arff
|   `-- madelon.arff
`-- results/
    |-- execution_time_results.csv
    |-- figure_1_execution_time_vs_input_size.png
    |-- figure_2_test_accuracy_vs_input_size.png
    |-- figure_3_selected_features_vs_input_size.png
    `-- figure_4_convergence_curve.png
```

## Datasets

The code uses the five NIPS 2003 feature selection challenge datasets. The files
must be downloaded manually and placed in the `dataset/` folder:

```text
dataset/
|-- arcene.arff
|-- gisette.arff
|-- Dexter.arff
|-- Dorothea.arff
`-- madelon.arff
```

The code does not download datasets automatically. This keeps the experiment
reproducible when the project is run on another computer.

## Input Sizes

The experiment uses different feature-pool sizes to construct the execution
time graph.

Default input sizes:

```text
arcene:   100, 200, 300, 400, 500
gisette:  100, 200, 300, 400, 500
dexter:   100, 200, 300, 400, 500
dorothea: 100, 200, 300, 400, 500
madelon:  100, 200, 300, 400, 500
```

For each input size, the code selects a deterministic random subset from the
full feature pool. This is better than always taking the first N features,
because it reduces the risk that the early columns are unusually easy or
unrepresentative.

## Data Splitting

The code uses three sets:

```text
train set       -> train the KNN model
validation set  -> calculate fitness during algorithm search
test set        -> report final accuracy only after the best subset is selected
```

This avoids data leakage. The test set is not used during the search process.
The scaler is also fitted on the training set only, then applied to the
validation and test sets.

## Fitness Function

The fitness function is minimized:

```text
fitness = 0.90 * classification_error + 0.10 * selected_feature_ratio
```

This means a better solution should:

- produce higher validation accuracy
- select fewer features

The final test accuracy is reported separately after the algorithm finishes.

## How to Run

Install the required packages if needed:

```powershell
pip install -r requirements.txt
```

Run the default experiment:

```powershell
python main.py
```

Recommended final experiment:

```powershell
python main.py --population-size 10 --iterations 10 --output-dir results_final
```

Run a quick check before the final experiment:

```powershell
python main.py --datasets madelon --input-sizes 100 --population-size 3 --iterations 2 --output-dir results_quick_check
```

Custom input sizes can also be provided:

```powershell
python main.py --datasets arcene gisette dexter dorothea madelon --input-sizes 100 200 300 400 500 --population-size 10 --iterations 10
```

## Output Files

The output folder contains:

```text
execution_time_results.csv
figure_1_execution_time_vs_input_size.png
figure_2_test_accuracy_vs_input_size.png
figure_3_selected_features_vs_input_size.png
figure_4_convergence_curve.png
```

When more than one dataset is used, the graph files are prefixed with the
dataset name, for example:

```text
arcene_figure_1_execution_time_vs_input_size.png
madelon_figure_1_execution_time_vs_input_size.png
```

The CSV file contains:

- `dataset`
- `algorithm`
- `input_features`
- `runtime_seconds`
- `best_fitness`
- `validation_accuracy`
- `test_accuracy`
- `selected_count`
- `selected_features`
- `convergence_history`

## Report Usage

Use the generated results for the Experimental Analysis section:

- Figure 1: Execution time vs input size
- Figure 2: Final test accuracy vs input size
- Figure 3: Selected feature count vs input size
- Figure 4: Convergence curve

The most important required graph is execution time vs input size. The accuracy,
selected-feature-count, and convergence graphs are added to make the result
analysis clearer.

## Notes

The implementations are written for coursework analysis. They are intentionally
kept readable so the algorithm flow can be explained line by line in the report
or presentation.
