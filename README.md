# TDA6323 Part 2: Metaheuristic Feature Selection Experiment

This folder contains the Python implementation for Part 2 of the TDA6323 project.

## Project Structure

```text
part2_feature_selection/
├── main.py
├── requirements.txt
├── README.md
├── algorithms/
│   ├── __init__.py
│   ├── bga.py
│   ├── bpso.py
│   ├── bgwo.py
│   └── bwoa.py
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── fitness.py
│   ├── helpers.py
│   └── plotting.py
└── results/
    ├── execution_time_results.csv
    ├── figure_1_execution_time_vs_input_size.png
    ├── figure_2_accuracy_vs_input_size.png
    └── figure_3_selected_features_vs_input_size.png
```

## Topic

A Comparative Study of Metaheuristic Algorithms for Feature Selection

## Algorithms Implemented

1. Binary Genetic Algorithm (BGA)
2. Binary Particle Swarm Optimization (BPSO)
3. Binary Grey Wolf Optimizer (BGWO)
4. Binary Whale Optimization Algorithm (BWOA)

Each algorithm searches for a binary feature subset:

- `1` means the feature is selected.
- `0` means the feature is excluded.

The selected feature subset is evaluated using a K-Nearest Neighbors classifier.

## Dataset

The experiment uses the Breast Cancer Wisconsin dataset from `scikit-learn`.

Different input sizes are created by using different numbers of features:

```text
5, 10, 15, 20, 25, 30
```

This allows the report to plot execution time against input size.

For fairness, all algorithms use the same train/test split for the same input
size. The algorithm search seed is separated from the dataset split seed, so
the algorithms can still search differently while being evaluated on the same
data split.

## Fitness Function

The fitness function is minimized:

```text
fitness = 0.90 * classification_error + 0.10 * selected_feature_ratio
```

This means a better solution should:

- produce higher classification accuracy
- select fewer features

## How to Run

From this folder:

```powershell
python main.py
```

Optional custom settings:

```powershell
python main.py --population-size 30 --iterations 40
```

If the required Python packages are not installed:

```powershell
pip install -r requirements.txt
```

## Output Files

After running the program, the `results` folder will contain:

```text
execution_time_results.csv
figure_1_execution_time_vs_input_size.png
figure_2_accuracy_vs_input_size.png
figure_3_selected_features_vs_input_size.png
```

The CSV file contains:

- algorithm name
- input size
- execution time
- best fitness
- accuracy
- number of selected features
- selected feature names

The generated figures can be used in the Experimental Analysis section of the final report:

- Figure 1: Execution time vs input size
- Figure 2: Accuracy vs input size
- Figure 3: Selected feature count vs input size

## File Descriptions

`main.py` controls the full experiment. It loads the dataset, calls the four algorithms, records the runtime, saves the CSV file, and generates the graphs.

`algorithms/bga.py` contains the Binary Genetic Algorithm implementation.

`algorithms/bpso.py` contains the Binary Particle Swarm Optimization implementation.

`algorithms/bgwo.py` contains the Binary Grey Wolf Optimizer implementation.

`algorithms/bwoa.py` contains the Binary Whale Optimization Algorithm implementation.

`utils/dataset.py` loads and scales the Breast Cancer Wisconsin dataset.

`utils/fitness.py` defines the fitness function and KNN accuracy evaluation.

`utils/helpers.py` contains helper functions such as sigmoid conversion and empty-subset handling.

`utils/plotting.py` generates the execution time, accuracy, and selected feature count graphs.

## Notes for Report Writing

The KNN classifier is not the main algorithm being compared. It is only used to evaluate how good each selected feature subset is. The algorithms being compared are BGA, BPSO, BGWO, and BWOA.

No external algorithm code was copied. The implementations are written for this coursework experiment, based on the standard algorithm designs discussed in the literature review.
