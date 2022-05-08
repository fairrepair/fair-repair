# FairRepair

## Our Purpose

Data-driven decision-making has gained prominence in automated resource allocation tasks. Many tasks that were previously performed by human (e.g., approval of loans) are now processed by ML models. However, real life training datasets may capture human biases, causing the trained models to be unfair. This project, *FairRepair* focuses on repairing such an unfair model. Our two main concerns are 1) fairness and 2) semantic difference.  

1. Fairness: we focus on group fairness in this work. Unlike fairness learning algorithms which train new models from scratch, FairRepair is able to modify the internal structures of an existing decision tree or random forest, so to meet the fairness requirements.

2. Semantic difference: this refers to the proportion of inputs receiving difference outcomes over the original and the repaired models. This is important for certain social or legislative decisions. For example, the [COMPAS](https://en.wikipedia.org/wiki/COMPAS_(software)) software predicts the recidivism rates of prisioners and uses the predictions for sentencing. The semantic difference has to be controlled under such circumstances.  

FairRepair is able to produce a repaired model satisfying both the fairness and the semantic bound requirement wrt a specific dataset. In addition, our algorithms is both sound and complete.

## The Algorithm

FairRepair repairs an unfair decision tree or a random forest. Given a sensitive attribute (e.g., gender), FairRepair assumes whitebox access to the classifier for each of the attribute values, and uses SMT solving to conduct the repair. It encodes for each decision tree path a pseudo-Boolean variable representing its label (classification), and converts the classifiers into their logical equivalents. Fairness and semantic bound requirements are encoded as SMT formulas. When necessary, FairRepair refines the decision tree paths and changes their leaf labels. Whenever the fairness and semantic bound constraints are satisfiable, our tool produces a sound repaired model as the solution.

## Installation

To get started, download or clone the repository. FairRepair runs in Python. It was tested on Python 3.5.2, 3.6.5 and 3.8.10. Some of the below packages may have Python version dependency.

- [scikit-learn](https://scikit-learn.org/stable/) for training decision models.  
- [NumPy](https://numpy.org/) for mathematical functions like `median` and `argmax`.  
- [pandas](https://pandas.pydata.org/) for pre-processing datasets.
- [Z3](https://github.com/Z3Prover/z3) for SMT solving.

These dependencies can be install by the following instruction.

```shell
pip install --user scikit-learn numpy pandas z3-solver
```

## Usage

To run the patches, call `./*_patch.py` files with Python. The following line calls FairRepair to repair a decision tree trained on the COMPAS dataset.

```shell
python compas_patch.py
```

The `./data/` folder contains three datasets, and their corresponding subsets with different sizes.

1. [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult)
2. [COMPAS Recidivism Risk Score Data and Analysis](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
3. [UFRGS Entrance Exam and GPA Data Set](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8)

The default fairness threshold is 0.8 and default semantic bound is 1.2. FairRepair offers command line parsers for its parameters. The following line calls FairRepair on the Adult dataset, with fairness threshold 0.9 and semantic bound 1.1.

```shell
python adult_patch.py -f 0.9 -a 1.1
```

To call FairRepair on a subset, use the `-i` flag. The following line uses a uniformly sampled subset of the Adult dataset, containing 10,000 data points.

```shell
python adult_patch.py -i "adult.data.10000"
```

For more command line argument, please call `-h` for details.

```shell
python adult_patch.py -h
```
