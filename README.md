# FairRepair

**Purpose:**
ML algorithsm are important and ought to satisfy fairness criteria. This project changes the internal structure of existing classifiers in a way that makes them more fair.

**How it works:**
Given a sensitive attribute (e.g., gender) the technique assumes whitebox access to the classifier for each of the attribute values; currently it does this by training a decision tree or a random forest, and accesses the internal parameters of the tree(s). It constructs hypercubes and changes the evaluation of the classifier within these hypercubes. It uses the underlying input data to evaluate the fairness of the resulting hypercube addition/change to the classifier until the fairness criteria is satisfied.

**Repo structure:**

- `patch.py` contains the core logic
- `hcube.py` contains the class and methods for hypercubes
- `*_patch.py` contains code to exercise this logic on several different datasets.
- `data/` folder contains a few datasets
- `scripts/` folder contains the bash scripts for the experiments
- `results/` folder contains the experimental results

**Dependencies:** Some of the below packages have Python version dependency. The code was tested with Python 3.5.2 and 3.6.5. 

- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/) We suggest the users install NumPy with ```pip install numpy==1.19.3```.
- [pandas](https://pandas.pydata.org/)
- [Z3](https://github.com/Z3Prover/z3)
- [PuLP](https://pypi.org/project/PuLP/)

**To run the patches:**

- `python german_patch.py`
- `python adult_patch.py`
- `python ufrgs_patch.py`
