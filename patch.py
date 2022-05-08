"""
Methods for patching unfair decision tree and random forest classifiers.
Usage:
    1. import * from patch
    2. use the patch function, e.g., patch(evalu, ...)
"""

import copy
import math
import time
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd
from z3 import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hcube import *


############################################################
# Decision model related
############################################################


def read_data(ds, cols, random_seed):
    # Read data from the given directory.
    dataset = pd.read_csv("./data/"+ds, index_col=None, sep=",")
    # Remove dup lines, add a "frequency" column and shuffle df
    df = dataset.groupby(dataset.columns.tolist()).size().reset_index().rename(columns={0:"frequency"}).sample(frac=1, random_state=random_seed)
    # Create dummies for categorical attrs
    noclass = df.drop(["Class", "frequency"], axis=1)
    X, y = pd.get_dummies(noclass, columns = cols), df["Class"]
    df_encoded = pd.get_dummies(df, columns = cols)

    # # apply SelectKBest class to extract top 10 best features
    # bestfeatures = SelectKBest(score_func=chi2, k=10)
    # fit = bestfeatures.fit(X, y)
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(X.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ["Specs", "Score"]  #naming the dataframe columns
    # print(featureScores.nlargest(20, "Score"))  #print 20 best features
    return len(dataset), len(df), df_encoded, X, y


def train_decision_tree(X, y, evalu):
    # Build the tree using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=evalu.get_seed())
    classifier = DecisionTreeClassifier(random_state=evalu.get_seed()).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # evalu.record_eval("accuracy-before", accuracy_score(y_test, y_pred))
    # evalu.record_eval("precision-before", precision_score(y_test, y_pred))
    # evalu.record_eval("recall-before", recall_score(y_test, y_pred))
    # evalu.record_eval("f1-score-before", f1_score(y_test, y_pred))
    return classifier, y_pred


def train_random_forest(X, y, evalu, forest_size):
    # Build random forest using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=evalu.get_seed())
    classifier = RandomForestClassifier(n_estimators=forest_size, random_state=evalu.get_seed()).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # evalu.record_eval("accuracy-before", accuracy_score(y_test, y_pred))
    # evalu.record_eval("precision-before", precision_score(y_test, y_pred))
    # evalu.record_eval("recall-before", recall_score(y_test, y_pred))
    # evalu.record_eval("f1-score-before", f1_score(y_test, y_pred))
    return classifier, y_pred


def forest_to_tree_list(forest):
    rf = []
    for idx, est in enumerate(forest.estimators_):
        rf.append(est)
    return rf 


############################################################
# Computation related
############################################################


def marginal(evalu, cols):
    """
    Compute the classification error rate for different forest size, 
    to determine the best forest size to use in the experiments.
    """
    forest_size_lst = []
    accuracy_lst = []

    # From 10 to 200, increase forest size by 10 each time
    for i in range(1, 21):
        forest_size_lst.append(i*10)
        acc = 1 - inner_marginal(evalu, cols, i*10)
        print("Forest size:", i*10, "- Accuracy:", acc)
        accuracy_lst.append(acc)

    # From 200 onwards, increase forest size by 50 each time
    for i in range(1, 17):
        forest_size_lst.append(200 + i*50)
        acc = 1 - inner_marginal(evalu, cols, 200 + i*50)
        print("Forest size:", 200 + i*50, "- Accuracy:", acc)
        accuracy_lst.append(acc)
        
    print("Forest size:\n", forest_size_lst)
    print("Accuracy:\n", accuracy_lst)
    return forest_size_lst, accuracy_lst


def inner_marginal(evalu, cols, forest_size):
    """
    Inner functionality for marginal.
    """
    # Define and gather constants
    ds = evalu.get_dataset()
    len_ds, df, X, y = read_data(ds, cols)

    # Train random forest
    classifier, y_pred = train_random_forest(X, y, evalu.get_seed(), forest_size)

    # Record original outcomes
    outcomes = [classifier.predict(X.iloc[[i]]) for i in range(len(X))]

    # Count misclassified data points
    count = 0
    for i in range(len(X)):
        if outcomes[i] != df.iloc[i]["Class"]:
            count += 1

    return count/len_ds


def accuracy_calculator(lst1, lst2, cls1, cls2, data_list, evalu, has_repaired):
    """
    This function computes the true/false positives/negatives given the predicted outcome and the actual outcome.
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(len(lst1)):
        if lst1[i] == cls2:
            if lst2[i] == cls2:
                true_pos += data_list[i]["frequency"]
            if lst2[i] == cls1:
                false_pos += data_list[i]["frequency"]
        if lst1[i] == cls1:
            if lst2[i] == cls2:
                false_neg += data_list[i]["frequency"]
            if lst2[i] == cls1:
                true_neg += data_list[i]["frequency"]
    
    accuracy = (true_neg + true_pos)/(true_pos + true_neg + false_pos + false_neg)
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1_score = 2*precision*recall/(precision + recall)

    if has_repaired:
        evalu.record_eval("after-accuracy", accuracy)
        evalu.record_eval("after-precision", precision)
        evalu.record_eval("after-recall", recall)
        evalu.record_eval("after-f1-score", f1_score)
    else:
        evalu.record_eval("before-accuracy", accuracy)
        evalu.record_eval("before-precision", precision)
        evalu.record_eval("before-recall", recall)
        evalu.record_eval("before-f1-score", f1_score)
    return true_pos, false_pos, true_neg, false_neg


def median(K):
    """
    scikit-learn random forest uses a different majority vote
    """
    return math.ceil(K/2)+1


def param_calculator(hcubes_list, hcube_set_indices, data_list):
    """
    This function calculates passing rates, path probabilities and proportions.
    """
    proportions = []
    passingNums = []
    passingRates = []
    sens_points = []
    passing_sens_points = []
    sens_size = len(hcube_set_indices)
    
    for i in range(len(hcubes_list)):
        sens_points.append([])
        passing_sens_points.append([])
        label = hcubes_list[i].get_value()
        for j in range(sens_size):
            sens_points[i].append(0)
            passing_sens_points[i].append(0)
        pts_list = hcubes_list[i].get_points()
        for pt in pts_list:
            sens_points[i][point_to_sens(data_list[pt], hcube_set_indices)] += data_list[pt]["frequency"]
            if label == 1:
                passing_sens_points[i][point_to_sens(data_list[pt], hcube_set_indices)] += data_list[pt]["frequency"]

    for j in range(sens_size):
        proportions.append(int(sum([sens_points[i][j] for i in range(len(hcubes_list))])))
        passingNums.append(int(sum([passing_sens_points[i][j] for i in range(len(hcubes_list))])))
        passingRates.append(passingNums[j]/proportions[j])
    
    return proportions, passingRates, passingNums, sens_points, passing_sens_points


def rate_change_calculator(passingRates, proportions, passingNums, c):
    """ 
    This function takes in passing rates (as an array), 
    proportions (as an array), and a fairness threshold (real, 0 < c < 1), 
    and outputs the minimal possible change in passing rates, 
    to achieve fairness requirement.
    """
    M = len(passingRates)
    opt = Optimize()
    (a, b) = c
    size = max(20, int(sum(proportions)/100))

    def abs_z3(x):
        return If(x >= 0, x, -x)

    # xs is the list of x_i"s. we change r_i"s to x_i"s
    xs = [Int("X_{0}".format(i)) for i in range(len(passingRates))]
    cost = Int("cost")

    # fairness requirement
    for i in range(M):
        for j in range(M):
            if proportions[i] >= size and proportions[j] >= size and i != j:
                opt.add(xs[i]*proportions[j]*b >= a*xs[j]*proportions[i])
    
    # Optimisation goal
    opt.add(cost == Sum([abs_z3(xs[i] - passingNums[i]) for i in range(M)]))
    opt.minimize(cost)

    if opt.check() == sat:
        m = opt.model()

    x_soln = [m.eval(xs[i]).as_long() for i in range(M)]
    y_soln = [abs(x_soln[i] - passingNums[i]) for i in range(M)]
    minChange = sum(y_soln)
    return minChange, y_soln, x_soln


def min_max_calculator(optPassRates, optPassNums, proportions, passingNums, size):
    """
    This function computes an estimate for the min/max passing rates in the repaired model.
    """
    max_ = max([optPassRates[i] for i in range(len(optPassRates)) if proportions[i] > size])
    min_ = min([optPassRates[i] for i in range(len(optPassRates)) if proportions[i] > size])
    lst = []
    for i in range(len(proportions)):
        if abs(passingNums[i] - optPassNums[i]) == 0:
            lst.append((passingNums[i], 0))
        elif passingNums[i] > optPassNums[i]:
            lst.append((passingNums[i], optPassNums[i] - 1))
        else:
            lst.append((passingNums[i], optPassNums[i] + 1))
    return min_, max_, lst


############################################################
# Tree to HCube conversion
############################################################


def tree_to_code(tree, feature_names):
    """
    Return the code corresponding to the decision tree classifier in the "tree" var as a string.
    Based on the stackoverflow post:
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    
    More info, here:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth, t_str):
        indent = "  " * depth * 2
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            t_str += "{}if {} <= {}:".format(indent, name, threshold) + "\n"
            t_str = recurse(tree_.children_left[node], depth + 1, t_str) 

            t_str += "{}else:  # if {} > {}".format(indent, name, threshold) + "\n"
            t_str = recurse(tree_.children_right[node], depth + 1, t_str)
        else:
            li = list(tree_.value[node])
            t_str += "{}return {}".format(indent, np.argmax(li)) + "\n"
        return t_str

    tree_str = "def tree({}):".format(", ".join(feature_names)) + "\n"
    return recurse(0, 1, tree_str)


def tree_to_hcubes(tree, feature_name, data_list):
    """
    Returns a map of hypercube-id (i.e., path-id) to hypercube, where
    each hypercube is an instance of the HCube class.
    Note: the hypercube id is "1" followed by a sequence of 0s and 1s.
    A 0 indicates a left branch, a 1 indicates a right branch in the
    tree. The result is that a hypercube id is unique to a path in the tree.
    """
    tree_ = tree.tree_

    class ComputeHCubes:
        def __init__(self):
            self.hcubes = {}
            self.curr_constraints = {}
        
        def get_hcubes(self):
            self.recurse(0)
            return self.hcubes

        def add_constraint(self, constraint):
            name = constraint.get_name()
            if self.curr_constraints.get(name) == None:
                self.curr_constraints[name] = [constraint]
            else:
                self.curr_constraints[name].append(constraint)

        def rm_constraint(self, constraint):
            name = constraint.get_name()
            self.curr_constraints[name].remove(constraint)
        
        def recurse(self, node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Non-terminal node with two branches: if and else, with constraints on one feature
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Handle the left branch, which is the upper bound constraint:
                # feature <= threshold
                constraint = Constraint(name, True, threshold)
                self.add_constraint(constraint)
                # self.hid = self.hid + "0"
                ret = self.recurse(tree_.children_left[node])
                self.rm_constraint(constraint)
                # self.hid = self.hid[:len(self.hid) - 1]
                
                # Handle the right branch, which is the lower bound constraint:
                # feature > threshold
                constraint = Constraint(name, False, threshold)
                self.add_constraint(constraint)
                # self.hid = self.hid + "1"
                ret = self.recurse(tree_.children_right[node])
                self.rm_constraint(constraint)
                # self.hid = self.hid[:len(self.hid) - 1]
            else:
                # Terminal node: record path, and classification decision (hcube_value)
                li = list(tree_.value[node])
                hcube_value = np.argmax(li)
                # hcube = HCube(copy.deepcopy(self.curr_constraints), hcube_value, self.hid)
                hcube = HCube(constraints=copy.deepcopy(self.curr_constraints), val=hcube_value, hid=str(node), data_list=data_list)
                # self.hcubes[self.hid] = hcube
                self.hcubes[node] = hcube
                return
            return

    computeHCubes = ComputeHCubes()
    hcubes = computeHCubes.get_hcubes()
    return hcubes


def point_to_hcube_id(tree, feature_name, point):
    """
    Returns a hypercube id that the point belongs to in the decision tree.
    Point is represented as a map of feature names to values.
    """
    tree_ = tree.tree_

    # TODO: convert this into a loop that checks if the feature exists in tree features, 
    # and if it does not exist, then stop/exit/return error.

    # TODO: rename feature_name to distinguish with feature_names input var
    # feature_name = [
    #     feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    #     for i in tree_.feature
    # ]

    recursive = False
    if not recursive:
        # Non-recursive implementation of the recursive code below:
        node = 0
        while tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # if not name in point:
            #     raise RuntimeError("No value for feature "" + name + "" in point: " + str(point))
            val = point[name]
            if val <= threshold:
                node = tree_.children_left[node]
            else:
                node = tree_.children_right[node]
        return str(node)

    class ComputeHId:
        def get_id(self):
            # self.hid = "1"
            return self.recurse(0)
            
        def recurse(self, node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Non-terminal node with two branches: if and else, with constraints on one feature
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # if not name in point:
                #     raise RuntimeError("No value for feature "" + name + "" in point: " + str(point))

                val = point[name]
                if val <= threshold:
                    # Handle the left branch, which is the upper bound constraint:
                    # feature <= threshold
                    # self.hid = self.hid + "0"
                    return self.recurse(tree_.children_left[node])
                else:
                    # Handle the right branch, which is the lower bound constraint:
                    # feature > threshold
                    # self.hid = self.hid + "1"
                    return self.recurse(tree_.children_right[node])
            else:
                # Terminal node: return the hcube id
                return str(node)
                # return self.hid

    computeHId = ComputeHId()
    return computeHId.get_id()


def assign_data_points(hcubes, dataset, tree, fnames, multicore, hids=None, pt_hid_map=None):
    """Assigns points in the dataset to their respective hypercubes, which
    are themselves spread across different HCubeSets, represented by
    the hsets map.
    
    TREE:
    Use, default pt_sens_groups = None, and pt_hid_map = None when
    using with a decision tree.
    FOREST:
    Use, non-default values when using with a random forest.
    """

    if multicore:
        cpus = 4
        pool = mp.Pool(cpus)
        results = []
        M = len(dataset)

        # Divide calls to point_to_hid functions into subtasks
        # Point index modulo cpus indicates the task index
        # e.g., let cpus = 4, tasks[0] contains point = 0, 4, 8, ...
        # tasks[1] contains point = 1, 5, 9, ...

        task_size = math.ceil(M/cpus)
        tasks = []
        for i in range(cpus-1):
            tasks.append((i*task_size, (i+1)*task_size))
        tasks.append(((cpus-1)*task_size, M))

        for i in range(cpus):
            result = pool.apply_async(innerAssignDataPoints, args=(i, tasks, tree, fnames, dataset))
            results.append(result)
        pool.close()
        pool.join()

        phids = list(itertools.chain(*[results[i].get() for i in range(cpus)]))

    for i in range(len(dataset)):
        if multicore:
            phid = int(phids[i])
        else:
            point = dataset[i]
            phid = int(point_to_hcube_id(tree, fnames, point))
        hcubes[phid].add_point(dataset[i])
        if pt_hid_map != None:
            pt_hid_map[i].append(phid)
            # pt_hid_map[i].append(hids.index(phid))
        # for kx, hset in hcubes.items():
        #     fl = hset.add_point(dataset[i], phid)
        #     if fl:
        #         assert(not already_added)
        #         already_added = True
        #         ##################
        #         # FOREST ONLY
        #         if pt_sens_groups != None:
        #             pt_sens_groups[i] = kx
        #             pt_hid_map[i].append(phid)
        #         #################
        #     added_point = added_point or fl
        # if not added_point:
        #     print(phid)
        #     assert(added_point)
    return


def innerAssignDataPoints(i, tasks, tree, fnames, dataset):
    hids = []
    start, end = tasks[i]
    for i in range(start, end):
        hids.append(point_to_hcube_id(tree, fnames, dataset[i]))
    return hids


def clear_constraints(hcubes):
    for hid, hc in hcubes.items():
        hcubes[hid].rm_all_constraints()
    return hcubes


############################################################
# Hypercube sets and sensitive attributes
############################################################


def sens_group_to_str(sens_group):
    sens_str = ""
    for i in range(len(sens_group)):
        sens_str += sens_group[i] + " X "
    sens_str = sens_str[:-3]
    return sens_str


def point_to_sens(pt, hcube_set_indices):
    for i in range(len(hcube_set_indices)):
        flag = True
        for sens in hcube_set_indices[i]:
            if pt[sens] == 0:
                flag = False
        if flag:
            return i
    return None


def flip_without_refine(evalu, min_, max_, lst_, minchanges, c, hcubes_list, sens_points, passingNums, proportions, hcube_set_indices, len_ds, alpha, actual_outcomes, cls1, cls2, data_list, timeout, foundit):
    # First try MinMax query
    p1 = mp.Process(target=retVsSolverMinMax, args=(min_, max_, lst_, hcubes_list, sens_points, passingNums, proportions, hcube_set_indices, len_ds, alpha, evalu, actual_outcomes, cls1, cls2, data_list, foundit))
    p1.start()
    # Wait for ? seconds or until process finishes
    p1.join(timeout)
    # If thread is still active
    if p1.is_alive():
        print("running... let's kill it...")
        # Terminate - may not work if process is stuck for good
        p1.terminate()
        p1.join()

    if foundit.is_set():
        evalu.record_eval("rounds-of-refinement", 0)
    else:
        # Try Ind query
        p2 = mp.Process(target=retVsSolverInd, args=(hcubes_list, hcube_set_indices, len_ds, sens_points, proportions, c, alpha, minchanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit))
        p2.start()
        # Wait for ? seconds or until process finishes
        p2.join(timeout)
        # If thread is still active
        if p2.is_alive():
            print("running... let's kill it...")
            # Terminate - may not work if process is stuck for good
            p2.terminate()
            p2.join()

        if foundit.is_set():
            evalu.record_eval("rounds-of-refinement", 0)
        else:
            # Try complete query
            p3 = mp.Process(target=retVsSolverAll, args=(hcubes_list, hcube_set_indices, len_ds, sens_points, proportions, c, alpha, minchanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit))
            p3.start()
            # Wait for ? seconds or until process finishes
            p3.join(timeout)
            # If thread is still active
            if p3.is_alive():
                print("running... let's kill it...")
                # Terminate - may not work if process is stuck for good
                p3.terminate()
                p3.join()

            if foundit.is_set():
                evalu.record_eval("rounds-of-refinement", 0)
    return None


def refine_procedure(evalu, refine_heuristics, hcubes, hcube_set_indices, data_list, min_, max_, lst_, len_ds, alpha, actual_outcomes, cls1, cls2, c, minchanges, timeout):
    refine_start = time.time()
    refine_rounds = 0
    len_refine = len(refine_heuristics)

    for attr, dcrt in refine_heuristics:
        refine_rounds += 1
        print("Round of refinement:", refine_rounds)
        if dcrt:
            # For refinement wrt a discrete attribute
            hcubes = refine_dcrt_hsets(hcubes, attr, data_list)
        else:
            # For refinement wrt a continuous attribute
            hcubes = refine_cont_hsets(hcubes, attr, data_list)
        
        # hids = list(hcubes.keys())
        hcubes_list = list(hcubes.values()) 
        proportions, passingRates, passingNums, sens_points, passing_sens_points = param_calculator(hcubes_list, hcube_set_indices, data_list)
        
        m1 = mp.Manager()
        foundit = m1.Event()
        p1 = mp.Process(target=retVsSolverMinMax, args=(min_, max_, lst_, hcubes_list, sens_points, passingNums, proportions, hcube_set_indices, len_ds, alpha, evalu, actual_outcomes, cls1, cls2, data_list, foundit, ))
        p1.start()
        # Wait for ? seconds or until process finishes
        p1.join(timeout)
        # If thread is still active
        if p1.is_alive():
            print("running... let's kill it...")
            # Terminate - may not work if process is stuck for good
            p1.terminate()
            # OR Kill - will work for sure, no chance for process to finish nicely however
            # p.kill()
            p1.join()

        if foundit.is_set():
            evalu.record_eval("rounds-of-refinement", refine_rounds)
            break

        m2 = mp.Manager()
        foundit = m2.Event()
        p2 = mp.Process(target=retVsSolverInd, args=(hcubes_list, hcube_set_indices, len_ds, sens_points, proportions, c, alpha, minchanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit))
        p2.start()
        # Wait for ? seconds or until process finishes
        p2.join(timeout)
        # If thread is still active
        if p2.is_alive():
            print("running... let's kill it...")
            # Terminate - may not work if process is stuck for good
            p2.terminate()
            # OR Kill - will work for sure, no chance for process to finish nicely however
            # p.kill()
            p2.join()

        if foundit.is_set():
            evalu.record_eval("rounds-of-refinement", refine_rounds)
            break

        m3 = mp.Manager()
        foundit = m3.Event()
        p3 = mp.Process(target=retVsSolverAll, args=(hcubes_list, hcube_set_indices, len_ds, sens_points, proportions, c, alpha, minchanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit))
        p3.start()
        # Wait for ? seconds or until process finishes
        p3.join(timeout)
        # If thread is still active
        if p3.is_alive():
            print("running... let's kill it...")
            # Terminate - may not work if process is stuck for good
            p3.terminate()
            # OR Kill - will work for sure, no chance for process to finish nicely however
            # p.kill()
            p3.join()

        if foundit.is_set():
            evalu.record_eval("rounds-of-refinement", refine_rounds)
            break

    evalu.record_eval("refine_time", time.time() - refine_start)


def refine_dcrt_hsets(hcubes, attr, data_list):   
    hids = list(hcubes.keys())
    for i in hids:
        for sens in attr:
            hcube = HCube(constraints=copy.deepcopy(hcubes[i].get_constraints()), val=hcubes[i].get_value(), hid=str(i)+"U["+sens+"]", pts=copy.deepcopy(hcubes[i].get_points()), desc=copy.deepcopy(hcubes[i].get_desc()), data_list=data_list)
            hcube.refine_one_hot(sens, data_list)
            hcube.add_desc(sens)
            hcubes[str(i)+"U["+sens+"]"] = hcube
        hcubes.pop(i, None)
    return hcubes      


def refine_cont_hsets(hcubes, sens, data_list):
    hids = list(hcubes.keys())
    for i in hids:
        rank = [data_list[pt][sens] for pt in hcubes[i].get_points()]
        if not len(rank) == 0:
            med = np.median(rank)
        else:
            med = 0
        hcube1 = HCube(constraints=copy.deepcopy(hcubes[i].get_constraints()), val=hcubes[i].get_value(), hid=str(i)+"U["+sens+"]", pts=copy.deepcopy(hcubes[i].get_points()), desc=copy.deepcopy(hcubes[i].get_desc()), data_list=data_list)
        hcube2 = copy.deepcopy(hcubes[i])
        hcube1.refine_cont_one_hot(sens, med, data_list)
        hcube1.add_desc(sens+">")
        hcubes[str(i)+"U["+sens+"]"] = hcube1
        hcubes[str(i)+"L["+sens+"]"] = hcube2.refine_cont_one_hot(sens, med, data_list)
        hcubes[str(i)+"L["+sens+"]"].add_desc(sens+"<")
        hcubes.pop(i, None)
    return hcubes


def pre_process_tree(t, rf, features, data_list, num_hcubes_before, hsets, hids, pt_hid_map, hcube_time, assign_time):
    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in rf[t].tree_.feature
    ]
    len_df = len(data_list)

    # Convert trees to hcubes.
    hcube_start = time.time()
    hcubes = tree_to_hcubes(rf[t], feature_name, data_list)
    hcube_time[t] = time.time() - hcube_start
    hsets[t] = hcubes
    num_hcubes_before[t] = len(hcubes)
    hids[t] = list(hcubes.keys())

    # Assign data points to hcubes.
    assign_start = time.time()
    assign_data_points(hsets[t], data_list, rf[t], feature_name, True, hids[t], pt_hid_map)
    assign_time[t] = time.time() - assign_start
    clear_constraints(hsets[t])
    return None


def intersectHsets(hsets, hcube_set_indices, pt_hid_map, dataList, forest_size, outcomes, class2):
    """
    This function takes in the random forest, and produces the intersected hcubes.
    """
    intersectedHsets = []
    dl = {i for i in range(len(dataList))}
    while len(dl) > 0:
        i = dl.pop()    
        hcubes = [hsets[j][pt_hid_map[i][j]] for j in range(forest_size)]
        count, hcube = intersectHcubes(hcubes, forest_size, dataList, outcomes, class2)
        triple = (hcube, pt_hid_map[i], count)
        intersectedHsets.append(triple)
        dl = {j for j in dl if not j in hcube.get_points()}
    return intersectedHsets


def intersectHcubes(hcubes, forest_size, dataList, outcomes, class2):
    """
    This function takes as input a list of hypercubes, and produces the intersection of them, but with all constraints removed.
    """
    if len(hcubes) == 0:
        return None
    ptList = intersectPoints(hcubes)
    count = sum([hcube.get_value() for hcube in hcubes])
    value = 0
    for i in ptList:
        break
    if outcomes[i] == class2:
        value = 1
    newHcube = HCube(constraints=None, val=value, hid=None, pts=ptList, desc=None, data_list=dataList)
    return count, newHcube


def intersectPoints(hcubes):
    """
    Given a list of hypercubes, this function computes their common datapoints.
    """
    pts = [set() for i in range(len(hcubes))]
    for i in range(len(hcubes)):
        for pt in hcubes[i].get_points():
            pts[i].add(pt)
    pts_indices = set.intersection(*pts)
    return pts_indices


############################################################
# SMT Solving functions
############################################################


def retVsSolverMinMax(min_, max_, lst_, hcubes, sens_points, passingNums, proportions, hcube_set_indices, size_of_dataset, alpha, evalu, actual_outcomes, cls1, cls2, data_list, foundit=None):
    """
    This function takes in the above parameters, and outputs the return values of all paths after necessary flippings without refining any 
    hypercubes
    
    An testing example:
    maxSMTSolving([[0.2, 0.3], [0.1, 0.4]], [[1, 0], [1, 0]], 0.06, 0.8, 1.2) -> UNSAT
    """
    
    # Initialisation
    M = len(hcube_set_indices)
    N = len(hcubes)
    opt = Solver()
    # Groups with size less than 20 will not be considered.
    size = max(20, int(size_of_dataset/100))

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hcubes)):
        if hcubes[i].get_value() == 1:
            retVs.append(True)
        else:
            retVs.append(False)
    for i in range(M):
        pathProbsRow = []
        for j in range(N):
            pathProbsRow.append(int(sens_points[j][i]))
        pathProbs.append(pathProbsRow)

    # Initialisation of the variable list
    X = [Bool("x_{0}".format(i)) for i in range(len(retVs))]

    # Fairness requirementss
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for j in range(N):
            if not pathProbs[i][j] == 0:
                m[i] += pathProbs[i][j] * If(X[j], 1, 0)
    
    for i in range(M):
        x, y = lst_[i]
        if y > 0:
            if x > y and proportions[i] > size:
                opt.add(m[i] <= y + 1)
                mc = math.ceil(abs(x - y) * alpha)
                opt.add(m[i] >= max(y - mc, math.ceil(min_ * proportions[i])) - 1)
            if x < y and proportions[i] > size:
                opt.add(m[i] >= y - 1)
                mc = math.ceil(abs(x - y) * alpha)
                opt.add(m[i] <= min(y + mc, math.floor(max_ * proportions[i])) + 1)
        else:
            if proportions[i] > size:
                for j in range(len(pathProbs[i])):
                    if pathProbs[i][j] != 0:
                        opt.add(X[j] == retVs[j])

    # Semantic difference requirements    
    sd = [0 for i in range(M)]
    for i in range(M):
        if proportions[i] > size:
            x, y = lst_[i]
            for j in range(len(retVs)):
                if retVs[j] == False and pathProbs[i][j] != 0:
                    sd[i] += If(X[j], 1, 0) * pathProbs[i][j]
                if retVs[j] == True and pathProbs[i][j] != 0:
                    sd[i] += If(X[j], 0, 1) * pathProbs[i][j]
            if y > 0:
                opt.add(sd[i] <= math.ceil(abs(x - y) * alpha))

    # Check SAT
    result = opt.check()
    if result == sat:
        evalu.record_eval("query-mode", "minmax")
        if foundit != None:
            foundit.set()
        m = opt.model()
        newRetVs = [True for i in range(N)]
        for i in range(N):
            if not sum([pathProbs[j][i] for j in range(M)]) == 0:
                if m.eval(X[i]) == True:
                    newRetVs[i] = bool(m.eval(X[i]))
                elif m.eval(X[i]) == False:
                    newRetVs[i] = bool(m.eval(X[i]))
            else:
                newRetVs[i] = retVs[i]

        # Evaluate on the dataset
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(M):
            for j in range(len(retVs)):
                if not newRetVs[j] == retVs[j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[j]:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        ratios_ = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if proportions[i] > size and proportions[j] > size:
                    ratios.append(((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]), i, j))
                    ratios_.append((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]))
        evalu.record_eval("final-passing-rates", [passingNums[i]/proportions[i] for i in range(len(passingNums))])
        evalu.record_eval("final-passing-nums", passingNums)
        evalu.record_eval("final-fairness", ratios)
        evalu.record_eval("final-p-rule-score", min(ratios_))
        evalu.record_eval("data-points-changed-group", count)
        evalu.record_eval("data-points-changed-total", sum(count))
        evalu.record_eval("final-sem-diff", sum(count)/len(data_list))
        
        repaired_outcomes = [None for i in range(size_of_dataset)]
        for i in range(N):
            if newRetVs[i]:
                hcubes[i].set_value(1)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls2
            else:
                hcubes[i].set_value(0)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls1

        accuracy_calculator(repaired_outcomes, actual_outcomes, cls1, cls2, data_list, evalu, True)
        return True, count, sum(count), ratios
    elif result == unsat:
        print("mmx: unsat")
        return False, [], [], []
    else:
        raise RuntimeError("Result is undefined.")


def retVsSolverInd(hcubes, hcube_set_indices, size_of_dataset, sens_points, proportions, c, alpha, minChanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit):
    """
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2, 0.3], [0.1, 0.4]], [[1, 0], [1, 0]], 0.06, 0.8, 1.2) -> UNSAT
    """
    
    # Initialisation
    M = len(hcube_set_indices)
    N = len(hcubes)
    opt = Solver()
    # Groups with size less than 20 will not be considered.
    size = max(20, int(size_of_dataset/100))
    (a, b) = c

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hcubes)):
        if hcubes[i].get_value() == 1:
            retVs.append(True)
        else:
            retVs.append(False)
    for i in range(M):
        pathProbsRow = []
        for j in range(N):
            pathProbsRow.append(int(sens_points[j][i]))
        pathProbs.append(pathProbsRow)

    # Calculate the minimun change in each sens group
    for i in range(M):
        minChanges[i] = int(math.ceil(alpha*minChanges[i]) + 1)

    # Initialisation of the variable list
    X = [Bool("x_{0}".format(i)) for i in range(len(retVs))]

    # Fairness requirementss
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for j in range(N):
            if not pathProbs[i][j] == 0:
                m[i] += pathProbs[i][j] * If(X[j], 1, 0) 

    A = 0
    B = 0
    if not type(a) == int:
        A = a.item()
    else:
        A = a
    if not type(a) == int:
        B = b.item()
    else:
        B = b

    for i in range(M):
        for j in range(M):
            # Only groups with large enough population are considered
            if proportions[j] > size and proportions[i] > size:
                opt.add(m[i]*int(proportions[j])*B > A*m[j]*int(proportions[i]))
    
    # Semantic difference requirement (number of people being affected)
    for i in range(M):
        opt.add(PbLe([(Xor(X[j], retVs[j]), pathProbs[i][j]) for j in range(N)], minChanges[i]))

    # Check SAT
    result = opt.check()
    if result == sat:
        evalu.record_eval("query-mode", "ind")
        if foundit != None:
            foundit.set()
        m = opt.model()
        newRetVs = [True for i in range(N)]
        for i in range(N):
            if not sum([pathProbs[j][i] for j in range(M)]) == 0:
                if m.eval(X[i]) == True:
                    newRetVs[i] = bool(m.eval(X[i]))
                elif m.eval(X[i]) == False:
                    newRetVs[i] = bool(m.eval(X[i]))
            else:
                newRetVs[i] = retVs[i]

        # Evaluate on the dataset
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(M):
            for j in range(len(retVs)):
                if not newRetVs[j] == retVs[j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[j]:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        ratios_ = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if proportions[i] > size and proportions[j] > size:
                    ratios.append(((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]), i, j))
                    ratios_.append((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]))
        evalu.record_eval("final-passing-rates", [passingNums[i]/proportions[i] for i in range(len(passingNums))])
        evalu.record_eval("final-passing-nums", passingNums)
        evalu.record_eval("final-fairness", ratios)
        evalu.record_eval("final-p-rule-score", min(ratios_))
        evalu.record_eval("data-points-changed-group", count)
        evalu.record_eval("data-points-changed-total", sum(count))
        evalu.record_eval("final-sem-diff", sum(count)/int(4*len(data_list)/5))
        
        repaired_outcomes = [None for i in range(size_of_dataset)]
        for i in range(N):
            if newRetVs[i]:
                hcubes[i].set_value(1)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls2
            else:
                hcubes[i].set_value(0)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls1

        accuracy_calculator(repaired_outcomes, actual_outcomes, cls1, cls2, data_list, evalu, True)
        return True, count, sum(count), ratios
    elif result == unsat:
        print("ind: unsat")
        return False, [], [], []
    else:
        raise RuntimeError("Result is undefined.")


def retVsSolverAll(hcubes, hcube_set_indices, size_of_dataset, sens_points, proportions, c, alpha, minChanges, evalu, actual_outcomes, cls1, cls2, data_list, foundit):
    """
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2, 0.3], [0.1, 0.4]], [[1, 0], [1, 0]], 0.06, 0.8, 1.2) -> UNSAT
    """
    
    # Initialisation
    M = len(hcube_set_indices)
    N = len(hcubes)
    opt = Solver()
    # Groups with size less than 20 will not be considered.
    size = max(20, int(size_of_dataset/100))
    (a, b) = c

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hcubes)):
        if hcubes[i].get_value() == 1:
            retVs.append(True)
        else:
            retVs.append(False)
    for i in range(M):
        pathProbsRow = []
        for j in range(N):
            pathProbsRow.append(int(sens_points[j][i]))
        pathProbs.append(pathProbsRow)

    # Calculate the minimun change in each sens group
    for i in range(M):
        minChanges[i] = math.ceil(alpha * minChanges[i])

    # Initialisation of the variable list
    X = [Bool("x_{0}".format(i)) for i in range(len(retVs))]

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for j in range(N):
            if not pathProbs[i][j] == 0:
                m[i] += pathProbs[i][j] * If(X[j], 1, 0) 

    A = 0
    B = 0
    if not type(a) == int:
        A = a.item()
    else:
        A = a
    if not type(a) == int:
        B = b.item()
    else:
        B = b

    for i in range(M):
        for j in range(M):
            # Only groups with large enough population are considered
            if proportions[j] > size and proportions[i] > size:
                opt.add(m[i]*proportions[j]*B > A*m[j]*proportions[i])
    
    # Semantic difference requirement (number of people being affected)
    opt.add(PbLe([(Xor(X[i], retVs[i]), int(sum([pathProbs[j][i] for j in range(M)]))) for i in range(N)], int(sum(minChanges))))
    
    # Check SAT
    result = opt.check()
    if result == sat:
        evalu.record_eval("query-mode", "all")
        if foundit != None:
            foundit.set()
        m = opt.model()
        newRetVs = [True for i in range(N)]
        for i in range(N):
            if not sum([pathProbs[j][i] for j in range(M)]) == 0:
                if m.eval(X[i]) == True:
                    newRetVs[i] = bool(m.eval(X[i]))
                elif m.eval(X[i]) == False:
                    newRetVs[i] = bool(m.eval(X[i]))
            else:
                newRetVs[i] = retVs[i]

        # Evaluate on the dataset
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(M):
            for j in range(len(retVs)):
                if not newRetVs[j] == retVs[j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[j]:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        ratios_ = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if proportions[i] > size and proportions[j] > size:
                    ratios.append(((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]), i, j))
                    ratios_.append((passingNums[i]*proportions[j])/(passingNums[j]*proportions[i]))
        evalu.record_eval("final-passing-rates", [passingNums[i]/proportions[i] for i in range(len(passingNums))])
        evalu.record_eval("final-passing-nums", passingNums)
        evalu.record_eval("final-fairness", ratios)
        evalu.record_eval("final-p-rule-score", min(ratios_))
        evalu.record_eval("data-points-changed-group", count)
        evalu.record_eval("data-points-changed-total", sum(count))
        evalu.record_eval("final-sem-diff", sum(count)/int(4*len(data_list)/5))
        
        repaired_outcomes = [None for i in range(size_of_dataset)]
        for i in range(N):
            if newRetVs[i]:
                hcubes[i].set_value(1)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls2
            else:
                hcubes[i].set_value(0)
                for pt in hcubes[i].get_points():
                    repaired_outcomes[pt] = cls1

        accuracy_calculator(repaired_outcomes, actual_outcomes, cls1, cls2, data_list, evalu, True)
        return True, count, sum(count), ratios
    elif result == unsat:
        print("all: unsat")
        return False, [], [], []
    else:
        raise RuntimeError("Result is undefined.")


def forestRetVsSolverAll(minchanges, hsets, pt_hids_map, pt_sens_map, data_list, size_of_dataset, proportions, hcube_set_indices, predicted_outcomes, c, alpha, evalu):
    """ Main solver function for patching random forest. """
    # Initialisation
    M = len(hcube_set_indices)
    opt = Solver()
    med = median(len(hsets))
    # Groups with size less than 20 will not be considered.
    size = max(20, int(size_of_dataset/100))
    (a, b) = c

    # Collect the hids of the hcubes in the hsets
    hc_hids_map = []
    for hset in hsets:
        hc_hids_map.append(list(hset.keys()))

    # Record the initial return values
    retVs = []
    for i in range(len(hsets)):
        retVsRow = []
        for hid in hc_hids_map[i]:
            if hsets[i][hid].get_value() == 1:
                retVsRow.append(True)
            else:
                retVsRow.append(False)
        retVs.append(retVsRow)

    # Calculate the minimum change in each sens group
    for i in range(M):
        minchanges[i] = math.ceil(alpha*minchanges[i])

    # Initialisation of the variable list
    X = [{hc_hids_map[i][j] : Bool("X_{}_{}".format(i, j)) for j in range(len(hc_hids_map[i]))} for i in range(len(hc_hids_map))]

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(len(data_list)):
        var = [X[j][pt_hids_map[i][j]] for j in range(len(hsets))]
        m[pt_sens_map[i]] += If(AtLeast(*var,med),1,0) * data_list[i]['frequency']
    
    A = 0
    B = 0
    if not type(a) == int:
        A = a.item()
    else:
        A = a
    if not type(a) == int:
        B = b.item()
    else:
        B = b

    for i in range(M):
        for j in range(M):
            # Only groups with large enough population are considered
            if proportions[j] > size and proportions[i] > size:
                opt.add(m[i]*proportions[j]*B > A*m[j]*proportions[i])
    
    # Semantic difference requirement (number of people being affected)
    sd = []
    for i in range(len(data_list)):
        var = [X[j][pt_hids_map[i][j]] for j in range(len(hsets))]
        outcome = False
        if predicted_outcomes[i] == 2:
            outcome = True
        sd.append(Xor(AtLeast(*var,med),outcome))
    opt.add(PbLe([(sd[i], data_list[i]['frequency']) for i in range(len(data_list))], int(sum(minchanges))))

    result = opt.check()
    # print(result)
    return None


############################################################
# Main logic
############################################################


def patch(evalu, cols, attr_map, refine_heuristics):
    """
    Main workhorse method for patching decision trees and random forests. Takes the following arguments:
        evalu: an instance of the eutil.EvalUtil class
        cols: categorical attributes
        refine_heuristics: pre-defined refinement order
        attr_map: possible valuations of attributes
    """
    if evalu.get_forest() != None:
        # Patch a random forest
        return patch_forest(evalu, cols, attr_map, refine_heuristics, evalu.get_forest_size())

    # Patch a decision tree
    return patch_tree(evalu, cols, attr_map, refine_heuristics)

    # # Compute accuracy vs. forest size
    # return marginal(evalu, cols)


def patch_tree(evalu, cols, attr_map, refine_heuristics):
    """
    Handles patching of a decision tree.
    """
    # Gather constants
    start_time = time.time()
    fairness_thresh = evalu.get_fairness_thresh()
    alpha = evalu.get_alpha()
    timeout = evalu.get_timeout()    
    ds = evalu.get_dataset()
    evalu.record_eval("param-fairness", fairness_thresh)
    evalu.record_eval("param-alpha", alpha)
    evalu.record_eval("param-timeout", timeout)
    evalu.record_eval("data-file", ds)
    # Separate fairness thresh into 2 integers for easier computation.
    c = (int(fairness_thresh*100), 100)
    # Classification labels are 1 (positive) and 2 (negative).
    cls1, cls2 = 1, 2

    # Pre-process data
    len_ds, len_df, df, X, y = read_data(ds, cols, evalu.get_seed())
    evalu.record_eval("param-num-inputs", len_ds)
    features = list(X.columns)
    # Only consider groups with size no less than 20 or 1% of the population.
    size = max(20, int(len_ds/100))

    # Train a decision tree
    classifier, y_pred = train_decision_tree(X, y, evalu)
    # # Record the trained decision tree
    # dataset_num = evalu.get_file_name()
    # f = open("./%s.tree" %dataset_num, "w+")
    # # f.write(tree_to_code(classifier, features))
    # evalu.record_eval("input-tree", "./%s.tree" %dataset_num)

    # Record outcomes and compute accuracy before repair
    df['index'] = [i for i in range(len_df)]
    data_list = df.to_dict('records')
    predicted_outcomes = list(classifier.predict(X))
    actual_outcomes = list(y)
    accuracy_calculator(predicted_outcomes, actual_outcomes, cls1, cls2, data_list, evalu, False)

    # Record sensitive attributes
    sensAttrs_str = eval(evalu.get_sensitive_attrs())
    sensAttrs = []
    for attr_str in sensAttrs_str:
        sensAttrs.append(attr_map[attr_str])
    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in classifier.tree_.feature
    ]
    
    # This is the cross product of all sensitive attrs. 
    hcube_set_indices = []
    for element in itertools.product(*sensAttrs):
        hcube_set_indices.append(list(element))
    sens_size = len(hcube_set_indices)
    evalu.record_eval("param-sens-attrs", sensAttrs_str)
    evalu.record_eval("param-sens-values", sensAttrs)
    evalu.record_eval("param-num-groups", sens_size)
    evalu.record_eval("param-sens-groups", hcube_set_indices)

    # Start repair procedure
    repair_start = time.time()
    # Get the path hypercubes with respect to sensitive groups, and assign 
    # data points to these hypercubes. hcubes1 is used for repairing and 
    # hcubes2 for accuracy evaluation.
    hcubes = tree_to_hcubes(classifier, feature_name, data_list)
    evalu.record_eval("num-hcubes-before", len(hcubes))

    # Assign data points to the hcubes
    assign_start = time.time()
    # True for multicore, False for single core
    assign_data_points(hcubes, data_list, classifier, feature_name, True)
    evalu.record_eval("time-assign-points", time.time() - assign_start)
    # Constraints of the hypercubes are no longer used. Fix the orders of 
    # the hcubes for refinement latter
    hcubes_list = list(clear_constraints(hcubes).values())

    # Calculate the proportions and passing rates of each sens group
    proportions, passingRates, passingNums, sens_points, passing_sens_points = param_calculator(hcubes_list, hcube_set_indices, data_list)
    evalu.record_eval("size-of-sens-groups", proportions)
    evalu.record_eval("init-passing-rates", passingRates)
    evalu.record_eval("init-passing-nums", passingNums)

    # Linear optimisation of the minimal theoretical semantic distance 
    minChange, minChanges, optPassNums = rate_change_calculator(passingRates, proportions, passingNums, c)
    # Compute the max and min for the changes in passing rates
    optPassRates = [optPassNums[i]/proportions[i] for i in range(len(proportions))]
    min_, max_, lst_ = min_max_calculator(optPassRates, optPassNums, proportions, passingNums, size)
    evalu.record_eval("lb-min-change", minChange)
    evalu.record_eval("lb-min-change-list", minChanges)
    evalu.record_eval("lb-optimal-passing-nums", optPassNums)
    evalu.record_eval("lb-optimal-passing-rates", optPassRates)

    # Flipping the hypercubes with timeout
    flip_start = time.time()
    m = mp.Manager()
    foundit = m.Event()
    flip_without_refine(evalu, min_, max_, lst_, minChanges, c, hcubes_list, sens_points, passingNums, proportions, hcube_set_indices, len_ds, alpha, actual_outcomes, cls1, cls2, data_list, timeout, foundit)
    flip_time = time.time() - flip_start
    evalu.record_eval("time-flip-smt", flip_time)

    if foundit.is_set():
        evalu.record_eval("rounds-of-refinement", 0)
    else:
        # If UNSAT, proceed to refine the hypercubes
        refine_procedure(evalu, refine_heuristics, hcubes, hcube_set_indices, data_list, min_, max_, lst_, len_ds, alpha, actual_outcomes, cls1, cls2, c, minChanges, timeout)

    # Record total time used
    evalu.record_eval("time-repair", time.time() - repair_start)
    evalu.record_eval("time-total", time.time() - start_time)
    
    return None


def patch_forest(evalu, cols, attr_map, refine_heuristics, forest_size):
    """
    Handles patching of a Random forest
    """
    # Gather constants
    start_time = time.time()
    fairness_thresh = evalu.get_fairness_thresh()
    alpha = evalu.get_alpha()
    timeout = evalu.get_timeout()    
    ds = evalu.get_dataset()
    evalu.record_eval("param-fairness", fairness_thresh)
    evalu.record_eval("param-alpha", alpha)
    evalu.record_eval("param-timeout", timeout)
    evalu.record_eval("data-file", ds)
    # Separate fairness thresh into 2 integers for easier computation.
    c = (int(fairness_thresh*100), 100)
    # Classification labels are 1 (positive) and 2 (negative).
    cls1, cls2 = 1, 2

    # Pre-process data
    len_ds, len_df, df, X, y = read_data(ds, cols, evalu.get_seed())
    evalu.record_eval("param-num-inputs", len_ds)
    features = list(X.columns)
    # Only consider groups with size no less than 20 or 1% of the population.
    size = max(20, int(len_ds/100))

    # Train random forest and collect decision trees
    classifier, y_pred = train_random_forest(X, y, evalu, forest_size)
    rf = forest_to_tree_list(classifier)

    # Record outcomes and compute accuracy before repair.
    df['index'] = [i for i in range(len_df)]
    data_list = df.to_dict('records')
    predicted_outcomes = list(classifier.predict(X))
    actual_outcomes = list(y)
    accuracy_calculator(predicted_outcomes, actual_outcomes, cls1, cls2, data_list, evalu, False)

    # Record sensitive groups
    sensAttrs_str = eval(evalu.get_sensitive_attrs())
    sensAttrs = []
    for attr_str in sensAttrs_str:
        sensAttrs.append(attr_map[attr_str])    
    
    # This is the cross product of all sensitive attrs. 
    hcube_set_indices = []
    for element in itertools.product(*sensAttrs):
        hcube_set_indices.append(list(element))
    sens_size = len(hcube_set_indices)
    evalu.record_eval("param-sens-attrs", sensAttrs_str)
    evalu.record_eval("param-sens-values", sensAttrs)
    evalu.record_eval("param-num-groups", sens_size)
    evalu.record_eval("param-sens-groups", hcube_set_indices)

    # Combined step of generating hcubes, dividing hcubes and assigning data 
    # points. Clear constrains after each individual hset is constructed to 
    # avoid memory explosion.
    repair_start = time.time()
    num_hcubes_before = [0 for i in range(forest_size)]
    hsets= [None for i in range(forest_size)]
    hids = [None for i in range(forest_size)]
    pt_hids_map = [[] for i in range(len(X))]
    pt_sens_map = [point_to_sens(data_list[i], hcube_set_indices) for i in range(len_df)]
    hcube_time = [0 for i in range(len(X))]
    assign_time = [0 for i in range(len(X))]  
    for t in range(forest_size):
        pre_process_tree(t, rf, features, data_list, num_hcubes_before, hsets, hids, pt_hids_map, hcube_time, assign_time)
    evalu.record_eval("num-hcubes-before", num_hcubes_before)
    evalu.record_eval("time-get-hcubes", sum(hcube_time))
    evalu.record_eval("time-assign-points", sum(assign_time))
    
    # Calculate the proportions and passing rates of each sens group
    proportions = [0 for i in hcube_set_indices]
    for i in pt_sens_map:
        proportions[i] += 1
    evalu.record_eval("size-of-sens-groups", proportions)
    passingNums = [0 for i in hcube_set_indices]
    for i in range(len_df):
        if predicted_outcomes[i] == cls2:
            passingNums[pt_sens_map[i]] += 1
    evalu.record_eval("init-passing-nums", passingNums)
    passingRates = [passingNums[i]/proportions[i] for i in range(len(proportions))]
    evalu.record_eval("init-passing-rates", passingRates)

    # Linear optimisation of the minimal theoretical semantic distance 
    minChange, minChanges, optPassNums = rate_change_calculator(passingRates, proportions, passingNums, c)
    # Compute the max and min for the changes in passing rates
    optPassRates = [optPassNums[i]/proportions[i] for i in range(len(proportions))]
    min_, max_, lst_ = min_max_calculator(optPassRates, optPassNums, proportions, passingNums, size)
    evalu.record_eval("lb-min-change", minChange)
    evalu.record_eval("lb-min-change-list", minChanges)
    evalu.record_eval("lb-optimal-passing-nums", optPassNums)
    evalu.record_eval("lb-optimal-passing-rates", optPassRates)

    # Flipping the hypercubes with timeout
    flip_start = time.time()
    m = mp.Manager()
    foundit = m.Event()
    forestRetVsSolverAll(minChanges, hsets, pt_hids_map, pt_sens_map, data_list, len_ds, proportions, hcube_set_indices, predicted_outcomes, c, alpha, evalu)
    
    if foundit.is_set():
        evalu.record_eval("rounds-of-refinement", 0)
        evalu.record_eval("time-repair", time.time() - repair_start)
        evalu.record_eval("time-total", time.time() - start_time)
        return None

    # Obtain the intersections
    intersect_start = time.time()
    intersectedHsets = intersectHsets(hsets, hcube_set_indices, pt_hids_map, data_list, forest_size, predicted_outcomes, cls2)
    hcubes_list = {}
    for i in range(len(intersectedHsets)):
        hc, hids, ct = intersectedHsets[i]
        hcubes_list[i] = hc
        hcubes_list[i].set_hid(str(i))
    
    # Calculate the proportions and passing rates of each sens group
    proportions_, passingRates_, passingNums_, sens_points, passing_sens_points = param_calculator(hcubes_list, hcube_set_indices, data_list)

    # Flipping the hypercubes with timeout
    flip_start = time.time()
    m = mp.Manager()
    foundit = m.Event()
    flip_without_refine(evalu, min_, max_, lst_, minChanges, c, hcubes_list, sens_points, passingNums, proportions, hcube_set_indices, len_ds, alpha, actual_outcomes, cls1, cls2, data_list, timeout, foundit)
    flip_time = time.time() - flip_start
    evalu.record_eval("time-flip-smt", flip_time)

    if foundit.is_set():
        evalu.record_eval("rounds-of-refinement", 0)
    else:
        # If UNSAT, proceed to refine the hypercubes
        refine_procedure(evalu, refine_heuristics, hcubes_list, hcube_set_indices, data_list, min_, max_, lst_, len_ds, alpha, actual_outcomes, cls1, cls2, c, minChanges, timeout)

    # Record total time used
    evalu.record_eval("time-repair", time.time() - repair_start)
    evalu.record_eval("time-total", time.time() - start_time)
    
    return None










