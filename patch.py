#!/usr/bin/env python
# coding: utf-8

'''
This is a library for patching decision tree classifiers to be fair.
Usage:
1. import * from patch
2. use the patch function, e.g.,:
patch(evalu)
'''

from sklearn import datasets
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import multiprocessing as mp
import threading
import itertools
import numpy as np
import pandas as pd
import pprint
import queue
import copy
import code
import sys
import time
import math

from z3 import *
from pulp import *
from hcube import *


def patch(evalu, cols, refineHeuristics, attr_map,classes):
    '''Main workhorse method for patching decision trees and random
    forests. Takes the following arguments:
    evalu : an instance of the eutil.EvalUtil class
    cols : categorical attributes
    refineHeuristics : pre-defined refinement order
    attr_map : possible valuations of attributes
    '''
    if evalu.get_forest() != None:
        # Patch a random forest:
        return patchForest(evalu, cols, refineHeuristics, attr_map, evalu.get_forest_size(),classes)
        
        # Compute accuracy vs. forest size
        # return marginal(evalu,cols,classes)

    # Patch a decision tree:
    return patchTree(evalu, cols, refineHeuristics, attr_map,classes)


def patchTree(evalu, cols, refineHeuristics, attr_map,classes):
    '''
    Handles patching of a decision tree
    '''
    # Define and gather constants.
    start_time = time.time()
    ds = evalu.get_dataset()
    len_ds,df,X,y = readData(ds, cols)
    features = list(X.columns)
    cls1,cls2 = classes
    size =  20
    # countMultipliesAlpha = 0

    # Train a decision tree:
    last_time = time.time()
    classifier, y_pred = trainDecisionTree(X,y,evalu.get_seed())
    evalu.record_eval("time-tree-construction", time.time() - last_time)

    sizeOfDataset = len_ds
    to_dict_start = time.time()
    dataList = []
    for i in range(len(df)):
        dataList.append(df.iloc[i].to_dict())
        dataList[i]['index'] = i
    evalu.record_eval("df-to-dict-time",time.time()-to_dict_start)

    # Record outcomes and compute accuracy
    classify_start = time.time()
    predicted_outcomes = [classifier.predict(X.iloc[[i]])[0] for i in range(len(df))]
    actual_outcomes = [dataList[i]['Class'] for i in range(len(df))]

    true_pos, false_pos, true_neg, false_neg = accuracyCalculator(predicted_outcomes,actual_outcomes,cls1,cls2,dataList)

    evalu.record_eval("classification-time", time.time()-classify_start)
    evalu.record_eval("true-positive-before", true_pos)
    evalu.record_eval("false-positive-before", false_pos)
    evalu.record_eval("true-negative-before", true_neg)
    evalu.record_eval("false-negative-before", false_neg)
    evalu.record_eval("precision-before",true_pos/(true_pos+false_pos))
    evalu.record_eval("recall-before",true_pos/(true_pos+false_neg))
    evalu.record_eval("accuracy-before",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))

    # default fairness_thresh = 0.8
    fairness_thresh = evalu.get_fairness_thresh()
    c = (int(fairness_thresh * 100), 100)

    # default alpha = 1.05
    alpha = evalu.get_alpha()

    # file to output EVALs
    num = evalu.get_file_name()

    evalu.record_eval("data-file", evalu.get_dataset())
    evalu.record_eval("num-input-points", len_ds)
    evalu.record_eval("fairness-thresh", fairness_thresh)
    evalu.record_eval("alpha", alpha)
    f = open("./%s.tree" %num, "w+")
    f.write(tree_to_code(classifier,features))
    evalu.record_eval("input-tree", "./%s.tree" %num)

    # Test the tree 
    # # print("Classifying 0th element: ", str(classifier.predict(X.iloc[[0]])))
    # # print("Dictionary-view of 0th element: ")
    # # print(X.iloc[0].to_dict())

    # NOTE: the eval statement turns a string like "['Sex', 'ForeignWorker']"
    # into a list of two strings: ['Sex', 'ForeignWorker']
    sensAttrs_str = eval(evalu.get_sensitive_attrs())
    sensAttrs = []
    for attr_str in sensAttrs_str:
        sensAttrs.append(attr_map[attr_str])    
    evalu.record_eval("sensitive-attrs", sensAttrs_str)
    evalu.record_eval("sensitive-[Attr]-values", sensAttrs)

    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in classifier.tree_.feature
    ]

    repair_start = time.time()
    #  Get the path hypercubes with respect to sensitive groups, and assign
    # data points to these hypercubes.
    # print("Executing hcube conversion...")
    hcubes = tree_to_hcubes(classifier,feature_name,dataList)
    # print("Done.")
    evalu.record_eval("num-hcubes-before", len(hcubes))

    last_time = time.time()
    # print("Executing divideHCubes...")
    hsets = divideHCubes(classifier,feature_name,sensAttrs,dataList)
    # print(hsets['Sex_A91'].get_hcubes_map()['10000000000000'])
    # print("Done.")    
    evalu.record_eval("time-divide-hcubes", time.time()-last_time)

    last_time = time.time()
    # print("Executing assignDataPoints...")
    assignDataPoints(evalu,hsets, dataList, classifier, feature_name, False)
    # print("Done.")
    evalu.record_eval("time-assign-points", time.time()-last_time)

    # constraints of the hypercubes are no longer used   
    clearConstraints(hsets)

    # This is the cross product of all sensitive attrs. Because the dict type needs 
    # strings as indices, we here explicitely write the sensitive group attributes as lists.
    hCubeSetIndices = []
    for element in itertools.product(*sensAttrs):
        hCubeSetIndices.append(list(element))
    # print("Computed HSets:")
    # p# print.p# print(hCubeSetIndices)
    evalu.record_eval("init-num-hcubesets", len(hCubeSetIndices))
    evalu.record_eval("sens-groups", hCubeSetIndices)

    # Record ordering of hids
    hids = getHids(hsets,hCubeSetIndices)      

    # Record number of hcubes after split
    numHcubes,numHcubesAll = countNumHcubes(hsets)
    evalu.record_eval("num-hcubes-split-nonzero", numHcubes)
    evalu.record_eval("num-hcubes-split-all", numHcubesAll)

    # Calculate the proportions and passing rates of each sens group
    proportions, passingRates, passingNums = paramCalculatorInt(hsets,hids,sizeOfDataset,hCubeSetIndices)
    evalu.record_eval("size-of-sens-groups", proportions)
    evalu.record_eval("init-passing-rates", passingRates)
    evalu.record_eval("init-passing-nums", passingNums)

    # Linear optimisation of the minimal theoretical semantic distance (integer based)
    minChange, minChanges, optPassNums = rateChangeCalculatorInt(passingRates, proportions, passingNums, c, sizeOfDataset)
    evalu.record_eval("theoretical-min-change", minChange)
    evalu.record_eval("theoretical-min-change-list", minChanges)
    evalu.record_eval("theoretical-optimal-passing-nums", optPassNums)

    optPassRates = [optPassNums[i]/proportions[i] for i in range(len(proportions))]
    min1,max1,lst1 = min_max_calculator(optPassRates,optPassNums,proportions,passingNums,size)
    proportions_ = [i/sizeOfDataset for i in proportions]
    minChange_, minChanges_,optPassRates = rateChangeCalculatorRatio(passingRates, proportions_, c, sizeOfDataset)
    optPassRates_ = [optPassRates[i]*proportions_[i]*sizeOfDataset for i in range(len(optPassRates))]

    min2,max2,lst2 = min_max_calculator(optPassRates,optPassRates_,proportions,passingNums,size)
    minChanges1 = minChanges
    minChanges2 = [math.ceil(minChanges_[i]*proportions[i]) for i in range(len(minChanges_))]

    # Flipping the hypercubes
    last_time = time.time()
    pool = mp.Pool(5)
    m = mp.Manager()
    foundit = m.Event()
    results = []

    p1 = pool.apply_async(retVsSolverOld,args=(hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges1,evalu,foundit,actual_outcomes,cls1,cls2,dataList))
    p2 = pool.apply_async(retVsSolverOld,args=(hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges2,evalu,foundit,actual_outcomes,cls1,cls2,dataList))
    p3 = pool.apply_async(retVsSolver,args=(min1,max1,lst1,hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
    p4 = pool.apply_async(retVsSolver,args=(min2,max2,lst2,hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
    p5 = pool.apply_async(retVsSolverFinal,args=(hsets, hids, hCubeSetIndices, sizeOfDataset, c, alpha, minChange, evalu,foundit,actual_outcomes,cls1,cls2,dataList))
    results.append(p1)
    results.append(p2)
    results.append(p3)
    results.append(p4)
    results.append(p5)

    sat = False
    while not sat:
        if foundit.is_set():
            time.sleep(5)
            pool.terminate()
            sat = True
        if all([p.ready() for p in results]):
            if foundit.is_set():
                time.sleep(5)
                pool.terminate()
                sat = True
            else:
                sat = False
                break

    flip_time = time.time()-last_time
    evalu.record_eval("time-flip-smt", flip_time)

    # The main function for refinement. If the refinemene achieves fairness
    # requirement, return and terminate. If not, keep refining using the predefined
    # heuristic 
    if sat:
        # If the fairness reqs are satisfiable without refinements
        evalu.record_eval("refinement-needed", "N")
        evalu.record_eval("post-refinement-steps-needed", "N")
        repair_time = time.time() - repair_start
        numHcubes,numHcubesAll = countNumHcubes(hsets)
        recordEval(evalu, flip_time, repair_time, start_time, ratiosCalculator(passingRates, proportions, sizeOfDataset), numHcubes, numHcubesAll, 0)
                
        return repaired_tree_to_code(classifier,features,hsets,hCubeSetIndices,refineHeuristics,0)

    # Go through the refinement procedures
    evalu.record_eval("refinement-needed", "Y")
    refine_result = refineProcedure(refineHeuristics, hids,hCubeSetIndices, hsets, c, alpha, minChanges1,minChanges2, sizeOfDataset, last_time, evalu,dataList,cls1,cls2,False,actual_outcomes,min1,max1,lst1,min2,max2,lst2)
    if refine_result != None:
        evalu.record_eval("post-refinement-steps-needed", "N")
        hsets, totalNumRef, refine_time, numHcubes = refine_result
        numHcubes,numHcubesAll = countNumHcubes(hsets)
        recordEval(evalu, refine_time + flip_time, time.time()-repair_start, start_time, ratiosCalculator(passingRates, proportions, sizeOfDataset), numHcubes, numHcubesAll, totalNumRef)
        
        return repaired_tree_to_code(classifier,features,hsets,hCubeSetIndices,refineHeuristics,totalNumRef)

    # If flipping unrefined tree doesn't work, and refinement doesn't
    # work, then we can use this procedure to flip individual points
    # (incomplete, since it is unclear what to do with individual
    # points... perhaps re-train the tree? :-) 
    #
    # evalu.record_eval("post-refinement-steps-needed", "Y")
    # newRetVs, ratios, ratios_after, N, count, sum_count = fullyRefinedSolver(hsets_to_list(hsets,hCubeSetIndices), hCubeSetIndices, c, alpha, evalu, minChanges, countMultipliesAlpha, dataList)
    # smt_time = time.time() - last_time
    # repair_time = time.time() - last_time
    # evalu.record_eval("multiplies-alpha", countMultipliesAlpha)
    # numHcubes,numHcubesAll = countNumHcubes(hsets)
    # recordEval(evalu, smt_time, repair_time, start_time, ratios, ratios_after, count, sum_count, numHcubes, numHcubesAll, totalNumRef)
    raise RuntimeError("No Repair Found!")
    # return None


def patchForest(evalu,cols,refineHeuristics,attr_map,forest_size,classes):
    '''
    Handles patching of a Random forest
    '''
    
    # Define and gather constants
    start_time = time.time()
    ds = evalu.get_dataset()
    len_ds,df,X,y = readData(ds, cols)
    cls1,cls2 = classes
    size = 20
    evalu.record_eval("forest-size", forest_size)

    # Train random forest and collect decision trees
    forest_construction_start = time.time()
    classifier, y_pred = trainRandomForest(X,y,evalu.get_seed(),forest_size)
    forest_construction_time = time.time()-forest_construction_start
    evalu.record_eval("time-forest-construction", forest_construction_time)
    features = list(X.columns)
    rf = forest_to_trees(classifier)

    # NOTE: to_dict() is necessary to convert pandas.core.series.Series
    # to a python dictionary, which allows us to more easily store points
    # Otherwise we have to deal with "ValueError: The truth value of a Series is ambiguous"
    # in HCube.remove_point when we perform self.pts.remove(point)
    to_dict_start = time.time()
    dataList = []
    for i in range(len(X)):
        dataList.append(df.iloc[i].to_dict())
        dataList[i]['index'] = i
    evalu.record_eval("df-to-dict-time",time.time()-to_dict_start)

    sizeOfDataset = 0
    for i in range(len(X)):
        sizeOfDataset += dataList[i]['frequency']

    # Collect constants, countMultipliesAlpha = 0
    fairness_thresh = evalu.get_fairness_thresh()
    c = (int(fairness_thresh * 100), 100)
    alpha = evalu.get_alpha()
    evalu.record_eval("data-file", evalu.get_dataset())
    evalu.record_eval("num-input-points", len_ds)
    evalu.record_eval("fairness-thresh", fairness_thresh)
    evalu.record_eval("alpha", alpha)

    # Record original outcomes
    classify_start = time.time()
    if type(cls1) == int:
        predicted_outcomes = [int(classifier.predict(X.iloc[[i]])[0]) for i in range(len(X))]
    if type(cls1) == str:
        predicted_outcomes = [classifier.predict(X.iloc[[i]])[0] for i in range(len(X))]
    actual_outcomes = [dataList[i]['Class'] for i in range(len(X))]
    true_pos, false_pos, true_neg, false_neg = accuracyCalculator(predicted_outcomes,actual_outcomes,cls1,cls2,dataList)

    evalu.record_eval("classification-time", time.time()-classify_start)
    evalu.record_eval("true-positive-before", true_pos)
    evalu.record_eval("false-positive-before", false_pos)
    evalu.record_eval("true-negative-before", true_neg)
    evalu.record_eval("false-negative-before", false_neg)
    evalu.record_eval("precision-before",true_pos/(true_pos+false_pos))
    evalu.record_eval("recall-before",true_pos/(true_pos+false_neg))
    evalu.record_eval("accuracy-before",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))

    # Compute sensitive groups
    sensAttrs_str = eval(evalu.get_sensitive_attrs())
    sensAttrs = []
    for attr_str in sensAttrs_str:
        sensAttrs.append(attr_map[attr_str])    
    evalu.record_eval("sensitive-attrs", sensAttrs_str)
    evalu.record_eval("sensitive-[Attr]-values", sensAttrs)

    # This is the cross product of all sensitive attrs. Because the dict type needs 
    # strings as indices, we here explicitely write the sensitive group attributes as lists.
    hCubeSetIndices = []
    for element in itertools.product(*sensAttrs):
        hCubeSetIndices.append(list(element))
    # print("Computed HSets:")
    # p# print.p# print(hCubeSetIndices)
    evalu.record_eval("init-num-hcubesets", len(hCubeSetIndices))
    evalu.record_eval("sens-groups", hCubeSetIndices)
    
    # Combined step of generating hcubes, dividing hcubes and assigning datapoints
    # Clear constrains after each individual hset is constructed,
    # to avoid memory explosion
    repair_start = time.time()
    num_hcubes_before = [0 for i in range(forest_size)]
    hsets = [None for i in range(forest_size)]
    pt_sens_groups = [None for i in range(len(X))]
    pt_hid_map = [[] for i in range(len(X))]
    hcube_time = [0 for i in range(len(X))]
    divide_time = [0 for i in range(len(X))]
    assign_time = [0 for i in range(len(X))]
    
    for t in range(forest_size):
        preProcessTree(evalu, t,rf,features,sensAttrs,dataList,num_hcubes_before,hsets,pt_sens_groups,pt_hid_map,hcube_time,divide_time,assign_time)

    evalu.record_eval("num-hcubes-before", num_hcubes_before)
    evalu.record_eval("time-get-hcubes", sum(hcube_time))
    evalu.record_eval("time-divide-hcubes", sum(divide_time))
    evalu.record_eval("time-assign-points", sum(assign_time))

    dsProportions, dsPassingNums = datasetFairness(hCubeSetIndices,dataList,pt_sens_groups,cls2)
    evalu.record_eval("dataset-passing-nums", dsPassingNums)

    # Obtain the intersections
    intersect_start = time.time()
    intersectedHsets = intersectHsets(hsets,hCubeSetIndices,pt_hid_map,pt_sens_groups,dataList,forest_size,predicted_outcomes,cls2)
    evalu.record_eval("time-intersection", time.time()-intersect_start)

    # Record the number of hcubes after split
    numHcubes = [0 for i in range(forest_size)]
    numHcubesAll = [0 for i in range(forest_size)]
    for i in range(forest_size):
        numHcubes[i],numHcubesAll[i] = countNumHcubes(hsets[i])
    evalu.record_eval("num-hcubes-split-nonzero", numHcubes)
    evalu.record_eval("num-hcubes-split-all", numHcubesAll)

    # Compute the proportions and passing rates 
    proportions = [0 for i in hCubeSetIndices]
    passingRates = [0 for i in hCubeSetIndices]
    passingNums = [0 for i in hCubeSetIndices]
    for i in range(len(hCubeSetIndices)):
        for j in range(len(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])])):
            (hcube,hids,ct_) = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j]
            proportions[i] += hcube.get_passing_rate()
            if hcube.get_value() == 1:
                passingNums[i] += hcube.get_passing_rate()
    passingRates = [passingNums[i]/proportions[i] for i in range(len(passingNums))] 
    evalu.record_eval("size-of-sens-groups", proportions)
    evalu.record_eval("init-passing-rates", passingRates)
    evalu.record_eval("init-passing-nums", passingNums)

    #  Linear optimisation of the minimal theoretical semantic distance
    minChange, minChanges, optPassNums = rateChangeCalculatorInt(passingRates, proportions, passingNums, c, sizeOfDataset)
    evalu.record_eval("theoretical-min-change", minChange)
    evalu.record_eval("theoretical-min-change-list", minChanges)
    evalu.record_eval("theoretical-optimal-passing-nums", optPassNums)    
    evalu.record_eval("refinement-needed", "Y")

    optPassRates = [optPassNums[i]/proportions[i] for i in range(len(proportions))]
    min1,max1,lst1 = min_max_calculator(optPassRates,optPassNums,proportions,passingNums,size)
    proportions_ = [i/sizeOfDataset for i in proportions]
    minChange_, minChanges_, optPassRates = rateChangeCalculatorRatio(passingRates, proportions_, c, sizeOfDataset)
    optPassRates_ = [optPassRates[i]*proportions_[i]*sizeOfDataset for i in range(len(optPassRates))]
    min2,max2,lst2 = min_max_calculator(optPassRates,optPassRates_,proportions,passingNums,size)
    minChanges1 = minChanges
    minChanges2 = [math.ceil(minChanges_[i]*proportions[i]) for i in range(len(minChanges_))]
    
    # First attemp of SMT flipping, with tightest constraints
    pool = mp.Pool(4)
    m = mp.Manager()
    foundit = m.Event()
    results = []

    p1 = pool.apply_async(retVsSolverIntersected,args=(intersectedHsets,forest_size,hCubeSetIndices, c, alpha, minChanges1, passingRates,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
    p2 = pool.apply_async(retVsSolverIntersected,args=(intersectedHsets,forest_size,hCubeSetIndices, c, alpha, minChanges2, passingRates,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
    p3 = pool.apply_async(anotherRetVsSolver,args=(min1,max1,intersectedHsets,lst1,forest_size, hCubeSetIndices, sizeOfDataset, c, alpha, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList))
    p4 = pool.apply_async(anotherRetVsSolver,args=(min2,max2,intersectedHsets,lst2,forest_size, hCubeSetIndices, sizeOfDataset, c, alpha, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList))

    results.append(p1)
    results.append(p2)
    results.append(p3)
    results.append(p4)
    
    isRepaired = False
    while not isRepaired:
        if foundit.is_set():
            time.sleep(5)
            pool.terminate()
            isRepaired = True
        if all([p.ready() for p in results]):
            if foundit.is_set():
                time.sleep(5)
                pool.terminate()
                isRepaired = True
            else:
                isRepaired = False
                break
            
    # result = forestRetVsSolver(min_,max_,lst,hsets,optPassRates,dataList,predicted_outcomes,intersectedHsets,pt_sens_groups,pt_hid_map,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges_,classes,passingRates,evalu)

    # Go through the refinement procedures
    totalNumRef = 0
    if not isRepaired:
        for attr,discrt in refineHeuristics:
            totalNumRef += 1
            for i in range(len(hsets)):
                if discrt:
                    hsets[i] = refine_dcrt_hsets(hsets[i],hCubeSetIndices,attr,dataList)
                else:
                    hsets[i] = refine_cont_hsets(hsets[i],hCubeSetIndices,attr,dataList)
                for hset in hsets[i].values():
                    hmap = hset.get_hcubes_map()
                    for hid_ in hmap.keys():
                        hcube = hmap[hid_]
                        if not hcube.get_passing_rate() == 0:
                            for pt in hcube.get_points():
                                pt_hid_map[pt][i]=hid_

            intersect_start = time.time()
            intersectedHsets = intersectHsets(hsets,hCubeSetIndices,pt_hid_map,pt_sens_groups,dataList,len(hsets),predicted_outcomes,cls2)
            evalu.record_eval("time-intersection", time.time()-intersect_start)
            
            pool = mp.Pool(4)
            m = mp.Manager()
            foundit = m.Event()
            results = []

            p1 = pool.apply_async(retVsSolverIntersected,args=(intersectedHsets,forest_size,hCubeSetIndices, c, alpha, minChanges1, passingRates,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
            p2 = pool.apply_async(retVsSolverIntersected,args=(intersectedHsets,forest_size,hCubeSetIndices, c, alpha, minChanges2, passingRates,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
            p3 = pool.apply_async(anotherRetVsSolver,args=(min1,max1,intersectedHsets,lst1,forest_size, hCubeSetIndices, sizeOfDataset, c, alpha, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList))
            p4 = pool.apply_async(anotherRetVsSolver,args=(min2,max2,intersectedHsets,lst2,forest_size, hCubeSetIndices, sizeOfDataset, c, alpha, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList))

            results.append(p1)
            results.append(p2)
            results.append(p3)
            results.append(p4)
            
            isRepaired = False
            while not isRepaired:
                if foundit.is_set():
                    time.sleep(5)
                    pool.terminate()
                    isRepaired = True
                if all([p.ready() for p in results]):
                    if foundit.is_set():
                        time.sleep(5)
                        pool.terminate()
                        isRepaired = True
                    else:
                        isRepaired = False
                        break

            # result = forestRetVsSolver(False,hsets,optPassRates,dataList,outcomes,intersectedHsets,pt_sens_groups,pt_hid_map,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges,classes, passingRates,evalu)
            
            # Record the number of hcubes after refine
            numHcubes = [0 for i in range(len(hsets))]
            numHcubesAll = [0 for i in range(len(hsets))]
            for i in range(len(hsets)):
                numHcubes[i],numHcubesAll[i] = countNumHcubes(hsets[i])
            evalu.record_eval("num-hcubes-refine-{}-nonzero".format(totalNumRef), numHcubes)
            evalu.record_eval("num-hcubes-refine-{}-all".format(totalNumRef), numHcubesAll)

            if isRepaired:
                evalu.record_eval("refinement-steps",totalNumRef)
                hsetsLists = []
                for i in range(len(hCubeSetIndices)):
                    hsetsRow = []
                    for j in range(len(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])])):
                        hsetsRow.append(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j])
                    hsetsLists.append(hsetsRow)
                break
    
    if not isRepaired:
        raise RuntimeError("No Repair Found!")

    # Record the number of hcubes after repair
    evalu.record_eval("repair-time",time.time()-repair_start)
    evalu.record_eval("init-fairness", ratiosCalculator(passingRates, proportions, sizeOfDataset))
    numHcubes = [0 for i in range(forest_size)]
    numHcubesAll = [0 for i in range(forest_size)]
    for i in range(forest_size):
        numHcubes[i],numHcubesAll[i] = countNumHcubes(hsets[i])
    evalu.record_eval("num-hcubes-final-nonzero", numHcubes)
    evalu.record_eval("num-hcubes-final-all", numHcubesAll)    
    evalu.record_eval("time-total",time.time()-start_time)
    return [repaired_tree_to_code(rf[i],features,hsets[i],hCubeSetIndices,refineHeuristics,totalNumRef) for i in range(forest_size)]


def preProcessTree(evalu, t,rf,features,sensAttrs,dataList,num_hcubes_before,hsets,pt_sens_groups,pt_hid_map,hcube_time,divide_time,assign_time):
    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in rf[t].tree_.feature
    ]

    hcube_start = time.time()
    hcubes = tree_to_hcubes(rf[t], feature_name,dataList)
    hcube_end = time.time()
    hcube_time[t] = hcube_end - hcube_start
    num_hcubes_before[t] = len(hcubes)
    divide_start = time.time()
    hsets[t] = divideHCubes(rf[t],feature_name,sensAttrs,dataList)
    divide_end = time.time()
    divide_time[t] = divide_end - divide_start
    assign_start = time.time()
    if len(dataList) >= 100000:
        assignDataPoints(evalu,hsets[t],dataList,rf[t],feature_name,True,pt_sens_groups,pt_hid_map)
    else:
        assignDataPoints(evalu,hsets[t],dataList,rf[t],feature_name,False,pt_sens_groups,pt_hid_map)
    assign_end = time.time()
    assign_time[t] = assign_end - assign_start
    clearConstraints(hsets[t])
    return None


def retVsSolver(min_,max_,lst_,hsets, hids, hCubeSetIndices, sizeOfDataset, c, alpha, evalu,actual_outcomes,cls1,cls2,foundit,dataList):
    '''
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2,0.3],[0.1,0.4]], [[1,0],[1,0]], 0.06, 0.8, 1.2) -> UNSAT
    '''
    
    # Initialisation
    M = len(hsets)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    opt = Solver()
    size = 20

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
        for j in range(len(hids[i])):
            hc = hmap[hids[i][j]]
            if hc.get_value() == 1:
                retVsRow.append(1)
            elif hc.get_value() == 0:
                retVsRow.append(0)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)

    passingNums = [0 for i in range(M)]
    for i in range(len(retVs)):
        for j in range(len(retVs[i])):
            if retVs[i][j] == True:
                passingNums[i] += pathProbs[i][j]

    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Initialisation of the variable list
    Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            for j in range(len(Y[i])):
                Y[i][j] = Int('y%sy%s' %(i,j))
                if not pathProbs[i][j] == 0:
                    opt.add(Or(Y[i][j]==0,Y[i][j]==1))

    # # Initialisation of the variable list
    # hSets = hsets_to_list(hsets,hCubeSetIndices)
    # Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    # for i in range(M):
    #     for j in range(len(retVs[i])):
    #         if not pathProbs[i][j] == 0:
    #             Y[i][j] = Bool('y%sy%s' %(i,j))
    #             for pt in hSets[i][j].get_points():
    #                 if actual_outcomes[pt] == cls2:
    #                     opt.add_soft(Y[i][j] == True)
    #                 else:
    #                     opt.add_soft(Y[i][j] == False)
    #             for k in range(pathProbs[i][j]):
    #                 opt.add(Y[i][j] == X[i][j][k]) 

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            # Count the data points in one sens group
            for k in range(len(pathProbs[i])):
                if not pathProbs[i][k] == 0:
                    m[i] += pathProbs[i][k] * Y[i][k]
    
    for i in range(M):
        x,y = lst_[i]
        if y > 0:
            if x > y and intProportions[i] > size:
                opt.add(m[i] <= y)
                mc = math.ceil(abs(x-y)*alpha)
                opt.add(m[i] >= max(y - mc,math.ceil(min_ * intProportions[i])))
            if x < y and intProportions[i] > size:
                opt.add(m[i] >= y)
                mc = math.ceil(abs(x-y)*alpha)
                opt.add(m[i] <= min(y + mc,math.floor(max_ * intProportions[i])))
        else:
            if intProportions[i] > size:
                for j in range(len(Y[i])):
                    opt.add(Y[i][j] == retVs[i][j])
    
    sd = [0 for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            x,y = lst_[i]
            for j in range(len(retVs[i])):
                if retVs[i][j] == 0:
                    sd[i] += Y[i][j]*pathProbs[i][j]
                if retVs[i][j] == 1:
                    sd[i] += (1-Y[i][j])*pathProbs[i][j]
            if y > 0:
                # opt.add(sd[i] <= math.ceil(abs(x-y)*alpha)+1)
                opt.add(sd[i] <= math.ceil(abs(x-y)*alpha))
    
    result = opt.check()
    if result == sat:
        m = opt.model()
        newRetVs = copy.deepcopy(retVs)
        for i in range(M):
            if intProportions[i] > size:
                for j in range(len(pathProbs[i])):
                    if not pathProbs[i] == 0:
                        if m.eval(Y[i][j]) == 1:
                            newRetVs[i][j] = 1
                        elif m.eval(Y[i][j]) == 0:
                            newRetVs[i][j] = 0
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(M):
            for j in range(len(retVs[i])):
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[i][j] == 1:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] > size and intProportions[j] > size:
                    ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
        if not foundit.is_set():
            foundit.set()
            repaired_outcomes = [None for i in range(sizeOfDataset)]
            for i in range(len(hCubeSetIndices)):
                hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
                for j in range(len(hids[i])):
                    if newRetVs[i][j]:
                        hmap[hids[i][j]].set_value(1)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls2
                    else:
                        hmap[hids[i][j]].set_value(0)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls1
            true_pos, false_pos, true_neg, false_neg = accuracyCalculator(repaired_outcomes,actual_outcomes,cls1,cls2,dataList)

            evalu.record_eval("final-passing-rates",[passingNums[i]/intProportions[i] for i in range(len(passingNums))])
            evalu.record_eval("final-passing-nums",passingNums)
            evalu.record_eval("final-fairness", ratios)
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("true-positive-after", true_pos)
            evalu.record_eval("false-positive-after", false_pos)
            evalu.record_eval("true-negative-after", true_neg)
            evalu.record_eval("false-negative-after", false_neg)
            evalu.record_eval("precision-after",true_pos/(true_pos+false_pos))
            evalu.record_eval("recall-after",true_pos/(true_pos+false_neg))
            evalu.record_eval("accuracy-after",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))
            # q.put((True, count, sum(count), ratios))

        return True, count, sum(count), ratios
    else:
        return False,[],[],[]


def retVsSolverOld(hsets, hids, hCubeSetIndices, sizeOfDataset, c, alpha, y_soln, evalu,foundit,actual_outcomes,cls1,cls2,dataList):
    '''
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2,0.3],[0.1,0.4]], [[1,0],[1,0]], 0.06, 0.8, 1.2) -> UNSAT
    '''
    
    # Initialisation
    M = len(hsets)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    opt = Solver()
    size = 20
    (a,b) = c

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
        for j in range(len(hids[i])):
            hc = hmap[hids[i][j]]
            if hc.get_value() == 1:
                retVsRow.append(True)
            elif hc.get_value() == 0:
                retVsRow.append(False)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)
    
    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Calculate the minimun change in each sens group
    minChanges = [0 for i in range(M)]
    for i in range(M):
        minChanges[i] = math.ceil(alpha*y_soln[i])

    # Initialisation of the variable list
    X = [[[] for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                for k in range(pathProbs[i][j]):
                    X[i][j].append(Bool('x%sx%sx%s' %(i,j,k)))
    Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                Y[i][j] = Bool('y%sy%s' %(i,j))
                for k in range(pathProbs[i][j]):
                    opt.add(Y[i][j] == X[i][j][k]) 

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for k in range(len(pathProbs[i])):
            if not pathProbs[i][k] == 0:
                m[i] += pathProbs[i][k] * If(Y[i][k],1,0)   

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
            if intProportions[j] > size and intProportions[i] > size:
                opt.add(m[i]*intProportions[j]*B > A*m[j]*intProportions[i])
    
    # Semantic difference requirement (number of people being affected)
    sd = []
    for i in range(M):
        sd_row = []
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                for k in range(pathProbs[i][j]):
                    sd_row.append(Xor(X[i][j][k],retVs[i][j]))      
        sd.append(sd_row)

    for i in range(M):
        opt.add(AtMost(*sd[i], minChanges[i]))
    
    result = opt.check()
    if result == sat:
        # print("The flipping result is: SAT.")
        m = opt.model()
        newRetVs = [[True for j in range(len(retVs[i]))] for i in range(M)]
        for i in range(M):
            for j in range(len(pathProbs[i])):
                if not pathProbs[i][j] == 0:
                    newRetVs[i][j] = m.eval(Y[i][j])
                else:
                    newRetVs[i][j] = retVs[i][j]
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(len(retVs)):
            for j in range(len(retVs[i])):
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[i][j] == True:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] > size and intProportions[j] > size:
                    ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
        
        if not foundit.is_set():
            foundit.set()
            repaired_outcomes = [None for i in range(sizeOfDataset)]
            for i in range(len(hCubeSetIndices)):
                hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
                for j in range(len(hids[i])):
                    if newRetVs[i][j]:
                        hmap[hids[i][j]].set_value(1)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls2
                    else:
                        hmap[hids[i][j]].set_value(0)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls1
            true_pos, false_pos, true_neg, false_neg = accuracyCalculator(repaired_outcomes,actual_outcomes,cls1,cls2,dataList)

            evalu.record_eval("final-passing-rates",[passingNums[i]/intProportions[i] for i in range(len(passingNums))])
            evalu.record_eval("final-passing-nums",passingNums)
            evalu.record_eval("final-fairness", ratios)
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("true-positive-after", true_pos)
            evalu.record_eval("false-positive-after", false_pos)
            evalu.record_eval("true-negative-after", true_neg)
            evalu.record_eval("false-negative-after", false_neg)
            evalu.record_eval("precision-after",true_pos/(true_pos+false_pos))
            evalu.record_eval("recall-after",true_pos/(true_pos+false_neg))
            evalu.record_eval("accuracy-after",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))
            # q.put((True, count, sum(count), ratios))
        return True, count, sum(count), ratios
    elif result == unsat:
        # print("The flipping result is: UNSAT.")
        return False,[],[],[]
    else:
        raise RuntimeError("Result is undefined.")


def retVsSolverFinal(hsets, hids, hCubeSetIndices, sizeOfDataset, c, alpha, minChange, evalu,foundit,actual_outcomes,cls1,cls2,dataList):
    '''
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2,0.3],[0.1,0.4]], [[1,0],[1,0]], 0.06, 0.8, 1.2) -> UNSAT
    '''
    
    # Initialisation
    M = len(hsets)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    opt = Solver()
    size = 20
    (a,b) = c

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
        for j in range(len(hids[i])):
            hc = hmap[hids[i][j]]
            if hc.get_value() == 1:
                retVsRow.append(True)
            elif hc.get_value() == 0:
                retVsRow.append(False)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)
    
    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Initialisation of the variable list
    X = [[[] for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                for k in range(pathProbs[i][j]):
                    X[i][j].append(Bool('x%sx%sx%s' %(i,j,k)))
    Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                Y[i][j] = Bool('y%sy%s' %(i,j))
                for k in range(pathProbs[i][j]):
                    opt.add(Y[i][j] == X[i][j][k]) 

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for k in range(len(pathProbs[i])):
            if not pathProbs[i][k] == 0:
                m[i] += pathProbs[i][k] * If(Y[i][k],1,0)   

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
            if intProportions[j] > size and intProportions[i] > size:
                opt.add(m[i]*intProportions[j]*B > A*m[j]*intProportions[i])
    
    # Semantic difference requirement (number of people being affected)
    sd = []
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                for k in range(pathProbs[i][j]):
                    sd.append(Xor(X[i][j][k],retVs[i][j]))      

    opt.add(AtMost(*sd, math.ceil(minChange*alpha)))
    
    result = opt.check()
    if result == sat:
        # print("The flipping result is: SAT.")
        m = opt.model()
        newRetVs = [[True for j in range(len(retVs[i]))] for i in range(M)]
        for i in range(M):
            for j in range(len(pathProbs[i])):
                if not pathProbs[i][j] == 0:
                    newRetVs[i][j] = m.eval(Y[i][j])
                else:
                    newRetVs[i][j] = retVs[i][j]
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(len(retVs)):
            for j in range(len(retVs[i])):
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[i][j] == True:
                    passingNums[i] += pathProbs[i][j]
        ratios = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] > size and intProportions[j] > size:
                    ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))

        if not foundit.is_set():
            foundit.set()
            repaired_outcomes = [None for i in range(sizeOfDataset)]
            for i in range(len(hCubeSetIndices)):
                hmap = hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map()
                for j in range(len(hids[i])):
                    if newRetVs[i][j]:
                        hmap[hids[i][j]].set_value(1)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls2
                    else:
                        hmap[hids[i][j]].set_value(0)
                        for pt in hmap[hids[i][j]].get_points():
                            repaired_outcomes[pt] = cls1
            true_pos, false_pos, true_neg, false_neg = accuracyCalculator(repaired_outcomes,actual_outcomes,cls1,cls2,dataList)

            evalu.record_eval("final-passing-rates",[passingNums[i]/intProportions[i] for i in range(len(passingNums))])
            evalu.record_eval("final-passing-nums",passingNums)
            evalu.record_eval("final-fairness", ratios)
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("true-positive-after", true_pos)
            evalu.record_eval("false-positive-after", false_pos)
            evalu.record_eval("true-negative-after", true_neg)
            evalu.record_eval("false-negative-after", false_neg)
            evalu.record_eval("precision-after",true_pos/(true_pos+false_pos))
            evalu.record_eval("recall-after",true_pos/(true_pos+false_neg))
            evalu.record_eval("accuracy-after",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))
            # q.put((True, count, sum(count), ratios))
        return True, count, sum(count), ratios
    elif result == unsat:
        # print("The flipping result is: UNSAT.")
        return False, [],[],[]
    else:
        raise RuntimeError("Result is undefined.")


def anotherRetVsSolver(min_,max_,intersectedHsets,lst_,forest_size, hCubeSetIndices, sizeOfDataset, c, alpha, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList):
    
    # Initialisation
    start = time.time()
    M = len(intersectedHsets)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    opt = Solver()
    size = 20

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])]
        for j in range(len(hmap)):
            (hc,lst,ct) = hmap[j]
            if hc.get_value() == 1:
                retVsRow.append(1)
            elif hc.get_value() == 0:
                retVsRow.append(0)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)

    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Initialisation of the variable list
    Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            for j in range(len(Y[i])):
                Y[i][j] = Int('y%sy%s' %(i,j))
                if not pathProbs[i][j] == 0:
                    opt.add(Or(Y[i][j]==0,Y[i][j]==1))
                else:
                    opt.add(Y[i][j] == retVs[i][j])
    
    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            # Count the data points in one sens group
            for k in range(len(Y[i])):
                if not pathProbs[i][k] == 0:
                    m[i] += pathProbs[i][k] * Y[i][k]

    for i in range(M):
        x,y = lst_[i]
        if y > 0:
            if x > y and intProportions[i] > size:
                opt.add(m[i] <= y)
                mc = math.floor(abs(x-y)*alpha)
                opt.add(m[i] >= max(y - mc,math.ceil(min_ * intProportions[i])))
                # print(y, max(y - mc,math.ceil(min_ * intProportions[i])))
            if x < y and intProportions[i] > size:
                opt.add(m[i] >= y)
                mc = math.floor(abs(x-y)*alpha)
                opt.add(m[i] <= min(y + mc,math.floor(max_ * intProportions[i])))
                # print(y,min(y + mc,math.floor(max_ * intProportions[i])))
        else:
            if intProportions[i] > size:
                for j in range(len(Y[i])):
                    opt.add(Y[i][j] == retVs[i][j])
    
    sd = [0 for i in range(M)]
    for i in range(M):
        if intProportions[i] > size:
            x,y = lst_[i]
            for j in range(len(retVs[i])):
                if retVs[i][j] == 0:
                    sd[i] += Y[i][j]*pathProbs[i][j]
                if retVs[i][j] == 1:
                    sd[i] += (1-Y[i][j])*pathProbs[i][j]
            if y > 0:
                # opt.add(sd[i] <= math.ceil(abs(x-y)*alpha)+1)
                opt.add(sd[i] <= math.ceil(abs(x-y)*alpha))
            
    result = opt.check()
    if result == sat:
        # print("The flipping result is: SAT.")
        m_ = opt.model()
        newRetVs = copy.deepcopy(retVs)
        for i in range(M):
            if intProportions[i] > size:
                for j in range(len(pathProbs[i])):
                    if not pathProbs[i][j] == 0:
                        if m_.eval(Y[i][j]) == 1:
                            newRetVs[i][j] = 1
                        elif m_.eval(Y[i][j]) == 0:
                            newRetVs[i][j] = 0
        count = [0 for i in range(M)]
        passingNums = [sum([pathProbs[i][j]*newRetVs[i][j] for j in range(len(newRetVs[i]))]) for i in range(M)]
        # print("what",passingNums)
        syntacticChange = 0
        for i in range(len(retVs)):
            for j in range(len(retVs[i])):
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                    hc,ls,ct = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j]
                    if ct <= median(forest_size):
                        syntacticChange += median(forest_size)-ct
                    else:
                        syntacticChange += median(forest_size)-(forest_size-ct)
        #         if newRetVs[i][j] == 1:
        #             passingNums[i] += pathProbs[i][j]
        # print("what",passingNums)
        
        ratios = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] > size and intProportions[j] > size:
                    ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
        hsetsLists = []
        for i in range(len(hCubeSetIndices)):
            hsetsRow = []
            for j in range(len(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])])):
                hsetsRow.append(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j])
            hsetsLists.append(hsetsRow)

        repaired_outcomes = [None for i in range(len(actual_outcomes))]
        for i in range(len(hsetsLists)):
            for j in range(len(hsetsLists[i])):
                hcube,hids,ct_ = hsetsLists[i][j]
                if newRetVs[i][j]:
                    for pt in hcube.get_points():
                        repaired_outcomes[pt] = cls2
                else:
                    for pt in hcube.get_points():
                        repaired_outcomes[pt] = cls1
        
        true_pos, false_pos, true_neg, false_neg = accuracyCalculator(repaired_outcomes,actual_outcomes,cls1,cls2,dataList)

        if not foundit.is_set():
            foundit.set() 
            evalu.record_eval("final-passing-rates",[passingNums[i]/intProportions[i] for i in range(len(passingNums))])
            evalu.record_eval("final-passing-nums",passingNums)
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("final-fairness", ratios)
            evalu.record_eval("average-syntactic-change",syntacticChange/forest_size)
            evalu.record_eval("time_flip", time.time()-start)
            evalu.record_eval("true-positive-after", true_pos)
            evalu.record_eval("false-positive-after", false_pos)
            evalu.record_eval("true-negative-after", true_neg)
            evalu.record_eval("false-negative-after", false_neg)
            evalu.record_eval("precision-after",true_pos/(true_pos+false_pos))
            evalu.record_eval("recall-after",true_pos/(true_pos+false_neg))
            evalu.record_eval("accuracy-after",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))
            # q.put(True, count, sum(count), ratios)

        return True, count, sum(count), ratios

    elif result == unsat:
        print("The flipping result is: UNSAT.")
        return False, [], 0, []
    else:
        raise RuntimeError("Result is undefined.")


def retVsSolverIntersected(intersectedHsets,forest_size, hCubeSetIndices, c, alpha, y_soln, passingRates, evalu,actual_outcomes,cls1,cls2,foundit,dataList):
    '''
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2,0.3],[0.1,0.4]], [[1,0],[1,0]], 0.06, 0.8, 1.2) -> UNSAT
    '''
    
    # Initialisation
    start = time.time()
    M = len(intersectedHsets)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    # opt1 = Tactic('psmt').solver()
    # opt2 =Tactic('psmt').solver()
    opt = Solver()
    opt1 = Solver()
    opt2 = Solver()
    (a,b) = c
    size = 20
    # size = max(int(len(actual_outcomes)/100),20)

    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])]
        for j in range(len(hmap)):
            (hc,lst,ct) = hmap[j]
            if hc.get_value() == 1:
                retVsRow.append(True)
            elif hc.get_value() == 0:
                retVsRow.append(False)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)
    
    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Calculate the minimun change in each sens group
    minChanges = y_soln

    # Initialisation of the variable list
    Y = [[None for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                Y[i][j] = Bool('y%sy%s' %(i,j))
                # hc,ls,ct = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j]
                # for pt in hc.get_points():
                #     if actual_outcomes[pt] == cls2:
                #         opt.add_soft(Y[i][j] == True)
                #     else:
                #         opt.add_soft(Y[i][j] == False)

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        # Count the data points in one sens group
        for k in range(len(pathProbs[i])):
            if not pathProbs[i][k] == 0:
                m[i] += pathProbs[i][k] * If(Y[i][k],1,0)

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
            if intProportions[j] > size and intProportions[i] > size:
                opt.add(m[i]*intProportions[j]*B >= A*m[j]*intProportions[i])
                opt1.add(m[i]*intProportions[j]*B >= A*m[j]*intProportions[i])
                opt2.add(m[i]*intProportions[j]*B >= A*m[j]*intProportions[i])
    
    # Semantic difference requirement (number of people being affected)
    sd = []
    sd1 = []
    for i in range(M):
        sd_row = []
        for j in range(len(retVs[i])):
            if not pathProbs[i][j] == 0:
                sd_row.append((Xor(Y[i][j],retVs[i][j]),pathProbs[i][j]))
                sd1.append((Xor(Y[i][j],retVs[i][j]),pathProbs[i][j]))
        sd.append(sd_row)

    for i in range(M):
        opt.add(PbLe(sd[i],math.ceil(alpha*minChanges[i])))
        opt2.add(PbLe(sd[i],int(minChanges[i])))
    
    opt1.add(PbLe(sd1,math.ceil(alpha*sum(minChanges))))

    flag = False
    result = opt.check()
    if result == sat:
        m = opt.model()
    else:
        print("NO")
        result = opt1.check()
        if result == sat:
            m = opt1.model()
        else:
            print("NO")
            flag = True
            # result = opt1.check()
            # if result == sat:
            #     m = opt1.model()
            # else:
            #     flag = True
    if flag:
        print("The flipping result is: UNSAT.")
        return False,[],[],[]
    else:
        newRetVs = copy.deepcopy(retVs)
        for i in range(M):
            for j in range(len(pathProbs[i])):
                if not pathProbs[i][j] == 0:
                    if m.eval(Y[i][j]) == True:
                        newRetVs[i][j] = True
                    elif m.eval(Y[i][j]) == False:
                        newRetVs[i][j] = False
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        syntacticChange = 0
        for i in range(len(retVs)):
            for j in range(len(retVs[i])):
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                    hc,ls,ct = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j]
                    if ct <= median(forest_size):
                        syntacticChange += median(forest_size)-ct
                    else:
                        syntacticChange += median(forest_size)-(forest_size-ct)
                if newRetVs[i][j] == True:
                    passingNums[i] += pathProbs[i][j]
        
        ratios = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] > size and intProportions[j] > size:
                    ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
        hsetsLists = []
        for i in range(len(hCubeSetIndices)):
            hsetsRow = []
            for j in range(len(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])])):
                hsetsRow.append(intersectedHsets[sens_group_to_str(hCubeSetIndices[i])][j])
            hsetsLists.append(hsetsRow)

        repaired_outcomes = [None for i in range(len(actual_outcomes))]
        for i in range(len(hsetsLists)):
            for j in range(len(hsetsLists[i])):
                hcube,hids,ct_ = hsetsLists[i][j]
                if newRetVs[i][j]:
                    for pt in hcube.get_points():
                        repaired_outcomes[pt] = cls2
                else:
                    for pt in hcube.get_points():
                        repaired_outcomes[pt] = cls1
        
        true_pos, false_pos, true_neg, false_neg = accuracyCalculator(repaired_outcomes,actual_outcomes,cls1,cls2,dataList)
        
        if not foundit.is_set():
            foundit.set()

            evalu.record_eval("final-passing-rates",[passingNums[i]/intProportions[i] for i in range(len(passingNums))])
            evalu.record_eval("final-passing-nums",passingNums)
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("final-fairness", ratios)
            evalu.record_eval("average-syntactic-change",syntacticChange/forest_size)
            evalu.record_eval("time_flip", time.time()-start)
            evalu.record_eval("true-positive-after", true_pos)
            evalu.record_eval("false-positive-after", false_pos)
            evalu.record_eval("true-negative-after", true_neg)
            evalu.record_eval("false-negative-after", false_neg)
            evalu.record_eval("precision-after",true_pos/(true_pos+false_pos))
            evalu.record_eval("recall-after",true_pos/(true_pos+false_neg))
            evalu.record_eval("accuracy-after",(true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg))
            
        return True, count, sum(count), ratios


def forestRetVsSolver(min_,max_,lst_,hsets,dataList,outcomes,intersectedHsets,pt_sens_groups,pt_hid_map,hCubeSetIndices,sizeOfDataset,c,alpha,y_soln,classes, passingRates,evalu):
    '''
    This function takes in the above parameters, and outputs the return values of 
    all paths after necessary flippings without refining any hypercubes
    
    An testing example:
    maxSMTSolving([[0.2,0.3],[0.1,0.4]], [[1,0],[1,0]], 0.06, 0.8, 1.2) -> UNSAT
    '''
    
    # Initialisation
    flip_start = time.time()
    M = len(hCubeSetIndices)
    # set_option("parallel.enable", True)
    # set_option("parallel.threads.max", 32)
    # opt = Tactic('psmt').solver()
    # opt1 = Tactic('psmt').solver()
    opt = Solver()
    opt1 = Solver()
    (a,b) = c
    forest_size = len(hsets)
    (class1,class2) = classes
    med = median(forest_size)
    size = 20
    # size = max(20,int(sizeOfDataset/100))
    
    # Record the return values and path probabilities
    retVs = []
    pathProbs = []
    for i in range(len(hCubeSetIndices)):
        retVsRow = []
        pathProbsRow = []
        hmap = intersectedHsets[sens_group_to_str(hCubeSetIndices[i])]
        for j in range(len(hmap)):
            (hc,lst,ct) = hmap[j]
            if hc.get_value() == 1:
                retVsRow.append(True)
            elif hc.get_value() == 0:
                retVsRow.append(False)
            else:
                raise RuntimeError("Invalid return value.")
            if isinstance(hc.get_passing_rate(), int):
                pathProbsRow.append(hc.get_passing_rate())
            else:
                pathProbsRow.append(hc.get_passing_rate().item())
        retVs.append(retVsRow)
        pathProbs.append(pathProbsRow)

    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    # Calculate the minimun change in each sens group
    minChanges = [0 for i in range(M)]
    for i in range(M):
        if y_soln[i] > 0:
            minChanges[i] = int(alpha*intProportions[i]*y_soln[i]) + 1
    
    # Fix the order of hids in each tree
    hids = [[list(hsets[i][sens_group_to_str(hCubeSetIndices[j])].get_hcubes_map().keys()) for j in range(M)] for i in range(forest_size)]
    hidDicts = [[{} for j in range(M)] for i in range(forest_size)]
    for i in range(forest_size):
        for j in range(M):
            for k in range(len(hids[i][j])):
                hidDicts[i][j][hids[i][j][k]] = k

    # Initialiation of the variable lists
    X = [[[Bool('X_%s_%s_%s' %(i,j,k)) for k in range(len(hids[i][j]))] for j in range(M)] for i in range(forest_size)]
   
    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        sens = sens_group_to_str(hCubeSetIndices[i])
        # Count the data points in one sens group
        for k in range(len(pathProbs[i])):
            if not pathProbs[i][k] == 0:
                hcube,hidMap,ct_ = intersectedHsets[sens][k]
                var = [X[j][i][hidDicts[j][i][hidMap[j]]] for j in range(forest_size)]
                m[i] += pathProbs[i][k] * If(AtLeast(*var,med),1,0)

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
            if intProportions[j] > size and intProportions[i] > size:
                opt.add(m[i]*intProportions[j]*B >= A*m[j]*intProportions[i])
    
    for i in range(M):
        x,y = lst_[i]
        if y > 0:
            pass
            if x > y and intProportions[i] > size:
                opt1.add(m[i] <= y)
                mc = math.ceil(abs(x-y)*alpha)
                opt1.add(m[i] >= max(y - mc,math.ceil(min_ * intProportions[i])))
                print(y, max(y - mc,math.ceil(min_ * intProportions[i])))
            if x < y and intProportions[i] > size:
                opt1.add(m[i] >= y)
                mc = math.ceil(abs(x-y)*alpha)
                opt1.add(m[i] <= min(y + mc,math.ceil(max_ * intProportions[i])))
                print(y,min(y + mc,math.floor(max_ * intProportions[i])))
        else:
            if intProportions[i] > size:
                for j in range(len(Y[i])):
                    opt1.add(Y[i][j] == retVs[i][j])

    # Semantic difference requirement (number of people being affected)
    sd = []
    for j in range(M):
        sd_row = []
        for i in range(len(pt_hid_map)):
            sens = None
            if pt_sens_groups[i] == sens_group_to_str(hCubeSetIndices[j]):
                sens = j
                var_ = [X[k][sens][hidDicts[k][sens][pt_hid_map[i][k]]] for k in range(forest_size)]
                outcome = False
                if outcomes[i] == class2:
                    outcome = True
                sd_row.append(Xor(AtLeast(*var_,med),outcome))
        sd.append(sd_row)

    for i in range(M):
        opt.add(AtMost(*sd[i], minChanges[i]))
        if intProportions[i] > size:
            x,y = lst_[i]
            if y > 0:
                opt1.add(AtMost(sd[i],math.ceil(abs(x-y)*alpha)+1))

    result = opt1.check()
    if result == sat:
        m = opt.model()
    else:
        print("No")
        result = opt.check()
        if result == sat:
            m = opt.model()
            newRetVs = [[[m.eval(X[i][j][k]) for k in range(len(hids[i][j]))] for j in range(M)] for i in range(forest_size)]
            count = [0 for i in range(M)]
            passingNums = [0 for i in range(M)]
            
            for i in range(len(pt_hid_map)):
                sens = None
                for j in range(M):
                    if sens_group_to_str(hCubeSetIndices[j]) == pt_sens_groups[i]:
                        sens = j
                ct = 0
                oc = None
                for j in range(forest_size):
                    if newRetVs[j][sens][hidDicts[j][sens][pt_hid_map[i][j]]]:
                        ct += 1
                if 2*ct > forest_size:
                    oc = class2
                    passingNums[sens] += dataList[i]['frequency']
                else:
                    oc = class1
                if not oc == outcomes[i]:
                    count[sens] += dataList[i]['frequency']
            print([passingNums[i]/intProportions[i] for i in range(M)])
            ratios = []
            for i in range(len(passingNums)):
                for j in range(len(passingNums)):
                    if intProportions[i] > size and intProportions[j] > size:
                        ratios.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
            evalu.record_eval("data-points-changed-group", count)
            evalu.record_eval("data-points-changed-total", sum(count))
            evalu.record_eval("final-fairness", ratios)
            # for ratio in ratios:
            #     r,s,t = ratio
            #     assert(r>=a/b)
            evalu.record_eval("time_flip", time.time()-flip_start)
            return True
        else:
            print("The flipping result is unsat.")
            return False


def fullyRefinedSolver(hSets, hCubeSetIndices, c, alpha, evalu, y_soln, count,dataList):
    '''
    This function treats the data points as paths, i.e., the fully refined hcubes.
    It accounts for the frequencies of the duplicated data points, and use them as 
    the path probilities.
    '''

    # Initialisation
    opt = Solver()
    M = len(hCubeSetIndices)
    N = 0
    (a,b) = c
    
    # Record the location of the data point in the data set
    # Grouped based on the sensitive attributes
    sensDataset = [[] for i in hCubeSetIndices]
    for i in range(M):
        for j in hSets[i]:
            for pt in j.get_points():
                for k in range(M):    
                    flag = True
                    for sensAttr in hCubeSetIndices[k]:
                        if dataList[pt][sensAttr] <= 0.5:
                            flag = False
                            break
                    if flag:
                        sensDataset[i].append([pt,j.get_value()])
                        N += 1
                        break

    # Record the path probabilities
    if isinstance(dataList[sensDataset[0][0][0]]['frequency'], int):
        pathProbs = [[j[0]['frequency'] for j in i] for i in sensDataset]
    else: 
        pathProbs = [[j[0]['frequency'].item() for j in i] for i in sensDataset]

    # Record the return values of the data points
    retVs = []
    for i in sensDataset:
        row = []
        for j in i:
            if j[1] == 1:
                row.append(True)
            elif j[1] == 0:
                row.append(False)
            else:
                raise RuntimeError("Invalid classification.")
        retVs.append(row)

    # Get the number of data points in each hset
    proportions = [sum([j for j in i]) for i in pathProbs]
    passingNums = [0 for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            if retVs[i][j]:
                passingNums[i] += pathProbs[i][j]
    
    intProportions = [0 for i in range(len(proportions))]
    for i in range(len(proportions)):
        if not type(proportions[i]) == int:
            intProportions[i] = proportions[i].item()
        else:
            intProportions[i] = proportions[i]

    temp_proportions = [i/N for i in intProportions]
    temp_passingRates = [passingNums[i]/intProportions[i] for i in range(M)]

    ratios = []
    for i in range(len(temp_passingRates)):
        for j in range(len(temp_passingRates)):
            if proportions[i]*N >= 10 and proportions[j]*N >= 10:
                if not temp_passingRates[j] == 0:
                    ratios.append((temp_passingRates[i]/temp_passingRates[j],i,j))
                else:
                    ratios.append('N/A')

    if count == 0:        
        minChange, y_soln = rateChangeCalculator(temp_passingRates,temp_proportions,c,N)
    minChanges = [0 for i in range(M)]
    for i in range(M):
        if y_soln[i] > 0:
            minChanges[i] = int(alpha*intProportions[i]*y_soln[i]) + 1
    count += 1
    # print(y_soln)

    # Initialisation of the variable list
    X = [[[Bool('x%sx%sx%s' %(i,j,k)) for k in range(pathProbs[i][j])] for j in range(len(retVs[i]))] for i in range(M)]
    Y = [[Bool('y%sx%s' %(i,j)) for j in range(len(retVs[i]))] for i in range(M)]
    for i in range(M):
        for j in range(len(retVs[i])):
            for k in range(pathProbs[i][j]):
                opt.add(Y[i][j] == X[i][j][k])
                # print(str(Y[i][j] == X[i][j][k]))

    # Fairness requirements
    m = [0 for i in range(M)]
    for i in range(M):
        for j in range(len(pathProbs[i])):
            m[i] += If(Y[i][j],1,0) * pathProbs[i][j]
            # print(str(If(Y[i][j],1,0) * pathProbs[i][j]))
    
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
            if intProportions[j] >= 10 and intProportions[i] >= 10:
                opt.add(m[i]*intProportions[j]*B > A*m[j]*intProportions[i])
                  
    # Semantic difference requirement (number of people being affected)
    sd = []
    for i in range(M):
        sd_row = []
        for j in range(len(retVs[i])):
            for k in range(pathProbs[i][j]):
                sd_row.append(Xor(X[i][j][k],retVs[i][j])) 
        sd.append(sd_row)
    
    for i in range(M):
        opt.add(AtMost(*sd[i], minChanges[i]))
    
    if opt.check() == sat:
        # print("SAT")        
        mod = opt.model()
        # print([mod.eval(m[i]) for i in range(len(m))])
        newRetVs = [[] for i in retVs]
        count = [0 for i in range(M)]
        passingNums = [0 for i in range(M)]
        for i in range(len(retVs)):
            for j in range(len(retVs[i])):
                newRetVs[i].append(mod.eval(Y[i][j]))
                if not newRetVs[i][j] == retVs[i][j]:
                    count[i] += pathProbs[i][j]
                if newRetVs[i][j] == True:
                    passingNums[i] += pathProbs[i][j]
        
        ratios_after = []
        for i in range(len(passingNums)):
            for j in range(len(passingNums)):
                if intProportions[i] >= 10 and intProportions[j] >= 10:
                    ratios_after.append(((passingNums[i]*intProportions[j])/(passingNums[j]*intProportions[i]),i,j))
        
        return newRetVs, ratios, ratios_after, N, count, sum(count)
    else:
        new_y_soln = [0 for i in y_soln]
        for i in range(len(y_soln)):
            if y_soln[i] == 0:
                new_y_soln[i] = 1/N
            else:
                new_y_soln[i] = y_soln[i] * alpha
                print(new_y_soln)
        return fullyRefinedSolver(hSets, hCubeSetIndices, c, alpha, evalu, new_y_soln, count,dataList)


def divideHCubes(tree,fnames,sensAttrs,dataList):
    '''Computes sensitive hcubes from a tree and then computes the cross
    product of the sensitive HCubeSets and returns these.
    '''
    hsets = tree_to_sensitive_hcubes(tree, fnames, sensAttrs[0],dataList)
    if len(sensAttrs) == 1:
        return hsets
    for i in range(len(sensAttrs)-1):
        hset_temp = tree_to_sensitive_hcubes(tree, fnames, sensAttrs[i+1],dataList)
        newHsets = HCubeSet.crossproduct(hsets, hset_temp)
        hsets = newHsets
    return newHsets


def ratiosCalculator(passingRates, proportions, sizeOfDataset):
    size = 20
    ratios = []
    for i in range(len(passingRates)):
        for j in range(len(passingRates)):
            if proportions[i] > size and proportions[j] > size:
                if not passingRates[j] == 0:
                    ratios.append((passingRates[i]/passingRates[j],i,j))
                else:
                    ratios.append('N/A')
    return ratios


def assignDataPoints(evalu, hsets,dataset,tree,fnames,multicore,pt_sens_groups=None,pt_hid_map=None):
    '''Assigns points in the dataset to their respective hypercubes, which
    are themselves spread across different HCubeSets, represented by
    the hsets map.
    
    TREE:
    Use, default pt_sens_groups = None, and pt_hid_map = None when
    using with a decision tree.
    FOREST:
    Use, non-default values when using with a random forest.
    '''

    if multicore:
        cpus = mp.cpu_count()-4
        pool = mp.Pool(cpus)
        results = []
        # results = []
        M = len(dataset)

        # Divide calls to point_to_hid functions into subtasks
        # Point index modulo cpus indicates the task index
        # e.g., let cpus = 4, tasks[0] contains point = 0,4,8,...
        # tasks[1] contains point = 1,5,9,...

        task_size = int(M/cpus)+1
        tasks = []
        i = 0
        for i in range(cpus):
            tasks.append((i*task_size,(i+1)*task_size))
        tasks[cpus-1] = ((cpus-1)*task_size,M)

        for i in range(cpus):
            result = pool.apply_async(innerAssignDataPoints,args=(i,tasks,tree,fnames,dataset))
            results.append(result)
        
        pool.close()
        pool.join()

        phids = list(itertools.chain(*[results[i].get() for i in range(cpus)]))

    for i in range(len(dataset)):
        # if (i % 100 == 0):
        #     print("\t% Completed: ", (i * 1.0 / len(dataset) * 100), end='\r')
        #     sys.stdout.flush()

        if multicore:
            # phid = phids[i%cpus][int(i/cpus)]
            phid = phids[i]
        else:
            point = dataset[i]
            phid = point_to_hcube_id(tree, fnames, point)
        # added_point = False
        # already_added = False
        for kx, hset in hsets.items():
            fl = hset.add_point(dataset[i],phid)
            if fl:
                # assert(not already_added)
                # already_added = True
                ##################
                # FOREST ONLY
                if pt_sens_groups != None:
                    pt_sens_groups[i] = kx
                    pt_hid_map[i].append(phid)
                ##################
            # added_point = added_point or fl
        # if not added_point:
        #     print(phid)
        #     assert(added_point)
    return


def innerAssignDataPoints(i,tasks,tree,fnames,dataset):
    hids = []
    start,end = tasks[i]
    for i in range(start,end):
        hids.append(point_to_hcube_id(tree,fnames,dataset[i]))
    return hids


def clearConstraints(hsets):
    for hset in hsets.values():
        for hid,hc in hset.get_hcubes_map().items():
            hset.get_hcubes_map()[hid].rm_all_constraints()
    return True


def getHids(hsets,hCubeSetIndices):
    hids = [[] for i in range(len(hCubeSetIndices))]
    for i in range(len(hCubeSetIndices)):
        hids[i] = [k for k in hsets[sens_group_to_str(hCubeSetIndices[i])].get_hcubes_map().keys()]
    return hids


def paramCalculatorInt(hsets,hids,sizeOfDataset,hCubeSetIndices):
    '''
    This function calculates passing rates, path probabilities and proportions.
    hsets is a map: 
      str -> HCubeSet instance
      where str is of the form 'sens_attr1 X send_attr2'
    '''
    proportions = []
    passingRates = []
    passingNums = []
    for i in range(len(hCubeSetIndices)):
        proportion = 0
        passingRate = 0
        sens_str = sens_group_to_str(hCubeSetIndices[i])
        hmap = hsets[sens_str].get_hcubes_map()
        for j in range(len(hmap)):
            pathSize = hmap[hids[i][j]].get_passing_rate()
            proportion += pathSize
            if hmap[hids[i][j]].get_value() == 1:
                passingRate += pathSize
        proportions.append(proportion)
        if proportion == 0:
            passingRates.append(0)
        else:
            passingRates.append(passingRate/proportion)
            passingNums.append(passingRate)
    
    return proportions, passingRates, passingNums


def paramCalculatorRatio(hsets,hids,sizeOfDataset,hCubeSetIndices):
    '''
    This function calculates passing rates, path probabilities and proportions.
    hsets is a map: 
      str -> HCubeSet instance
      where str is of the form 'sens_attr1 X send_attr2'
    '''
    proportions = []
    passingRates = []
    count = 0
    for i in range(len(hCubeSetIndices)):
        proportion = 0
        passingRate = 0
        sens_str = sens_group_to_str(hCubeSetIndices[i])
        hmap = hsets[sens_str].get_hcubes_map()
        for j in range(len(hmap)):
            pathSize = hmap[hids[i][j]].get_passing_rate()
            pathProb = pathSize/sizeOfDataset
            proportion += pathProb
            count += hmap[hids[i][j]].get_passing_rate()
            if hmap[hids[i][j]].get_value() == 1:
                passingRate += pathProb
        proportions.append(proportion)
        if proportion == 0:
            passingRates.append(0.0)
        else:
            passingRates.append(passingRate/proportion)
    
    return proportions, passingRates


def rateChangeCalculatorInt(passingRates, proportions, passingNums, c, sizeOfDataset):
    ''' 
    This function takes in passing rates (as an array),
    proportions (as an array), and a fairness threshold (real, 0 < c < 1),
    and outputs the minimal possible change in passing rates,
    to achieve fairness requirement.
    '''
    M = len(passingRates)
    prob = LpProblem("Minimal-Passing-Rate-Change", LpMinimize)
    (a,b) = c
    # size = 20
    size = 20

    # xs is the list of x_i's. we change r_i's to x_i's
    var_names = ['x_' + str(i) for i in range(M)]
    xs = [LpVariable(var_names[i], lowBound=0, upBound=proportions[i], cat='Integer') for i in range(len(var_names))]

    # ys is used for absolute values. pulp doesn't support absolute values directly
    var_names1 = ['y_' + str(i) for i in range(M)]
    ys = [LpVariable(var_names1[i], lowBound=0, upBound=proportions[i], cat='Integer') for i in range(len(var_names1))]

    prob += lpSum(ys[i] for i in range(M))
    
    # fairness requirement
    for i in range(M):
        for j in range(M):
            if proportions[i] > size and proportions[j] > size:
                prob += xs[i]*proportions[j]*b >= a*xs[j]*proportions[i]
    
    # absolute value
    for i in range(M):
        prob += ys[i] >= xs[i] - passingNums[i]
        prob += ys[i] >= passingNums[i] - xs[i]

    prob.solve()

    x_soln = np.array([xs[i].varValue for i in range(M)])
    y_soln = np.array([ys[i].varValue for i in range(M)])

    minChange = 0
    for i in range(len(y_soln)):
        if not y_soln[i] == 0:
            minChange += y_soln[i]
            # if x_soln[i] > passingRates[i]:
            #     optPassRates[i] = int(proportions[i] * x_soln[i] * sizeOfDataset) + 1
            # elif x_soln[i] < passingRates[i]:
            #     optPassRates[i] = int(proportions[i] * x_soln[i] * sizeOfDataset)
    # print("The minimal change is:")
    # print(minChange)
    return minChange, y_soln, x_soln


def rateChangeCalculatorRatio(passingRates, proportions, c, sizeOfDataset):
    ''' 
    This function takes in passing rates (as an array),
    proportions (as an array), and a fairness threshold (real, 0 < c < 1),
    and outputs the minimal possible change in passing rates,
    to achieve fairness requirement.
    '''
    M = len(passingRates)
    prob = LpProblem("Minimal-Passing-Rate-Change", LpMinimize)
    (a,b) = c
    c = a/b
    size = 20
    
    # xs is the list of x_i's. we change r_i's to x_i's
    var_names = ['x_' + str(i) for i in range(M)]
    xs = [LpVariable(i, lowBound=0, upBound=1) for i in var_names]

    # ys is used for absolute values. pulp doesn't support absolute values directly
    var_names1 = ['y_' + str(i) for i in range(M)]
    ys = [LpVariable(i, lowBound=0, upBound=1) for i in var_names1]

    prob += lpSum(ys[i]*proportions[i] for i in range(M))
    
    # fairness requirement
    for i in range(M):
        for j in range(M):
            if proportions[i]*sizeOfDataset > size and proportions[j]*sizeOfDataset > size:
                prob += xs[i] >= c*xs[j]
    
    # absolute value
    for i in range(M):
        prob += ys[i] >= xs[i] - passingRates[i]
        prob += ys[i] >= passingRates[i] - xs[i]

    prob.solve()

    x_soln = np.array([xs[i].varValue for i in range(M)])
    y_soln = np.array([ys[i].varValue for i in range(M)])

    minChange = 0
    for i in range(len(y_soln)):
        if not y_soln[i] == 0:
            minChange += math.ceil(proportions[i] * y_soln[i] * sizeOfDataset)

    return minChange, y_soln, x_soln


def min_max_calculator(optPassRates,optPassRates_,proportions,passingNums,size):
    max_ = max([optPassRates[i] for i in range(len(optPassRates)) if proportions[i] > size])
    min_ = min([optPassRates[i] for i in range(len(optPassRates)) if proportions[i] > size])
    lst = []    
    for i in range(len(proportions)):
        if abs(passingNums[i]-optPassRates_[i]) < 0.01:
            lst.append((passingNums[i],0))
        elif passingNums[i] > optPassRates_[i]:
            lst.append((passingNums[i],math.floor(optPassRates_[i])))
        else:
            lst.append((passingNums[i],math.ceil(optPassRates_[i])))
    return min_,max_,lst


def countNumHcubes(hsets):
    # Record number of hcubes after split/refinement
    numHcubes = 0
    numHcubesAll = 0
    for i in hsets.values():
        for j in i.get_hcubes_list():
            numHcubesAll += 1
            if not j.get_passing_rate() == 0:
                numHcubes += 1                
    return numHcubes, numHcubesAll


def countNumPoints(hsets):
    # Record number of points after split/refinement
    count = 0
    for i in hsets.values():
        for j in i.get_hcubes_list():
            count += j.get_passing_rate()
    return count


def refineProcedure(refineHeuristics,hids,hCubeSetIndices,hsets,c,alpha,minChanges1,minChanges2,sizeOfDataset,last_time,evalu,dataList,cls1,cls2,fullyRefined,actual_outcomes,min1,max1,lst1,min2,max2,lst2):
    start_refine = time.time()
    totalNumRef = 0
    lenRefine = len(refineHeuristics)

    for attr,discrt in refineHeuristics:
        # print("refine!")
        totalNumRef += 1
        minChange = max(math.ceil(sum(minChanges1)),math.ceil(sum(minChanges2)))

        if fullyRefined:
            alpha = alpha * alpha
        
        if discrt:
            # For refinement wrt a discrete attribute
            hsets = refine_dcrt_hsets(hsets,hCubeSetIndices,attr,dataList)
        else:
            # For refinement wrt a continuous attribute
            hsets = refine_cont_hsets(hsets,hCubeSetIndices,attr,dataList)

        numHcubes,numHcubesAll = countNumHcubes(hsets)
        hids = getHids(hsets,hCubeSetIndices)
        evalu.record_eval("num-hcubes-after-{}-refinement-nonzero".format(totalNumRef), numHcubes)
        if numHcubes == sizeOfDataset:
            fullyRefined = True
        evalu.record_eval("num-hcubes-after-{}-refinement-all".format(totalNumRef), numHcubesAll)

        pool = mp.Pool(5)
        m = mp.Manager()
        foundit = m.Event()
        results = []

        p1 = pool.apply_async(retVsSolverOld,args=(hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges1,evalu,foundit,actual_outcomes,cls1,cls2,dataList))
        p2 = pool.apply_async(retVsSolverOld,args=(hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges2,evalu,foundit,actual_outcomes,cls1,cls2,dataList))
        p3 = pool.apply_async(retVsSolver,args=(min1,max1,lst1,hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
        p4 = pool.apply_async(retVsSolver,args=(min2,max2,lst2,hsets,hids,hCubeSetIndices,sizeOfDataset,c,alpha,evalu,actual_outcomes,cls1,cls2,foundit,dataList))
        p5 = pool.apply_async(retVsSolverFinal,args=(hsets, hids, hCubeSetIndices, sizeOfDataset, c, alpha, minChange, evalu,foundit,actual_outcomes,cls1,cls2,dataList))
        results.append(p1)
        results.append(p2)
        results.append(p3)
        results.append(p4)
        results.append(p5)

        sat = False
        while not sat:
            if foundit.is_set():
                time.sleep(5)
                pool.terminate()
                sat = True
            if all([p.ready() for p in results]):
                if foundit.is_set():
                    time.sleep(5)
                    pool.terminate()
                    sat = True
                else:
                    sat = False
                    break
        
        if sat:
            refine_time = time.time()-start_refine
            # calculate number of hcubes after refinement
            numHcubes = 0
            numHcubesAll = 0
            for i in hsets.values():
                for j in i.get_hcubes_list():
                    numHcubesAll += 1
                    if not j.get_passing_rate() == 0:
                        numHcubes += 1
            return hsets, totalNumRef, refine_time, numHcubes
        
        if totalNumRef == lenRefine:
            return None


def refineProcedureForest(refineHeuristics,outcomes,evalu,hCubeSetIndices,hsets,pt_sens_groups,pt_hid_map,c,alpha,minChanges,classes,sizeOfDataset,dataList,passingRates):
    totalNumRef = 0
    lenRefine = len(refineHeuristics)
    cls1,cls2 = classes
    for attr,discrt in refineHeuristics:
        totalNumRef += 1
        for i in range(len(hsets)):
            if discrt:
                hsets[i] = refine_dcrt_hsets(hsets[i],hCubeSetIndices,attr,dataList)
            else:
                hsets[i] = refine_cont_hsets(hsets[i],hCubeSetIndices,attr,dataList)
            for hset in hsets[i].values():
                hmap = hset.get_hcubes_map()
                for hid_ in hmap.keys():
                    hcube = hmap[hid_]
                    if not hcube.get_passing_rate() == 0:
                        for pt in hcube.get_points():
                            pt_hid_map[pt][i]=hid_

        intersect_start = time.time()
        intersectedHsets = intersectHsets(hsets,hCubeSetIndices,pt_hid_map,pt_sens_groups,dataList,len(hsets),outcomes,cls2)
        evalu.record_eval("time-intersection", time.time()-intersect_start)
        result = retVsSolverIntersected(intersectedHsets,len(hsets),hCubeSetIndices, c, alpha, minChanges, passingRates,evalu,outcomes,cls1,cls2)
        # result = forestRetVsSolver(False,hsets,optPassRates,dataList,outcomes,intersectedHsets,pt_sens_groups,pt_hid_map,hCubeSetIndices,sizeOfDataset,c,alpha,minChanges,classes, passingRates,evalu)
        
        # Record the number of hcubes after refine
        numHcubes = [0 for i in range(len(hsets))]
        numHcubesAll = [0 for i in range(len(hsets))]
        for i in range(len(hsets)):
            numHcubes[i],numHcubesAll[i] = countNumHcubes(hsets[i])
        evalu.record_eval("num-hcubes-refine-{}-nonzero".format(totalNumRef), numHcubes)
        evalu.record_eval("num-hcubes-refine-{}-all".format(totalNumRef), numHcubesAll)

        if result:
            evalu.record_eval("refinement-steps",totalNumRef)
            return totalNumRef
        
        if totalNumRef == lenRefine:
            return -1


def refine_dcrt_hsets(hsets,hCubeSetIndices,attr,dataList):
    for sens_group in hCubeSetIndices:
        sens_key = sens_group_to_str(sens_group)
        hset = hsets[sens_key]
        hmap = hset.get_hcubes_map()
        hids = list(hmap.keys())
        for i in hids:
            if not len(hmap[i].get_points()) == 0:
                for sens in attr:
                    hcube = HCube(constraints=copy.deepcopy(hmap[i].get_constraints()),val=hmap[i].get_value(),hid=i+'U['+sens+']',pts=copy.deepcopy(hmap[i].get_points()),desc=copy.deepcopy(hmap[i].get_desc()),dataList=dataList)
                    hcube.refine_one_hot(sens,dataList)
                    hcube.add_desc(sens)
                    hmap[i+'U['+sens+']'] = hcube
                hmap.pop(i,None)
    return hsets        


def refine_cont_hsets(hsets,hCubeSetIndices,sens,dataList):
    for sens_group in hCubeSetIndices:
        sens_key = sens_group_to_str(sens_group)
        hset = hsets[sens_key]
        hmap = hset.get_hcubes_map()
        hids = list(hmap.keys())
        for i in hids:
            if not len(hmap[i].get_points()) == 0:
                rank = [dataList[pt][sens] for pt in hmap[i].get_points()]
                med = np.median(rank)
                hcube1 = HCube(constraints=copy.deepcopy(hmap[i].get_constraints()),val=hmap[i].get_value(),hid=i+'U['+sens+']',pts=copy.deepcopy(hmap[i].get_points()),desc=copy.deepcopy(hmap[i].get_desc()),dataList=dataList)
                hcube2 = copy.deepcopy(hmap[i])
                hcube1.refine_cont_one_hot(sens,med,dataList)
                hcube1.add_desc(sens+'>')
                hmap[i+'U['+sens+']'] = hcube1
                hmap[i+'L['+sens+']'] = hcube2.refine_cont_one_hot(sens,med,dataList)
                hmap[i+'L['+sens+']'].add_desc(sens+'<')
                hmap.pop(i,None)
    return hsets        


def intersectPoints(hcubes,dataList):
    '''
    Given a list of hypercubes, this function computes their common datapoints.
    '''
    pts = [set() for i in range(len(hcubes))]
    for i in range(len(hcubes)):
        for pt in hcubes[i].get_points():
            pts[i].add(pt)
    pts_indices = set.intersection(*pts)
    return pts_indices


def intersectHcubes(hcubes,forest_size,dataList,outcomes,class2):
    '''
    This function takes as input a list of hypercubes, and produces the intersection 
    of them, but with all constrants removed.
    '''
    if len(hcubes) == 0:
        return None
    ptList = intersectPoints(hcubes,dataList)
    count = sum([hcube.get_value() for hcube in hcubes])
    value = 0
    for i in ptList:
        break
    if outcomes[i] == class2:
        value = 1
    newHcube = HCube(constraints=None,val=value,hid=None,pts=ptList,desc=None,dataList=dataList)
    return count,newHcube


def intersectHsets(hsets,hCubeSetIndices,pt_hid_map,pt_sens_groups,dataList,forest_size,outcomes,class2):
    '''
    This function takes in the random forest, and produces the intersected hcubes.
    '''
    intersectedHsets = {}
    for sens in hCubeSetIndices:
        intersectedHsets[sens_group_to_str(sens)] = []
    dl = {i for i in range(len(dataList))}
    while len(dl) > 0:
        i = dl.pop()
        sens = pt_sens_groups[i]        
        hcubes = [hsets[j][sens].get_hcubes_map()[pt_hid_map[i][j]] for j in range(forest_size)]
        count,hcube = intersectHcubes(hcubes,forest_size,dataList,outcomes,class2)
        triple = (hcube,pt_hid_map[i],count)
        intersectedHsets[sens].append(triple)
        dl = {j for j in dl if not j in hcube.get_points()}
    return intersectedHsets


def sens_group_to_str(sens_group):
    sens_str = ""
    for i in range(len(sens_group)):
        sens_str += sens_group[i] + ' X '
    sens_str = sens_str[:-3]
    return sens_str


def hsets_to_list(hsets,hCubeSetIndices):
    hSets = [[] for i in range(len(hCubeSetIndices))]
    list_hsets_indices = [k for k in hsets.keys()]
    list_hsets = [hsets[k] for k in list_hsets_indices]
    for i in range(len(list_hsets)):
        for j in list_hsets[i].get_hcubes_list():
            hSets[i].append(j)
    return hSets


def accuracyCalculator(lst1,lst2,cls1,cls2,dataList):
    '''
    This function computes the true/false positives/negatives given the predicted outcome and the actual outcome.
    '''
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for i in range(len(lst1)):
        if lst1[i] == cls2:
            if lst2[i] == cls2:
                true_pos += dataList[i]['frequency']
            if lst2[i] == cls1:
                false_pos += dataList[i]['frequency']
        if lst1[i] == cls1:
            if lst2[i] == cls2:
                false_neg += dataList[i]['frequency']
            if lst2[i] == cls1:
                true_neg += dataList[i]['frequency']
    return true_pos, false_pos, true_neg, false_neg


def readData(ds, cols):
    # Read data from the given directory.
    dataset = pd.read_csv('./data/'+ds,index_col=None,sep=',')
    len_ds = len(dataset)
    # Removes dup lines and adds a column 'frequency' with the number of occurences of each line
    df = dataset.groupby(dataset.columns.tolist()).size().reset_index().rename(columns={0:'frequency'})

    noclass = df.drop(['Class','frequency'], axis=1)
    y = df['Class']

    # Get dummy variables for categorical atrributes.
    # print('Getting dummies for X...')
    X = pd.get_dummies(noclass, columns = cols)
    # print('done')

    # Get dummy variables for categorical atrributes.
    # print('Getting dummies for df...')
    df_encoded = pd.get_dummies(df, columns = cols)
    # print('done')

    # Test if the dummy variables work
    # print(X.shape)
    # print(X.head())
    return len_ds,df_encoded,X,y


def trainDecisionTree(X,y,random_seed):
    # Build the tree using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed)
    classifier = DecisionTreeClassifier(random_state=random_seed)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    return classifier, y_pred


def trainRandomForest(X,y,random_seed,forest_size):
    # Build random forest using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed)
    classifier = RandomForestClassifier(n_estimators=forest_size,random_state=random_seed)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    return classifier, y_pred


def tree_to_code(tree, feature_names):
    '''
    Return the code corresponding to the decision tree classifier in the 'tree' var as a string.
    Based on the stackoverflow post:
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    
    More info, here:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    '''
    tree_ = tree.tree_

    # TODO: convert this into a loop that checks if the feature exists in tree features,
    # and if it does not exist, then stop/exit/return error.

    # TODO: rename feature_name to distinguish with feature_names input var
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


def tree_to_sensitive_hcubes(tree, fnames, sens_attrs,dataList):
    '''Computes HCubeSet instances, one for each discrete value of 
    sens_attrs. An HCubeSet is a set of hypercubes, each one of which
    correspond to a path in the input tree.
    Internally, the function calls on tree_to_hcubes and then assigns
    hcubes to newly generated HCubeSets and returns these sets.
    Note that this function mutates hcubes by eliminating any
    constraints on sens_attrs.
    '''
    
    # Currently we assume that the sens_attrs includes attributes that are discrete 
    # and either 0 or 1. e.g.,:
    # ['Sex_A91', 'Sex_A92', 'Sex_A93', 'Sex_A94']
    #
    # These multiple sens_attrs may hot-encode a single attribute from the input dataset,
    # but cannot be enabled at the same time (mutually exclusive): only one of these may be
    # true per path in the tree.
    #
    # Tree is expected to have constraints like:
    # Sex_A92 <= 0.5:
    # Sex_A93 > 0.5
    
    # hcubes is a map: str hypercube-id (i.e., path-id) -> HCube instances
    hcubes = tree_to_hcubes(tree, fnames,dataList)

    # Collect all known constraints on the sensitive attribute from
    # the hyper cubes corresponding to tree.
    sens_constraints = []
    for hc in hcubes.values():
        for attr in sens_attrs:
            sens_constraints = sens_constraints + hc.get_constraints(attr)

    # If the entire tree has no constraints on any of the sensitive
    # attributes, then we have nothing to do -- no unfairness relative
    # to sensitive attribute.
    # if sens_constraints == []:
    #     raise RuntimeError("Tree has no constraints on sensitive attributes: " + sens_attrs)
    
    # Now at this point we have to reduce the set of constraints to
    # figure out (1) how many hypercube sets to generate, (2) and
    # which ranges on sensitive attribute to associate them with.

    # For now, we generate one HCubeSet per sens_attrs item:
    hsets = {}
    for attr in sens_attrs:
        # Create hset for attr
        c = Constraint(attr, False, 0.5)
        hset_attrs = {attr : c}
        hset = HCubeSet(hset_attrs)
        # Record het in hsets
        hsets[attr] = hset

    # Iterate through hypercubes and assign them to some of HCubeSets,
    # depending on the sensitive attribute constraints that they might
    # have.
    #
    # For now, we assume that the hcube has just a single constraint
    # on each of the sens_attrs (can be one of those).
    for hc in hcubes.values():
        # Collect all sensitive constraints in this hcube 
        sens_hc_constraints = []
        for attr in sens_attrs:
            sens_hc_constraints += hc.get_constraints(attr)

            # remove *all* sensitive constraints from this hypercube,
            # since this info is already recorded in the hypercube sets --
            # it is a fixed value, depending on the hypercube set.
            hc.rm_constraints(attr)
        
        # Identify the sensitive constraint, which is the sensitive
        # attribute that is true for this hcube; i.e., a constraint
        # that is a lower-bound, of the form (Sex_A94 >= 0.5)
        sensitive_lower_c = None

        # Any upper constraint *names* on sensitive attrs in this hcube.
        sensitive_upper_cs_names = []
        for c in sens_hc_constraints:
            if c.is_upper():
                sensitive_upper_cs_names.append(c.get_name())
            elif c.is_lower():
                if sensitive_lower_c is not None:
                    # At the same time, check that at most one of
                    # the attributes is a lower-bound.  It's okay
                    # to have multiple upper-bounds of the form:
                    # (Sex_A91 < 0.5), (Sex_A93 < 0.5) But, at
                    # most one can be an lower-bound, e.g.,:
                    # (Sex_A94 >= 0.5)
                    raise RuntimeError("HCube has multiple lower-bound sensitive attribute constraints instead of at most one: ", sens_hc_constraints)        
                sensitive_lower_c = c

        if sensitive_lower_c is None:
            # If the hcube has no lower-bound constraints on sensitive attr, then
            # assign it to *all* HCubeSet instances that do not conflict with any
            # upper bound constraints on this HCube.
            for attr,hset in hsets.items():
                if attr in sensitive_upper_cs_names:
                    continue
                hc_copy = copy.deepcopy(hc)
                hset.add_hcube(hc_copy)
        else:
            # Add this hcube to the set corresponding to the sensitive
            # constraint in this hcube.
            attr_c = sensitive_lower_c.get_name()
            assert(attr_c in hsets)
            hsets[attr_c].add_hcube(hc)

    return hsets
    

def tree_to_hcubes(tree, feature_name,dataList):
    '''
    Returns a map of hypercube-id (i.e., path-id) to hypercube, where
    each hypercube is an instance of the HCube class.
    Note: the hypercube id is "1" followed by a sequence of 0s and 1s.
    A 0 indicates a left branch, a 1 indicates a right branch in the
    tree. The result is that a hypercube id is unique to a path in the tree.
    '''
    tree_ = tree.tree_

    # TODO: convert this into a loop that checks if the feature exists in tree features,
    # and if it does not exist, then stop/exit/return error.

    # TODO: rename feature_name to distinguish with feature_names input var
    # feature_name = [
    #     feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    #     for i in tree_.feature
    # ]

    class ComputeHCubes:
        def __init__(self):
            self.hcubes = {}
            self.curr_constraints = {}
            # TODO: convert operations/interpretation of the id from string to binary
            # self.hid = "1"
        
        def get_hcubes(self):
            self.recurse(0)
            return self.hcubes

        def add_constraint(self,constraint):
            name = constraint.get_name()
            if self.curr_constraints.get(name) == None:
                self.curr_constraints[name] = [constraint]
            else:
                self.curr_constraints[name].append(constraint)

        def rm_constraint(self,constraint):
            name = constraint.get_name()
            self.curr_constraints[name].remove(constraint)
            
        # def recurse_(self,node):
        #     while tree_.feature[node] != _tree.TREE_UNDEFINED:
        #         # Non-terminal node with two branches: if and else, with constraints on one feature
        #         name = feature_name[node]
        #         threshold = tree_.threshold[node]

        #         # Handle the left branch, which is the upper bound constraint:
        #         # feature <= threshold
        #         constraint = Constraint(name,True,threshold)
        #         self.add_constraint(constraint)
        #         # self.hid = self.hid + "0"
        #         ret = self.recurse_(tree_.children_left[node])
        #         self.rm_constraint(constraint)
        #         # self.hid = self.hid[:len(self.hid) - 1]
                
        #         # Handle the right branch, which is the lower bound constraint:
        #         # feature > threshold
        #         constraint = Constraint(name,False,threshold)
        #         self.add_constraint(constraint)
        #         # self.hid = self.hid + "1"
        #         ret = self.recurse_(tree_.children_right[node])
        #         self.rm_constraint(constraint)
        #         # self.hid = self.hid[:len(self.hid) - 1]
            
        #     # Terminal node: record path, and classification decision (hcube_value)
        #     li = list(tree_.value[node])
        #     hcube_value = np.argmax(li)
        #     # hcube = HCube(copy.deepcopy(self.curr_constraints), hcube_value, self.hid)
        #     hcube = HCube(constraints=copy.deepcopy(self.curr_constraints),val=hcube_value,hid=str(node),dataList=dataList)
        #     # self.hcubes[self.hid] = hcube
        #     self.hcubes[node] = hcube
        #     return
        
        def recurse(self,node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Non-terminal node with two branches: if and else, with constraints on one feature
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Handle the left branch, which is the upper bound constraint:
                # feature <= threshold
                constraint = Constraint(name,True,threshold)
                self.add_constraint(constraint)
                # self.hid = self.hid + "0"
                ret = self.recurse(tree_.children_left[node])
                self.rm_constraint(constraint)
                # self.hid = self.hid[:len(self.hid) - 1]
                
                # Handle the right branch, which is the lower bound constraint:
                # feature > threshold
                constraint = Constraint(name,False,threshold)
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
                hcube = HCube(constraints=copy.deepcopy(self.curr_constraints),val=hcube_value,hid=str(node),dataList=dataList)
                # self.hcubes[self.hid] = hcube
                self.hcubes[node] = hcube
                return
            return

    computeHCubes = ComputeHCubes()
    hcubes = computeHCubes.get_hcubes()
    return hcubes


def repaired_tree_to_code(tree,feature_names,hsets,sens_groups,refineHeuristics,totalNumRef):
    if totalNumRef < 0:
        return None
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    hid = 1

    def recurse(node, depth, t_str, hid):
        indent = "  " * depth * 2
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            t_str += "{}if {} <= {}:".format(indent, name, threshold) + "\n"
            t_str = recurse(tree_.children_left[node], depth + 1, t_str, hid+'0') 

            t_str += "{}else:  # if {} > {}".format(indent, name, threshold) + "\n"
            t_str = recurse(tree_.children_right[node], depth + 1, t_str,hid+'1')
        else:
            li = list(tree_.value[node])
            t_str = hcubes_to_tree(t_str,li,hsets,hid,sens_groups,indent,refineHeuristics,totalNumRef)
            t_str += "{}else:\n{}return {}".format(indent,indent+"    ",np.argmax(li)) + " --- " + hid + "\n"
        return t_str

    tree_str = "def tree({}):".format(", ".join(feature_names)) + "\n"
    return recurse(0, 1, tree_str,'1')


def hcubes_to_tree(t_str,li,hsets,hid,sens_groups,indent,refineHeuristics,totalNumRef):
    for sens_group in sens_groups:
        sens = sens_group_to_str(sens_group)
        hlist = {}
        for key in hsets[sens].get_hcubes_map().keys():
            if hid in key and not hsets[sens].get_hcubes_map()[key].get_passing_rate() == 0:
                hlist[key] = hsets[sens].get_hcubes_map()[key]
        # if len(hlist) == 0:
        #     t_str += "{}return {}".format(indent+"    ",np.argmax(li)) + " --- " + hid + "\n"
        
        if len(hlist) != 0:
            ind = indent
            for i in sens_group:
                t_str += "{}if {} > 0.5:".format(ind,i) + "\n"
                ind += "    "
            if totalNumRef == 0:
                for hc in hlist.values():
                    t_str += "{}return {}".format(ind,hc.get_value()) + "\n"
            else:
                t_str = inner_hcubes_to_tree(list(hlist.values()),t_str,ind,refineHeuristics,totalNumRef,0)
            # t_str += "{}else:\n{}return {}".format(indent,indent+"    ",np.argmax(li)) + "\n"
            t_str += "{}return {}".format(indent+"    ",np.argmax(li)) + "\n"
    return t_str


def inner_hcubes_to_tree(hlist,t_str,indent,refineHeuristics,totalNumRef,numRef):
    if numRef >= totalNumRef:
        return
    else:
        attr,discrt = refineHeuristics[numRef]
        numRef += 1
        if discrt:
            for at in attr:
                hlist1 = []
                for hcube in hlist:
                    if at in hcube.get_desc():
                        hlist1.append(hcube)
                if not len(hlist1) == 0:
                    t_str += "{}if {} > 0.5:".format(indent,at) + "\n"
                    ind = indent + "    "
                    if numRef == totalNumRef or len(hlist1) == 1: 
                        t_str += "{}return {}".format(ind,hlist1[0].get_value()) + "\n"
                    else:    
                        t_str = inner_hcubes_to_tree(hlist1,t_str,ind,refineHeuristics,totalNumRef,numRef)
        else:
            if numRef == totalNumRef or len(hlist) == 1:
                t_str += "{}if {}:".format(indent,hlist[0].get_constraints(attr)[0]) + "\n"
                t_str += "{}return {}".format(indent + "    ",hlist[0].get_value()) + "\n"
            else:
                hlist1 = []
                hlist2 = []
                for hcube in hlist:
                    if attr+'<' in hcube.get_desc():
                        hlist1.append(hcube)
                    if attr+'>' in hcube.get_desc():
                        hlist2.append(hcube)
                if not len(hlist1) == 0:
                    t_str += "{}if {}:".format(indent,hlist1[0].get_constraints(attr)[0]) + "\n"
                    ind = indent + "    "
                    if numRef == totalNumRef or len(hlist1) == 1: 
                        t_str += "{}return {}".format(ind,hlist1[0].get_value()) + "\n"
                    else:
                        t_str = inner_hcubes_to_tree(hlist1,t_str,ind,refineHeuristics,totalNumRef,numRef)
                if not len(hlist2) == 0:
                    t_str += "{}if {}:".format(indent,hlist2[0].get_constraints(attr)[0]) + "\n"
                    ind = indent + "    "
                    if numRef == totalNumRef or len(hlist2) == 1: 
                        t_str += "{}return {}".format(ind,hlist2[0].get_value()) + "\n"
                    else:
                        t_str = inner_hcubes_to_tree(hlist2,t_str,ind,refineHeuristics,totalNumRef,numRef)
    return t_str


def point_to_hcube_id(tree, feature_name, point):
    '''
    Returns a hypercube id that the point belongs to in the decision tree.
    Point is represented as a map of feature names to values.
    '''
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
            #     raise RuntimeError("No value for feature '" + name + "' in point: " + str(point))
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
            
        def recurse(self,node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Non-terminal node with two branches: if and else, with constraints on one feature
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # if not name in point:
                #     raise RuntimeError("No value for feature '" + name + "' in point: " + str(point))

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


def forest_to_trees(forest):
    rf = {}
    for idx, est in enumerate(forest.estimators_):
        rf[idx] = est
    return rf 


def median(K):
    '''
    scikit-learn random forest uses a different majority vote
    '''
    return math.ceil(K/2)+1


def recordEval(evalu, smt_time, repair_time, start_time, ratios, numHcubes, numHcubesAll, totalNumRef):
    evalu.record_eval("time-smt-calls", smt_time)
    evalu.record_eval("time-repair-all", repair_time)
    evalu.record_eval("init-fairness", ratios)
    evalu.record_eval("refinement-steps-total", totalNumRef)    
    evalu.record_eval("num-hcubes-after-nonzero", numHcubes)
    evalu.record_eval("num-hcubes-after-all", numHcubesAll)
    evalu.record_eval("time-total", time.time()-start_time)


def marginal(evalu,cols,classes):
    '''
    Compute the classification error rate for different forest size,
    to determine the best forest size to use in the experiments.
    '''
    for i in range(1,21):
        print(i*10,innerMarginal(evalu,cols,i*10,classes))
    for i in range(1,17):
        print(200+i*50,innerMarginal(evalu,cols,200+i*50,classes))


def innerMarginal(evalu,cols,forest_size,classes):
    '''
    Inner functionality for marginal.
    '''
    # Define and gather constants
    ds = evalu.get_dataset()
    len_ds,df,X,y = readData(ds, cols)
    sizeOfDataset = len(X)    
    (class1,class2) = classes

    # Train random forest and collect decision trees
    classifier, y_pred = trainRandomForest(X,y,evalu.get_seed(),forest_size)
    features = list(X.columns)
    rf = forest_to_trees(classifier)

    # Record original outcomes
    classify_start = time.time()
    if type(class1) == int:
        outcomes = [int(classifier.predict(X.iloc[[i]])) for i in range(sizeOfDataset)]
    if type(class1) == str:
        outcomes = [classifier.predict(X.iloc[[i]])[0] for i in range(sizeOfDataset)]
    # evalu.record_eval("classification-time", time.time()-classify_start)

    count = 0
    for i in range(sizeOfDataset):
        if outcomes[i] != df.iloc[i]['Class']:
            count += 1
    # print(forest_size,evalu.get_seed(),count)
    return count


def datasetFairness(hCubeSetIndices,dataList,pt_sens_groups,cls2):
    proportions = [0 for i in range(len(hCubeSetIndices))]
    passingNums = [0 for i in range(len(hCubeSetIndices))]
    for i in range(len(dataList)):
        for j in range(len(hCubeSetIndices)):
            if sens_group_to_str(hCubeSetIndices[j]) == pt_sens_groups[i]:
                proportions[j] += dataList[i]['frequency']
                if dataList[i]['Class'] == cls2:
                    passingNums[j] += dataList[i]['frequency']
    return proportions,passingNums
