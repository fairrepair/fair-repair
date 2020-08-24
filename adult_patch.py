import pandas as pd
import numpy as np
import itertools
import argparse
import os.path
import random
import time

import eutil
import patch

########################################################
# Constrants for Adult dataset
########################################################

# Select categorical attributes
cols = [
    # 'age',
    'workclass',
    # 'fnlwgt',
    'education',
    # 'educationnum',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'sex',
    # 'capitalgain',
    # 'capitalloss',
    # 'hoursperweek',
    'nativecountry',
    # 'Class'
] 

# Define the potential sensitive attributes and their vategorical values.
sex = ['sex_Female', 'sex_Male']
race = ['race_AmerIndianEskimo', 'race_AsianPacIslander', 'race_Black', 'race_Other', 'race_White']
# race = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']
age = 'age'              
fnlwgt = 'fnlwgt'
educationnum = 'educationnum'
capitalgain = 'capitalgain'
capitalloss = 'capitalloss'
workclass = [ 'workclass_Federalgov', 'workclass_Localgov', 'workclass_Private', 'workclass_Selfempinc', 'workclass_Selfempnotinc', 'workclass_Stategov', 'workclass_Withoutpay']
education = [ 'education_10th', 'education_11th', 'education_12th', 'education_1st4th', 'education_5th6th', 'education_7th8th', 'education_9th', 'education_Assocacdm', 'education_Assocvoc', 'education_Bachelors', 'education_Doctorate', 'education_HSgrad', 'education_Masters', 'education_Preschool', 'education_Profschool', 'education_Somecollege']
maritalstatus = [ 'maritalstatus_Divorced', 'maritalstatus_MarriedAFspouse', 'maritalstatus_Marriedcivspouse', 'maritalstatus_Marriedspouseabsent', 'maritalstatus_Nevermarried', 'maritalstatus_Separated', 'maritalstatus_Widowed']

classes = ('<=50K','>50K')

attr_map = {
    'age' : age,
    'workclass' : workclass,
    'fnlwgt' : fnlwgt,
    'education' : education,
    'educationnum' : educationnum,
    'maritalstatus' : maritalstatus,
    'race' : race,
    'sex' : sex,
    'capitalgain' : capitalgain,
    'capitalloss' : capitalloss,
}

# Some pre-defined refinement heuristics
refineHeuristics = [(capitalgain, False), (capitalloss, False),
                    (fnlwgt, False), (educationnum, False),
                    (maritalstatus, True), (age, False),
                    (workclass, True), (education, True),
                    (fnlwgt, False), (educationnum, False)          
                    ]
########################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch adult dataset.',
        sensitive_attrs_default = "['race']",
        dataset_default = 'adult.data')
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, refineHeuristics, attr_map,classes)
    evalu.save_vals()
