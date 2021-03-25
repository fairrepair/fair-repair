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
# Constrants for UFRGS dataset
########################################################

# Select categorical attributes
cols = [
    'Gender'
] 

# Get all attributes, continuous and discrete.
Gender = ['Gender_0','Gender_1']
Physics = 'Physics'
Biology = 'Biology'
History = 'History'
Seclan = 'Seclan'
Geography = 'Geography'
Literature = 'Literature'
Portuguese = 'Portuguese'
Math = 'Math'
Chemistry = 'Chemistry'

classes = (0,1)

# Gather all attributes into a map
attr_map = {
    'Gender': Gender,
    'Physics': Physics,
    'Biology': Biology,
    'History': History,
    'Seclan': Seclan,
    'Geography': Geography,
    'Literature': Literature,
    'Portuguese': Portuguese,
    'Math': Math,
    'Chemistry': Chemistry
}

# Some pre-defined refinement heuristics
refineHeuristics = [(Physics, False), (Biology, False), (History, False),
                    (Seclan, False), (Geography, False), (Literature, False),
                    (Portuguese, False), (Math, False), (Chemistry, False)
                    ]
########################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch UFRGS dataset.',
        sensitive_attrs_default = "['Gender']",
        dataset_default = 'ufrgs.data')
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, refineHeuristics, attr_map,classes)
    evalu.save_vals()