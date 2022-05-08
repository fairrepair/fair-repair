import random

import eutil
import patch

############################################################
### Constants for UFRGS dataset
############################################################

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

# Select categorical attributes
cols = [
    'Gender'
] 

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
refineHeuristics = [
    (Literature,False), (Portuguese,False), (Biology,False), (History, False), 
    (Seclan, False), (Chemistry, False), (Physics, False), (Geography, False), 
    (Math, False), (Literature,False), (Portuguese,False), (Biology,False),
    (History, False), (Seclan, False), (Chemistry, False), (Physics, False), 
    (Geography, False), (Math, False), 
]

############################################################
### Repair wrt UFRGS dataset
############################################################

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
    patch.patch(evalu, cols, attr_map, refineHeuristics)
    evalu.save_vals()