import random

import eutil
import patch

############################################################
### Constants for COMPAS dataset
############################################################

# Get all attributes, continuous and discrete.
age = 'age'
c_charge_degree = ['c_charge_degree_F', 'c_charge_degree_M']
# race = ['race_African-American', 'race_Caucasian']
race = ['race_Non-Caucasian', 'race_Caucasian']
age_cat = ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25']
score_text = ['score_text_Low', 'score_text_Medium', 'score_text_High']
sex = ['sex_Female', 'sex_Male']
priors_count = 'priors_count'
days_b_screening_arrest = 'days_b_screening_arrest'
decile_score = 'decile_score'
length_of_stay = 'length_of_stay'
# is_recid = ['is_recid_0', 'is_recid_1']
# two_year_recid = ['two_year_recid_0', 'two_year_recid_1']

# Select categorical attributes
cols = [
    # 'age',
    'c_charge_degree',
    'race',
    'age_cat',
    'score_text',
    'sex',
    # 'priors_count',
    # 'days_b_screening_arrest',
    # 'decile_score',
    # 'is_recid',
    # 'length_of_stay',
    # 'two_year_recid'
] 

# Gather all attributes into a map
attr_map = {
    'age' : age,
    'c_charge_degree' : c_charge_degree,
    'race' : race,
    'age_cat' : age_cat,
    'score_text' : score_text,
    'sex' : sex,
    'priors_count' : priors_count,
    'days_b_screening_arrest' : days_b_screening_arrest,
    'decile_score' : decile_score,
    'length_of_stay' : length_of_stay,
    # 'is_recid' : is_recid,
    # 'two_year_recid' : two_year_recid,
}

# Some pre-defined refinement heuristics
refine_heuristics = [
    (age, False), (priors_count, False), (length_of_stay, False), 
    (priors_count,False), (decile_score, False), (length_of_stay, False),
    (score_text, True), (age_cat, True), (sex, True), (priors_count, False), 
    (decile_score, False), (age, False), (c_charge_degree, True), 
    (age, False), (days_b_screening_arrest, False), (c_charge_degree, True),
    (length_of_stay, False),
]

############################################################
### Repair wrt COMPAS dataset
############################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch COMPAS dataset.',
        sensitive_attrs_default = "['race']",
        dataset_default = 'compas.data')
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, attr_map, refine_heuristics)
    evalu.save_vals()
