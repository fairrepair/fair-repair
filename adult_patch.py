import random

import eutil
import patch

############################################################
### Constants for Adult dataset
############################################################

# Define the potential sensitive attributes and their categorical values.
sex = ['sex_Female', 'sex_Male']
race = ['race_AmerIndianEskimo', 'race_AsianPacIslander', 'race_Black', 'race_Other', 'race_White']
age = 'age'              
fnlwgt = 'fnlwgt'
educationnum = 'educationnum'
capitalgain = 'capitalgain'
capitalloss = 'capitalloss'
hoursperweek = 'hoursperweek'
workclass = [ 'workclass_Federalgov', 'workclass_Localgov', 'workclass_Private', 'workclass_Selfempinc', 'workclass_Selfempnotinc', 'workclass_Stategov', 'workclass_Withoutpay']
education = [ 'education_10th', 'education_11th', 'education_12th', 'education_1st4th', 'education_5th6th', 'education_7th8th', 'education_9th', 'education_Assocacdm', 'education_Assocvoc', 'education_Bachelors', 'education_Doctorate', 'education_HSgrad', 'education_Masters', 'education_Preschool', 'education_Profschool', 'education_Somecollege']
maritalstatus = [ 'maritalstatus_Divorced', 'maritalstatus_MarriedAFspouse', 'maritalstatus_Marriedcivspouse', 'maritalstatus_Marriedspouseabsent', 'maritalstatus_Nevermarried', 'maritalstatus_Separated', 'maritalstatus_Widowed']
occupation = ['occupation_?', 'occupation_Admclerical', 'occupation_ArmedForces', 'occupation_Craftrepair', 'occupation_Execmanagerial', 'occupation_Farmingfishing', 'occupation_Handlerscleaners', 'occupation_Machineopinspct', 'occupation_Otherservice', 'occupation_Privhouseserv', 'occupation_Profspecialty', 'occupation_Protectiveserv', 'occupation_Sales', 'occupation_Techsupport', 'occupation_Transportmoving',]
relationship = ['relationship_Husband', 'relationship_Notinfamily', 'relationship_Otherrelative', 'relationship_Ownchild', 'relationship_Unmarried', 'relationship_Wife',]

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

# Gather all attributes into a map
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
    'occupation' : occupation,
    'relationship' : relationship
}

# Some pre-defined refinement heuristics
refineHeuristics = [
    (capitalgain, False), (capitalloss, False), (fnlwgt, False), (age, False), 
    (hoursperweek, False), (maritalstatus, True), (relationship, True),
    (capitalgain, False), (capitalloss, False), (fnlwgt, False), (age, False), 
    (hoursperweek, False), (capitalgain, False), (capitalloss, False), (age, False),
]

############################################################
### Repair wrt Adult dataset
############################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch Adult dataset.',
        sensitive_attrs_default = "['race']",
        dataset_default = 'adult.data')
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, attr_map, refineHeuristics)
    evalu.save_vals()
