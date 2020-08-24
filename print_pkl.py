import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
import os.path
import random
import time
import pprint

from eutil import *

def main(args, vals):
    pprint.pprint(vals)

def parse_args():
    parser = argparse.ArgumentParser(description='Prints a pickled evaluation file')
    parser.add_argument('eval_file', type=str,
                        help='Filename where the pickled evaluation data was stored')
    
    args = parser.parse_args()
    # Restore the pickled file into a vals dictionary
    vals = EvalUtil.restore_vals(args.eval_file)
    return args, vals

if __name__ == '__main__':
    args, vals = parse_args()
    main(args, vals)
