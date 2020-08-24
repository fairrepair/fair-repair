"""
A set of useful utilities.
"""

import pickle
import random
import argparse

def create_base_parser(description, sensitive_attrs_default, dataset_default):
    '''Creates and returns an instance of arg parser with common command
    line arguments that are used for all FairRepair programs,
    including adult_patch.py and german_patch.py
    '''
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-r', '--random-seed', type=int, help='Random seed to use', 
                    default=0, 
                    required=False)
    parser.add_argument('-a', '--alpha', type=float, help='Alpha value to use',
                      default=1.05, 
                      required=False)
    parser.add_argument('-f', '--fairness-thresh', type=float,
                      help='Fairness threshold value to use',
                      default=0.8, 
                    required=False)
    parser.add_argument('-s', '--sensitive-attrs', type=str,
                      help="Sensitive attributes, e.g., '['sex', 'race']', or '['Sex', 'ForeignWorker']'",
                      default = sensitive_attrs_default,
                      required=False)
    parser.add_argument('--eval-file-output', type=str,
                       help='Filename where to write out evaluation data (default: stdout)',
                       default=None,
                      required=False)
    # Dataset arg:
    parser.add_argument('-i', '--dataset', type=str,
                        help='Choice of dataset (or subset)',
                        default=dataset_default,
                        required=False)
    # Random forest arg:
    parser.add_argument('-t', '--forest', type=int,
                        help='Random forest with a specific number of trees',
                        default=None,
                        required=False)
    return parser


class EvalUtil:
    def __init__(self, args):
        self.args = args
        self.vals = {}

        # Set python's random seed.
        random.seed(args.random_seed)
        self.record_eval("random-seed", args.random_seed)

    def get_forest_size(self):
        return self.args.forest
    
    def get_seed(self):
        return self.args.random_seed

    def get_alpha(self):
        return self.args.alpha

    def get_fairness_thresh(self):
        return self.args.fairness_thresh

    def get_sensitive_attrs(self):
        return self.args.sensitive_attrs
    
    def get_file_name(self):
        return self.args.eval_file_output
    
    def get_dataset(self):
        return self.args.dataset

    def get_forest(self):
        return self.args.forest

    def record_eval(self, metric_name, value):
        self.vals[metric_name] = value
        print("EVAL {} {}".format(metric_name, value))

    def save_vals(self):
        '''Saves self.vals in pickled format to the file pointed to
        args.eval_file_output. This function should be called at the
        end of the program run.
        '''
        fname = self.args.eval_file_output
        if fname is not None:
            outf = open(fname,'wb')
            pickle.dump(self.vals, outf)
            outf.close()
        return

    @staticmethod
    def restore_vals(fname):
        '''Reads the pickled vals dictionary from a file with name fname and
        returns it.
        '''
        infile = open(fname,'rb')
        vals = pickle.load(infile)
        infile.close()
        return vals
    
