import argparse
import pickle

from Trial import Trial

############# ARGUMENT PARSING #############

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, default='', help='File path for pickled Trial object')
args = parser.parse_args()

############# RUN EXPERIMENT(S) #############

with open(args.file, 'rb') as pickle_file:
    trial : Trial = pickle.load(pickle_file)

trial.Run()
