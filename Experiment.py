import time
import pickle
import os
import numpy as np

from Trial import Trial


class Experiment:
    """Experiment class
    - Initialized when an experiment is submitted
    - Creates and manages experiment directory
    """
    def __init__(self, experiment_directory='.', exp_file=None):
        if experiment_directory: # Continue existing experiment
            self.pickle_file = f'{experiment_directory}/experiment.pkl'
            self.experiment_directory = experiment_directory

            with open(self.pickle_file, 'rb') as pkl:
                self.experiment_object = pickle.load(pkl, 'rb')
                self.experiment_directory = self.experiment_object['experiment_directory']
                self.trials = self.experiment_object['trials']
                self.experiment_params = self.experiment_object['experiment_parameters']

        else: # Initialize a new experiment
            self.Initialize_Directory(exp_file)

    def Run_Local(self):
        """Run the different runs until we max out"""
        for _, trial in self.trials.items():
            trial.Run()

    def Run_Vacc(self):
        # Submit a job for each trial...
        for _, trial in self.trials.items():
            os.system('sbatch run_trial_vacc_bind.sh ' + trial.pickle_file)

    def Initialize_Directory(self, exp_file):
          """Initialize a new experiment directory and populate it with the necessary files and directories."""
          # 1. Create a new experiment directory
          experiment_parameters = self.Get_Experiment_Parameters(exp_file)
          exp_name = experiment_parameters['name']
          self.n_runs = experiment_parameters['n_trials']
          
          timestr = time.strftime('%b%d_%I_%M')
          
          self.experiment_directory = f'experiments/{timestr}_{exp_name}'
          os.system(f'mkdir {self.experiment_directory}')
          os.system(f'mkdir {self.experiment_directory}/plots')

          # 2. Print experiment parameters to file
          param_file = open(f'{self.experiment_directory}/exp_file.txt', 'w')
          param_file.write(str(experiment_parameters) + '\n')
          param_file.close()

          # 3. Copy experiment parameters for 1 trial into trial directories
          exp_params_1trial = experiment_parameters
          exp_params_1trial['n_trials'] = 1
          for run_idx in range(self.n_runs):
              os.system(f'mkdir {self.experiment_directory}/trial_{run_idx}')
              param_file_1trial = open(f'{self.experiment_directory}/trial_{run_idx}/params_trial_{run_idx}.txt', 'w')
              param_file_1trial.write(str(exp_params_1trial) + '\n')
              param_file.close()
          
          # 4. Initialize n_runs Trial objects and pickle them into an aggregate trial directories
          self.trials = { run_idx: Trial(run_idx, self.experiment_directory, exp_params_1trial) for run_idx in range(self.n_runs) }
          self.experiment_params = experiment_parameters
          experiment_object = {
              'trials': self.trials,
              'experiment_parameters': experiment_parameters,
              'experiment_directory': self.experiment_directory
          }
          with open(f'{self.experiment_directory}/experiment.pkl', 'wb') as pkl:
              pickle.dump(experiment_object, pkl)
      
    def Get_Experiment_Parameters(self, experiment_file):
        """Read in the experiment parameters from a file."""
        expfile = open(experiment_file)
        exp_string = expfile.read()
        exp_params = eval(exp_string)
        expfile.close()
        return exp_params

