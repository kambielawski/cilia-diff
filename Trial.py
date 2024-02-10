import pickle

from diff_control import DiffControl
from robot import Robot, RobotType


class Trial:
    """Trial class
    - Initialized when an experiment is submitted
    - Creates and manages experiment directory
    - Has access to the AFPO-level pickle file
    """
    def __init__(self, run_idx, experiment_directory, experiment_parameters):
        self.run_idx = run_idx
        self.experiment_directory = experiment_directory
        self.experiment_parameters = experiment_parameters
        self.trial_directory = self.experiment_directory + f'/trial_{run_idx}'
        self.pickle_file = f'{self.trial_directory}/trial_{run_idx}.pkl'

        # Create trial pickle file for Trial object (self)
        with open(self.pickle_file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def Run(self):
        """Run a single instance of optimization"""
        rbt = Robot(robot_type=RobotType.ANTH, experiment_parameters=self.experiment_parameters)
        dc = DiffControl(experiment_parameters=self.experiment_parameters, savedata_folder=self.trial_directory)
        dc.init(rbt)
        dc.run(self.experiment_parameters['iters'])
        dc.pickle_act(f'trial{self.run_idx}_act.pkl')
        dc.pickle_positions(f'trial{self.run_idx}_pos.pkl')
