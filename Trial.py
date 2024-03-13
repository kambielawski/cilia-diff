import pickle

from diff_control import DiffControl
from robot import Robot, RobotType

# class RobotType(Enum):
#     ANTH = 1
#     ANTH_SPHERE = 2
#     CILIA_BALL = 3
#     SOLID = 4
#     FLUIDISH = 5


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

    def robot_str_to_enum(self, robot_type_str):
        if robot_type_str == 'ANTH':
            return RobotType.ANTH
        elif robot_type_str == 'ANTH_SPHERE':
            return RobotType.ANTH_SPHERE
        elif robot_type_str == 'CILIA_BALL':
            return RobotType.CILIA_BALL
        elif robot_type_str == 'SOLID':
            return RobotType.SOLID
        elif robot_type_str == 'FLUIDISH':
            return RobotType.FLUIDISH

    def Run(self):
        """Run a single instance of optimization"""
        robot_type_str = self.experiment_parameters['body_type']
        robot_type = self.robot_str_to_enum(robot_type_str)
        
        rbt = Robot(robot_type=robot_type, experiment_parameters=self.experiment_parameters)
        dc = DiffControl(experiment_parameters=self.experiment_parameters, savedata_folder=self.trial_directory)
        dc.init(rbt)
        dc.run(self.experiment_parameters['iters'])
        # dc.pickle_act(f'trial{self.run_idx}_act.pkl')
        # dc.pickle_positions(f'trial{self.run_idx}_pos.pkl')
        dc.pickle_loss(f'trial{self.run_idx}_loss.pkl')
        dc.save_weights(self.experiment_parameters['iters'])
