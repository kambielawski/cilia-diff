import argparse
import os

from robot import Robot, RobotType
from diff_control import DiffControl

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str)
parser.add_argument('exp_name', type=str)
args = parser.parse_args()

# Read the experiment file into exp_arms variable
exp_file = open(args.exp_file)
exp_string = exp_file.read()
exp_arms = eval(exp_string)
exp_file.close()

robot_type_map = {
    'ANTHROBOT': RobotType.ANTH,
    'ANTH_SPHERE': RobotType.ANTH_SPHERE,
    'CILIA_BALL': RobotType.CILIA_BALL,
    'SOLID': RobotType.SOLID,
    'FLUIDISH': RobotType.FLUIDISH
}

def main():
    os.makedirs(f'./experiments/{args.exp_name}', exist_ok=True)
    os.system(f'mv {args.exp_file} ./experiments/{args.exp_name}')
    
    for arm in exp_arms:
            # Create experiments directory if it doesn't already exist
            os.makedirs(f'./experiments/{args.exp_name}/{arm}', exist_ok=True)
            
            experiment_parameters = exp_arms[arm]
            body_type = robot_type_map[experiment_parameters['body_type']]
            
            rbt = Robot(robot_type=body_type, experiment_parameters=experiment_parameters)
            dc = DiffControl(savedata_folder=f'./experiments/{args.exp_name}/{arm}', experiment_parameters=experiment_parameters)
            dc.init(rbt)
            dc.run(experiment_parameters['iters'])
            dc.pickle_positions(f'{arm}_positions.pkl')

if __name__ == '__main__':
    main()