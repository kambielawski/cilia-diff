import argparse
import os

from robot import Robot, RobotType
from diff_control import DiffControl

# Create experiments directory if it doesn't already exist
if not os.path.exists('./experiments'):
    os.system('mkdir experiments')

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

def main():
    for arm in exp_arms:
            # Create experiments directory if it doesn't already exist
            if not os.path.exists(f'./experiments/{args.exp_name}/{arm}'):
                os.system(f'mkdir ./experiments/{args.exp_name}/{arm}')
            
            experiment_parameters = exp_arms[arm]
            rbt = Robot(robot_type=RobotType.ANTH)
            dc = DiffControl(save_folder=f'./experiments/{args.exp_name}/{arm}', experiment_parameters=experiment_parameters)
            dc.init(rbt)
            dc.run(experiment_parameters['iters'])
            dc.visualize_actuation()

if __name__ == '__main__':
    main()