import argparse

from robot import Robot, RobotType
from diff_control import DiffControl

experiment_parameters = {
    'dt': 1e-3
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=5000)
    options = parser.parse_args()
    rbt = Robot(robot_type=RobotType.ANTH)
    dc = DiffControl(experiment_parameters=experiment_parameters)
    dc.init(rbt)
    dc.run(options.iters)