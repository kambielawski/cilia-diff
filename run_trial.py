from robot import Robot, RobotType
from diff_control import DiffControl

experiment_parameters = {
            'iters': 5,
            'dt': 1e-3,
            'gravity': 5,
            'actuation_omega': 40, 
            'actuation_strength': 5,
        }

def main():
    rbt = Robot(robot_type=RobotType.ANTH)
    dc = DiffControl(experiment_parameters=experiment_parameters)
    dc.init(rbt)
    dc.run(experiment_parameters['iters'])
    dc.visualize_actuation()

if __name__ == '__main__':
    main()