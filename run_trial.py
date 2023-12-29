from robot import Robot, RobotType
from diff_control import DiffControl

experiment_parameters = {
            'iters': 5,
            'dt': 1e-3,
            'gravity': 5,
            'actuation_omega': 10, 
            'actuation_strength': 1,
            'sim_body_size': 0.1,
            'learning_rate': 20,
            'body_type': 'ANTHROBOT',
            'actuation_axes': [0,1]
        }

folder = './experiments/test_2axes'

def main():
    rbt = Robot(robot_type=RobotType.ANTH, experiment_parameters=experiment_parameters)
    dc = DiffControl(experiment_parameters=experiment_parameters, savedata_folder=folder)
    dc.init(rbt)
    dc.run(experiment_parameters['iters'])
    dc.pickle_act(f'test_2axes_act.pkl')
    dc.pickle_positions(f'test_2axes_pos.pkl')
    

if __name__ == '__main__':
    main()