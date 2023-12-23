from robot import Robot, RobotType
from diff_control import DiffControl, TrajFollow

experiment_parameters = {
    'iters': 5000,
    'dt': 1e-3,
    'gravity': 5,
    'actuation_omega': 40,
    'actuation_strength': 20,
    'sim_body_size': 0.1,
    'learning_rate': 30,
    'body_type': 'ANTHROBOT',
    'opt_problem': 'BOTH',
    'mass_fluid_default': 4,
    'mass_solid_default': 1,
    'E_default': 50,
    'nu_default': 0.1,
        }

def main():
    rbt = Robot(robot_type=RobotType.ANTH, experiment_parameters=experiment_parameters)
    dc = TrajFollow("in_vitro_data/cleaned_data/Run8bot4.csv", experiment_parameters=experiment_parameters)
    dc.init(rbt)
    dc.run(experiment_parameters['iters'])
    dc.visualize_actuation()

if __name__ == '__main__':
    main()