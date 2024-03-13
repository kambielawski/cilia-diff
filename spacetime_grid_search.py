import os

import numpy as np

param_template = {
    'name': 'grid_search',
    'n_trials': 1,
    'iters': 200,
    'dt': None,
    'grid_res': None,
    'gravity': 5,
    'actuation_omega': 40, 
    'actuation_strength': 5,
    'sim_body_size': 0.1,
    'learning_rate': 30,
    'body_type': 'CILIA_BALL',
    'actuation_axes': [2]
}

# Grid search over timestep size (dt) and sim_body_size
dt_values = np.linspace(0.0002,0.002,10)
grid_res_values = np.linspace(10, 100, 10)

print(dt_values, grid_res_values)

# Create a new experiment file for each combination of dt and sim_body_size
for dt in dt_values:
    for grid_res in grid_res_values:
        param_template['dt'] = dt
        param_template['grid_res'] = grid_res
        param_template['name'] = f'grid_search_dt_{dt}_grid_res_{grid_res}'

        # Create experiment file
        with open(f'dt_{dt}_grid_res_{grid_res}.exp', 'w') as exp_file:
            exp_file.write(str(param_template) + '\n')
            
        # Run the experiment
        # This should launch a single trial with the specified parameters on the VACC
        os.system(f'python run_exp.py --exp dt_{dt}_grid_res_{grid_res}.exp --vacc')
