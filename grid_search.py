import os

import numpy as np

param_template = {
    'name': 'grid_search',
    'n_trials': 1,
    'iters': 1000,
    'dt': None,
    'gravity': 5,
    'actuation_omega': 40, 
    'actuation_strength': 5,
    'sim_body_size': None,
    'learning_rate': 30,
    'body_type': 'ANTHROBOT',
    'actuation_axes': [2]
}

# Grid search over timestep size (dt) and sim_body_size
dt_values = np.linspace(1e-3, 1e-2, 10)
sim_body_size_values = np.linspace(0.05, 0.5, 10)

# Create a new experiment file for each combination of dt and sim_body_size
for dt in dt_values:
    for sim_body_size in sim_body_size_values:
        param_template['dt'] = dt
        param_template['sim_body_size'] = sim_body_size
        param_template['name'] = f'grid_search_dt_{dt}_sim_body_size_{sim_body_size}'

        # Create experiment file
        with open(f'dt_{dt}_sim_body_size_{sim_body_size}.exp', 'w') as exp_file:
            exp_file.write(str(param_template) + '\n')
            
        # Run the experiment
        # This should launch a single trial with the specified parameters on the VACC
        os.system(f'python run_exp.py --exp dt_{dt}_sim_body_size_{sim_body_size}.exp --vacc')
