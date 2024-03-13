import os

param_template = {
    'name': 'example_exp',
    'n_trials': 1,
    'iters': 1000,
    'dt': 1e-3,
    'grid_res': 96,
    'gravity': 5,
    'actuation_omega': 40, 
    'actuation_strength': 10,
    'sim_body_size': 0.1,
    'learning_rate': 100,
    'body_type': 'CILIA_BALL',
    'n_sin_wave': None,
    'actuation_axes': [2]
}

# Grid search over timestep size (dt) and sim_body_size
n_sin_waves_grid = list(range(1, 17))

print(n_sin_waves_grid)

# Create a new experiment file for each combination of dt and sim_body_size
for n_sin_waves in n_sin_waves_grid:
    param_template['name'] = f'actuation_{n_sin_waves}sinwaves'
    param_template['n_sin_wave'] = n_sin_waves

    # Create experiment file
    with open(f'actuation_{n_sin_waves}sinwaves.exp', 'w') as exp_file:
        exp_file.write(str(param_template) + '\n')
        
    # Run the experiment
    # This should launch a single trial with the specified parameters on the VACC
    os.system(f'python run_exp.py --exp actuation_{n_sin_waves}sinwaves.exp --vacc')
