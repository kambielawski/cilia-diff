# Simulating Anthrobots using TaiChi

## Specifying an Experiment

Make an experiment file structured like this:
```
{
    'exp1': 
        {
            'iters': 5000,
            'dt': 1e-3,
            'gravity': 5,
            'actuation_omega': 40, 
            'actuation_strength': 5,
        }
}
```

The above file specifies an experiment with one experiment arm: "exp1". 

## Running the Experiment Locally

To run the experiment locally, 

```
python3 run_exp.py <experiment_file>
```

This will create an `experiments` directory, a subdirectory for the experiment, and a subsubdirectory for each of the experiment arms. This is where the experiment results will be placed. 

## Running on the VACC

To run the experiment on the VACC, 

```
sbatch vacc_submit_exp.sh <experiment_file> <experiment_name>
```

**This will spawn a single VACC job for every trial specified in the experiment.** In the example above, 10 VACC jobs will be spawned (5 trials for each experiment arm). Be wary of how many resources you're using.

Launch a shell on a DeepGreen GPU node
```
srun --partition=dggpu --nodes=1 --ntasks=1 --gpus=1 --job-name=ciliadiff --time=01:00:00 --pty /bin/bash
```