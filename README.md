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

---

## Launching on the VACC

Load the module singularity for container launching
```
module load singularity
```

Pull the container you need, found from 
```
singularity pull docker://nvcr.io/nvidia/your_container:version_tag

# e.g.: 
singularity pull docker://nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04
```

Run a shell on an actual DeepGreen node to test sbatch files before submitting them
```
srun --partition=dggpu --nodes=1 --ntasks=1 --gpus=1 --job-name=ciliadiff --time=01:00:00 --pty /bin/bash
```

Make a sandbox from a singularity container (this basically creates a directory on the host machine which has all of the files from the container)
```
singularity build --sandbox taichi-vacc-x11_latest ~/taichi-vacc-x11_latest.sif
```

Get a shell inside of a Singularity container 
```
singularity shell --bind /gpfs1/home:/users --writable ./taichi-vacc-x11_latest
```