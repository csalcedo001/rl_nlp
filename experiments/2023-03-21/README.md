# Benchmark: MineDojo vs. MineRL

## Installation

MineDojo and MineRL require different versions of gym that are incompatible with one another. You will have to install their dependencies in two different environments. To install with conda do

```
conda env create --file environment_minedojo.yaml       # default environment name: minedojo
conda env create --file environment_minerl.yaml         # default environment name: minerl
```

then activate the corresponding environments depending on which environment you want to use. Finally, create a `.env` file from `.env.template` and define your W&B entity and project.

```
WANDB_ENTITY=
WANDB_PROJECT=
```

## Running script

Use `rollout.py` to run the tests. The script supports the following CLI arguments:
* framework: either minerl or minedojo. Required option.
* episodes: how many episodes to run in total (number of calls to env.reset). Default: 100.
* iterations: for a given episode, how many iterations to run in total before calling env.reset manually. Default: 1000.

Note: If you are running in a headless system, remember to prepend the command `xvfb-run` to the Python command.

### MineDojo

```
conda activate minedojo
python rollout.py --framework minedojo
```

### MineRL

```
conda activate minerl
python rollout.py --framework minerl
```