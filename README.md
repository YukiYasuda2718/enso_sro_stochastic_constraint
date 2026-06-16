# enso_sro_stochastic_constraint

This repository contains all data and source code used in "A Stochastic-Thermodynamic Constraint on the Seasonal Phase Locking of the El Niño–Southern Oscillation" by Yuki Yasuda and Tsubasa Kohyama.

## Build Environments

### Linux (including WSL2)

1. Install VSCode
2. Install `Remote Development` extension
3. Start a devcontainer (select `info_ao_dyn_pytorch`) through `Dev Containers: Rebuild and Reopen in Container` on your command palette.

- Note: We ran all experiments on the `info_ao_dyn_pytorch_gpu` container.
  - But, GPUs were not used in these experiments.
  - This container, `info_ao_dyn_pytorch_gpu`, is included just for reference.
  - The `info_ao_dyn_pytorch` container is for the CPU environment.

### Mac (only for apple silicon)

1. Install VSCode
2. Install `Remote Development` extension
3. Start a devcontainer (select `info_ao_dyn_mac`) through `Dev Containers: Rebuild and Reopen in Container` on your command palette.

## Run Experiments

1. Make data using [make_data.ipynb](./python/notebooks/make_data.ipynb)
   - The time length of simulations (`t_max`) is set to 10_000 yr.
     - The data size for the 10_000-yr case is about 1.3 GB (per one parameter set, e.g., KA21)
   - To reproduce our results, set `t_max` to 50_000 yr.
     - See the comments in that notebook.
2. Make figures using [make_figures.ipynb](./python/notebooks/make_figures.ipynb)
