# enso_sro_stochastic_constraint

This repository contains all data and source code used in "A Stochastic-Thermodynamic Constraint on the Seasonal Phase Locking of the El Niño–Southern Oscillation" by Yuki Yasuda and Tsubasa Kohyama.

## Build Environments

### Linux (including WSL2)

1. Install VSCode
2. Install `Remote Development` extension
3. Start a devcontainer (select `info_ao_dyn_pytorch`) through `Dev Containers: Rebuild and Reopen in Container` in your command palette.

- Note: We ran all experiments on the `info_ao_dyn_pytorch_gpu` container.
  - However, GPUs were not used in these experiments.
  - This container, `info_ao_dyn_pytorch_gpu`, is included just for reference.
  - The `info_ao_dyn_pytorch` container is for the CPU environment.

### Mac (only for Apple silicon)

1. Install VSCode
2. Install `Remote Development` extension
3. Start a devcontainer (select `info_ao_dyn_mac`) through `Dev Containers: Rebuild and Reopen in Container` in your command palette.

## Run Experiments

- To reproduce our results, set `t_max` to 50_000 yr.
  - By default, the time length of simulations (`t_max`) is set to 10_000 yr.
  - See the comments in the following notebooks.
1. Make data using [make_data.ipynb](./python/notebooks/make_data.ipynb)
   - The data size for the 10_000-yr (50_000-yr) case is about  6 GB (30 GB) per parameter set (e.g., KA21)
2. Make figures using [make_figures.ipynb](./python/notebooks/make_figures.ipynb)
