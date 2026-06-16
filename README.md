# enso_sro_stochastic_constraint

This repository contains all data and source code used in "A Stochastic-Thermodynamic Constraint on the Seasonal Phase Locking of the El Niño–Southern OscillationA Stochastic-Thermodynamic Constraint on the Seasonal Phase Locking of the El Niño–Southern Oscillation" by Yuki Yasuda and Tsubasa Kohyama.

## Build Environments

### Linux (including WSL)

### Mac (only for apple silicon)

## Run Experiments

1. Make data using [make_data.ipynb](./python/notebooks/make_data.ipynb)
   - The time length of simulations is set to 10_000 yr.
     - The total size for the 10_000-yr case is about 24 GB
   - To reproduce our results, set it to 50_000 yr.
     - See the comments in that notebook.
2. Make figures using [make_figures.ipynb](./python/notebooks/make_figures.ipynb)
