# fearLearning_coupling

Analysis scripts for the fear learning coupling paper. We examine how amygdala–hippocampus and amygdala–vmPFC functional coupling relate to prediction errors during a fear conditioning paradigm, across three groups: Healthy Controls (HC), Combat Controls (VCC), and veterans with PTSD (VPTSD).

## Repository Structure

```
fearLearning_coupling/

├── analysis_script.py             # Main coupling × group PyMC models
├── sensitivity_stuff.py           # Sensitivity analysis (alternative dataset)
├── scr_Pymc.ipynb                 # SCR-based reinforcement learning models
├── table1_demographics.py         # Demographic Table 1 generator
├── motion_fd_conditions_pymc.py   # Framewise displacement QC check
└── utils.py                       # Shared plotting utilities (plot_pd)
```

## Analysis Scripts

### `analysis_script.py`
The primary analysis script. Fits two Bayesian hierarchical linear regression models using PyMC:

1. **Amygdala–Hippocampus model** — predicts trial-level prediction error (PE) from amygdala–hippocampus coupling, amygdala activation, trial number, and group, with a group × coupling interaction term.
2. **Amygdala–vmPFC model** — identical structure but uses amygdala–vmPFC coupling as the predictor.

Both models include per-subject random intercepts (non-centered parameterization) and use treatment coding with HC as the reference group. Post-hoc group-specific slopes are extracted from the posterior and visualised as panel plots (A–F layout, saved as PNGs).

- **Input:** `data/scr_brain_group.csv`
- **Outputs:** `amg_hipp_coupling_panel.png`, `amg_vmpfc_coupling_panel.png`, `coupling_slopes_panel_A-F_fixed_xlim.png`

---

### `sensitivity_stuff.py`
Mirrors the structure of `analysis_script.py` but runs on an alternative dataset and merges in demographic covariates. Used to confirm that main results are robust to different data-processing choices. Produces identical model structures (amygdala–hippocampus and amygdala–vmPFC) and the same post-hoc slope plots, without saving figures to disk by default.

- **Input:** `data/scr_amg_hipp_all_PH.csv`, `data/demographic.csv`, `data/coupling_hrf0.csv`, `data/coupling_hrf2.csv`

---

### `scr_Pymc.ipynb`
Notebook implementing reinforcement learning (RL) models fitted to skin conductance responses (SCR). Compares three learning models:

- **Rescorla–Wagner (RW)** — standard delta-rule with a single learning rate `α` and softmax inverse temperature `β` (hierarchical, partially pooled).
- **Counterfactual** — separate learning rates for chosen and unchosen stimuli.
- **Pearce–Hall** — attention-modulated learning rate that scales with prediction error magnitude.

Models are compared with `az.compare` (ELPD-LOO). Subject-level expected values and prediction errors extracted from the winning model are merged with fMRI coupling data to create the analysis dataset used in `analysis_script.py`.

- **Input:** raw SCR trial-level data, amygdala–hippocampus FC data
- **Output:** `data/scr_amg_hipp_all.csv`, `outputs/az_compare.html/.docx`

---

### `table1_demographics.py`
Generates a publication-ready Table 1 (demographic characteristics) overall and by group (HC, Combat Controls, PTSD). Reports N, age (mean ± SD and median [IQR]), and gender distribution. Optionally filters the demographics file to only subjects present in a given analysis dataset.

Exports to **CSV**, **Markdown**, **LaTeX**, and optionally **Word (.docx)** (requires `python-docx`).

- **Input:** `data/demographic.csv`, optionally `data/scr_amg_hipp_all_noShock.csv`
- **Output:** `outputs/table1_demographics.{csv,md,tex,docx}`

---

### `motion_fd_conditions_pymc.py`
A Bayesian quality-control check for head motion. Fits a hierarchical model (subject random intercepts + condition fixed effects) to framewise displacement (FD) values, comparing **Shock** (US) vs. **NoShock** (CS+ without shock, CS−) trials. Ensures that any coupling differences are not confounded by motion.

- **Input:** `data/fdData.csv`
- **Output:** posterior contrast plot (Shock − NoShock FD difference)

---

### `utils.py`
Shared utility functions. `plot_pd()` plots the posterior distribution of an xarray/arviz object and annotates it with the Probability of Direction (PD) and 89% HDI.

## Dependencies

```
pymc >= 5
arviz
numpy
pandas
matplotlib
seaborn
python-docx  # optional, for Word export in table1_demographics.py
```
