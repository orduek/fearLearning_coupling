"""
Simple PyMC script to test motion (framewise displacement; FD) differences across conditions.

Matches the structure you use in `sensitivity_stuff.py`:
read data → encode indices → PyMC model → az.summary → post-hoc contrasts + plot_pd + PD.
"""

#%% Loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

#%% Read file
df = pd.read_csv("data/fdData.csv")

#%% Encode subject and condition indices
df["sub_idx"] = pd.Categorical(df["subject"].astype(str)).codes
n_subs = df["sub_idx"].nunique()

# Map conditions to Shock (US) vs NoShock (CS, CSm)
df["shock_cond"] = df["cond"].map({"CS": "NoShock", "CSm": "NoShock", "US": "Shock"})

# Make ordering explicit and pick reference condition
cond_order = ["NoShock", "Shock"]
df["shock_cond"] = pd.Categorical(df["shock_cond"].astype(str), categories=cond_order, ordered=True)
df["cond_idx"] = df["shock_cond"].cat.codes
n_conds = df["cond_idx"].nunique()

print("Condition coding (0 = reference):", {c: i for i, c in enumerate(df["shock_cond"].cat.categories)})
if n_conds != len(cond_order):
    raise ValueError(f"Expected {len(cond_order)} conditions {cond_order}, but found n_conds={n_conds}.")

#%% Extract variables
fd = df["fd"].astype(float).values
sub_idx = df["sub_idx"].values
cond_idx = df["cond_idx"].values

#%% FD by condition model (condition fixed effects + subject random intercept)
with pm.Model() as model_fd:

    # Condition main effect: treatment coding with reference condition = index 0 ("CS")
    beta_cond_raw = pm.Normal("beta_cond_raw", mu=0, sigma=0.2, shape=n_conds - 1)
    beta_cond = pm.math.concatenate([[0], beta_cond_raw])  # pad ref with 0

    # Hyperpriors for random intercepts (non-centered)
    mu_a = pm.Normal("mu_a", mu=float(np.mean(fd)), sigma=0.2)
    sigma_a = pm.HalfNormal("sigma_a", sigma=0.2)
    z_a = pm.Normal("z_a", mu=0, sigma=1, shape=n_subs)
    a = pm.Deterministic("a", mu_a + z_a * sigma_a)

    # Expected value
    mu = a[sub_idx] + beta_cond[cond_idx]

    # Likelihood
    sigma = pm.HalfNormal("sigma", sigma=0.2)
    y_obs = pm.Normal("fd", mu=mu, sigma=sigma, observed=fd)

    trace_fd = pm.sample(chains=4, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

#%% Summary
az.summary(trace_fd, var_names=["beta_cond_raw", "mu_a", "sigma_a", "sigma"], hdi_prob=0.89)

#%% Post-hoc: condition means (group-level) and contrasts
posterior = trace_fd.posterior

# Group-level mean FD per condition (subject intercept average; i.e., mu_a + beta_cond)
beta_NoShock = 0
beta_Shock = posterior["beta_cond_raw"][:, :, cond_order.index("Shock") - 1]

mean_NoShock = posterior["mu_a"] + beta_NoShock
mean_Shock = posterior["mu_a"] + beta_Shock

def pd_from_xarray(x):
    s = x.values.flatten().astype(float)
    p_pos = (s > 0).mean()
    p_neg = (s < 0).mean()
    return max(p_pos, p_neg), p_pos, p_neg

# Contrasts (differences in FD)
diff_Shock_minus_NoShock = mean_Shock - mean_NoShock

for name, diff in [
    ("Shock - NoShock", diff_Shock_minus_NoShock),
]:
    pd_, p_pos, p_neg = pd_from_xarray(diff)
    print(
        f"{name}: mean={float(diff.mean()):.4f}, SD={float(diff.std()):.4f}, "
        f"HDI89={az.hdi(diff.values.flatten(), hdi_prob=0.89)}, "
        f"PD={pd_*100:.1f}% (P>0={p_pos*100:.1f}%, P<0={p_neg*100:.1f}%)"
    )

#%% Plot posterior distributions of contrasts + PD (same convention as other scripts)
from utils import plot_pd

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_pd(diff_Shock_minus_NoShock, "Shock - NoShock (FD difference)", ax=ax, binwidth=0.002)
fig.suptitle("Posterior distribution of FD contrast", y=1.05)
fig.tight_layout()
plt.show()

