#%% Loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az 


# %%
# read file
df = pd.read_csv('data/scr_amg_hipp_all_noShock.csv') # change to relevant file here
#dem = pd.read_csv('data/demographic.csv')
#dem['sub'] = dem['sub_id'].astype(str)
#df = df.merge(dem, on='sub', how='left')
# %% amygdala-hippocampus coupling pymc model
# Encode 'sub' as integer indices
df['sub_idx'] = pd.Categorical(df['sub']).codes
n_subs = df['sub_idx'].nunique()

# Encode 'group' as integer indices (make ordering explicit!)
# Data uses: HC (healthy controls), VCC (combat controls), VPTSD (PTSD)
group_order = ['HC', 'VCC', 'VPTSD']
df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
df['group_idx'] = df['group'].cat.codes
n_groups = df['group_idx'].nunique()
#%%
# Check which group is reference (index 0)
print("Group coding (0 = reference):", {g: i for i, g in enumerate(df['group'].cat.categories)})

# Extract variables
pe = df['pe'].values
coupling = df['coupling'].values # set relevant coupling
amg = df['amg'].values
trialNo = df['trialNo'].values
sub_idx = df['sub_idx'].values
group_idx = df['group_idx'].values

with pm.Model() as model:
    
    # Fixed effects (main effects)
    beta_coupling = pm.Normal('beta_coupling', mu=0, sigma=1)  # Coupling effect for reference group
    beta_amg = pm.Normal('beta_amg', mu=0, sigma=1)
    beta_trialNo = pm.Normal('beta_trialNo', mu=0, sigma=1)
    
    # Main effect of group (reference group = 0, so n_groups-1 parameters)
    beta_group_raw = pm.Normal('beta_group_raw', mu=0, sigma=1, shape=n_groups - 1)
    beta_group = pm.math.concatenate([[0], beta_group_raw])  # Pad reference group with 0
    
    # Group × Coupling interaction (deviation from reference group's slope)
    beta_interaction_raw = pm.Normal('beta_interaction_raw', mu=0, sigma=1, shape=n_groups - 1)
    beta_interaction = pm.math.concatenate([[0], beta_interaction_raw])  # Pad reference with 0
    
    # Hyperpriors for random intercepts
    mu_a = pm.Normal('mu_a', mu=0, sigma=1)
    sigma_a = pm.HalfNormal('sigma_a', sigma=1)
    
    # Non-centered random intercepts
    z_a = pm.Normal('z_a', mu=0, sigma=1, shape=n_subs)
    a = pm.Deterministic('a', mu_a + z_a * sigma_a)
    
    # Expected value of outcome
    mu = (
        a[sub_idx] +
        beta_group[group_idx] +                        # Main effect of group
        beta_coupling * coupling +                      # Main effect of coupling (reference slope)
        beta_interaction[group_idx] * coupling +        # Interaction: group-specific slope adjustment
        beta_amg * amg +
        beta_trialNo * trialNo
    )
    
    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    y_obs = pm.Normal('pe', mu=mu, sigma=sigma, observed=pe)
    
    trace = pm.sample(chains=4, return_inferencedata=True,
                      idata_kwargs={"log_likelihood": True})


# %% trace
az.summary(trace, var_names=['beta_coupling', 'beta_group_raw', 'beta_interaction_raw'], 
           hdi_prob=0.89)

# %% grab group specific slopes for coupling
# group specific slope

# Post-hoc calculation of group-specific slopes
posterior = trace.posterior
slope_hipp_HC = posterior['beta_coupling']
if n_groups != 3:
    raise ValueError(f"Expected 3 groups {group_order}, but found n_groups={n_groups}.")
slope_hipp_VCC = posterior['beta_coupling'] + posterior['beta_interaction_raw'][:, :, group_order.index('VCC') - 1]
slope_hipp_VPTSD = posterior['beta_coupling'] + posterior['beta_interaction_raw'][:, :, group_order.index('VPTSD') - 1]

print("HC slope (mean, SD, HDI):", float(slope_hipp_HC.mean()), float(slope_hipp_HC.std()), az.hdi(slope_hipp_HC.values.flatten(), hdi_prob=0.89))
print("VCC slope (mean, SD, HDI):", float(slope_hipp_VCC.mean()), float(slope_hipp_VCC.std()), az.hdi(slope_hipp_VCC.values.flatten(), hdi_prob=0.89))
print("VPTSD slope (mean, SD, HDI):", float(slope_hipp_VPTSD.mean()), float(slope_hipp_VPTSD.std()), az.hdi(slope_hipp_VPTSD.values.flatten(), hdi_prob=0.89))

#%% [markdown] 
# Now we build a panel plot showing the posterior distribution of each slope of the interaction
#%% panel plot of the slopes
from utils import plot_pd

#%% run for total coupling
plot_pd(trace.posterior['beta_coupling'], "Total coupling")
# %% run panel pd
fig, axes = plt.subplots(3, 1, figsize=(5, 15), sharey=True)

plot_pd(slope_hipp_HC, "Healthy Controls", ax=axes[0])
plot_pd(slope_hipp_VCC, "Combat Controls", ax=axes[1])
plot_pd(slope_hipp_VPTSD, "PTSD", ax=axes[2])

fig.suptitle("Posterior distributions of coupling slopes by group", y=1.02)
fig.tight_layout()
#plt.savefig('amg_hipp_coupling_panel.png', dpi=600)
plt.show()




# %% [markdown]
# Now we will repeat the analysis for the amygdala-PFC coupling
#%% amygdala-PFC coupling pymc model
# Encode 'sub' as integer indices
df['sub_idx'] = pd.Categorical(df['sub']).codes
n_subs = df['sub_idx'].nunique()

# Encode 'group' as integer indices (keep the same explicit ordering)
df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
df['group_idx'] = df['group'].cat.codes
n_groups = df['group_idx'].nunique()

# Check which group is reference (index 0)
print("Group coding (0 = reference):", {g: i for i, g in enumerate(df['group'].cat.categories)})
# %%
coupling = df['amg_vmpfc'].values
amg = df['amg'].values
trialNo = df['trialNo'].values
sub_idx = df['sub_idx'].values
group_idx = df['group_idx'].values

with pm.Model() as model_vmpfc:
    
    # Fixed effects (main effects)
    beta_coupling = pm.Normal('beta_coupling', mu=0, sigma=1)  # Coupling effect for reference group
    beta_amg = pm.Normal('beta_amg', mu=0, sigma=1)
    beta_trialNo = pm.Normal('beta_trialNo', mu=0, sigma=1)
    
    # Main effect of group (reference group = 0, so n_groups-1 parameters)
    beta_group_raw = pm.Normal('beta_group_raw', mu=0, sigma=1, shape=n_groups - 1)
    beta_group = pm.math.concatenate([[0], beta_group_raw])  # Pad reference group with 0
    
    # Group × Coupling interaction (deviation from reference group's slope)
    beta_interaction_raw = pm.Normal('beta_interaction_raw', mu=0, sigma=1, shape=n_groups - 1)
    beta_interaction = pm.math.concatenate([[0], beta_interaction_raw])  # Pad reference with 0
    
    # Hyperpriors for random intercepts
    mu_a = pm.Normal('mu_a', mu=0, sigma=1)
    sigma_a = pm.HalfNormal('sigma_a', sigma=1)
    
    # Non-centered random intercepts
    z_a = pm.Normal('z_a', mu=0, sigma=1, shape=n_subs)
    a = pm.Deterministic('a', mu_a + z_a * sigma_a)
    
    # Expected value of outcome
    mu = (
        a[sub_idx] +
        beta_group[group_idx] +                        # Main effect of group
        beta_coupling * coupling +                      # Main effect of coupling (reference slope)
        beta_interaction[group_idx] * coupling +        # Interaction: group-specific slope adjustment
        beta_amg * amg +
        beta_trialNo * trialNo
    )
    
    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    y_obs = pm.Normal('pe', mu=mu, sigma=sigma, observed=pe)
    
    trace_vmpfc = pm.sample(chains=4, return_inferencedata=True,
                      idata_kwargs={"log_likelihood": True})

# %%
az.summary(trace_vmpfc, var_names=['beta_coupling', 'beta_group_raw', 'beta_interaction_raw'], 
           hdi_prob=0.89)
# %% grab group specific slopes for coupling
# group specific slope

# Post-hoc calculation of group-specific slopes
posterior = trace_vmpfc.posterior
slope_vmpfc_HC = posterior['beta_coupling']
if n_groups != 3:
    raise ValueError(f"Expected 3 groups {group_order}, but found n_groups={n_groups}.")
slope_vmpfc_VCC = posterior['beta_coupling'] + posterior['beta_interaction_raw'][:, :, group_order.index('VCC') - 1]
slope_vmpfc_VPTSD = posterior['beta_coupling'] + posterior['beta_interaction_raw'][:, :, group_order.index('VPTSD') - 1]

print("HC slope mean (SD, HDI):", float(slope_vmpfc_HC.mean()), float(slope_vmpfc_HC.std()), az.hdi(slope_vmpfc_HC.values.flatten(), hdi_prob=0.89))
print("VCC slope mean (SD, HDI):", float(slope_vmpfc_VCC.mean()), float(slope_vmpfc_VCC.std()), az.hdi(slope_vmpfc_VCC.values.flatten(), hdi_prob=0.89))
print("VPTSD slope mean (SD, HDI):", float(slope_vmpfc_VPTSD.mean()), float(slope_vmpfc_VPTSD.std()), az.hdi(slope_vmpfc_VPTSD.values.flatten(), hdi_prob=0.89))

#%% run for total coupling
plot_pd(trace_vmpfc.posterior['beta_coupling'], "Total coupling")
# %% panel plot of the slopes
from utils import plot_pd
fig, axes = plt.subplots(3, 1, figsize=(5, 15), sharey=True)

plot_pd(slope_vmpfc_HC, "Healthy Controls", ax=axes[0])
plot_pd(slope_vmpfc_VCC, "Combat Controls", ax=axes[1])
plot_pd(slope_vmpfc_VPTSD, "PTSD", ax=axes[2])
fig.suptitle("Posterior distributions of coupling slopes by group", y=1.02)
fig.tight_layout()
#plt.savefig('amg_vmpfc_coupling_panel.png', dpi=600)
plt.show()
# %%
