#%% [markdown]
# Statistical analysis of the data
# ---
# In this section, we will perform a statistical analysis of the data for both amygdala-hippocampus and amygdala-PFC coupling


#%% Loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az 


def plot_slope_panel_3x2(
    *,
    left_slopes: dict,
    right_slopes: dict,
    row_order: list,
    row_display_names: dict,
    left_title: str,
    right_title: str,
    suptitle: str,
    outpath: str,
    figsize=(11, 8.5),
    dpi=600,
    row_label_x=-0.02,
    pad_inches=0.2,
):
    """
    Create a 3x2 panel (A–F) of posterior slope distributions.

    `left_slopes`/`right_slopes` map group codes (e.g., 'HC') -> slope posterior (xarray / arviz).
    """
    import matplotlib.pyplot as plt
    from utils import plot_pd

    fig, axes = plt.subplots(
        nrows=len(row_order),
        ncols=2,
        figsize=figsize,
        sharey=False,
        constrained_layout=True,
    )

    letters = "ABCDEF"
    li = 0

    for r, g in enumerate(row_order):
        display = row_display_names.get(g, g)

        ax_l = axes[r, 0]
        plot_pd(left_slopes[g], display, ax=ax_l, title_prefix="", show_title=False)
        ax_l.text(
            -0.12,
            1.05,
            letters[li],
            transform=ax_l.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
        li += 1

        ax_r = axes[r, 1]
        plot_pd(right_slopes[g], display, ax=ax_r, title_prefix="", show_title=False)
        ax_r.text(
            -0.12,
            1.05,
            letters[li],
            transform=ax_r.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
        li += 1

    # Column titles (set after plotting so they aren't overwritten)
    axes[0, 0].set_title(left_title, fontsize=13, pad=14)
    axes[0, 1].set_title(right_title, fontsize=13, pad=14)

    # Row labels (group names) on the left side of the grid
    for r, g in enumerate(row_order):
        display = row_display_names.get(g, g)
        pos = axes[r, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            row_label_x,
            y_mid,
            display,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    return fig


def plot_slope_panel_3x2_fixed_xlim(
    *,
    left_slopes: dict,
    right_slopes: dict,
    row_order: list,
    row_display_names: dict,
    left_title: str,
    right_title: str,
    suptitle: str,
    outpath: str,
    xlim=(-0.2, 0.2),
    figsize=(11, 8.5),
    dpi=600,
    row_label_x=-0.02,
    pad_inches=0.2,
):
    """
    Same as `plot_slope_panel_3x2`, but with a fixed x-axis range so 0 is
    always in a consistent location across all panels.
    """
    import matplotlib.pyplot as plt
    from utils import plot_pd

    fig, axes = plt.subplots(
        nrows=len(row_order),
        ncols=2,
        figsize=figsize,
        sharey=False,
        constrained_layout=True,
    )

    letters = "ABCDEF"
    li = 0

    for r, g in enumerate(row_order):
        display = row_display_names.get(g, g)

        ax_l = axes[r, 0]
        plot_pd(left_slopes[g], display, ax=ax_l, title_prefix="", show_title=False)
        ax_l.set_xlim(*xlim)
        ax_l.text(
            -0.12,
            1.05,
            letters[li],
            transform=ax_l.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
        li += 1

        ax_r = axes[r, 1]
        plot_pd(right_slopes[g], display, ax=ax_r, title_prefix="", show_title=False)
        ax_r.set_xlim(*xlim)
        ax_r.text(
            -0.12,
            1.05,
            letters[li],
            transform=ax_r.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
        li += 1

    # Column titles (set after plotting so they aren't overwritten)
    axes[0, 0].set_title(left_title, fontsize=13, pad=14)
    axes[0, 1].set_title(right_title, fontsize=13, pad=14)

    # Row labels (group names) on the left side of the grid
    for r, g in enumerate(row_order):
        display = row_display_names.get(g, g)
        pos = axes[r, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            row_label_x,
            y_mid,
            display,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    return fig

# %%
# read file
df = pd.read_csv('data/scr_brain_group.csv')
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

# Check which group is reference (index 0)
print("Group coding (0 = reference):", {g: i for i, g in enumerate(df['group'].cat.categories)})

# Extract variables
pe = df['pe'].values
coupling = df['coupling'].values
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
# %% run panel pd
fig, axes = plt.subplots(3, 1, figsize=(5, 15), sharey=True)

plot_pd(slope_hipp_HC, "Healthy Controls", ax=axes[0])
plot_pd(slope_hipp_VCC, "Combat Controls", ax=axes[1])
plot_pd(slope_hipp_VPTSD, "PTSD", ax=axes[2])

fig.suptitle("Posterior distributions of coupling slopes by group", y=1.02)
fig.tight_layout()
plt.savefig('amg_hipp_coupling_panel.png', dpi=600)
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
# %% panel plot of the slopes
from utils import plot_pd
fig, axes = plt.subplots(3, 1, figsize=(5, 15), sharey=True)

plot_pd(slope_vmpfc_HC, "Healthy Controls", ax=axes[0])
plot_pd(slope_vmpfc_VCC, "Combat Controls", ax=axes[1])
plot_pd(slope_vmpfc_VPTSD, "PTSD", ax=axes[2])
fig.suptitle("Posterior distributions of coupling slopes by group", y=1.02)
fig.tight_layout()
plt.savefig('amg_vmpfc_coupling_panel.png', dpi=600)
plt.show()

# %% combined A–F style panel (3 rows x 2 columns)
left_slopes = {"HC": slope_hipp_HC, "VCC": slope_hipp_VCC, "VPTSD": slope_hipp_VPTSD}
right_slopes = {"HC": slope_vmpfc_HC, "VCC": slope_vmpfc_VCC, "VPTSD": slope_vmpfc_VPTSD}

row_order = ["HC", "VCC", "VPTSD"]
row_display_names = {"HC": "Healthy Controls", "VCC": "Combat Controls", "VPTSD": "PTSD"}

plot_slope_panel_3x2_fixed_xlim(
    left_slopes=left_slopes,
    right_slopes=right_slopes,
    row_order=row_order,
    row_display_names=row_display_names,
    left_title="Amygdala–Hippocampus",
    right_title="Amygdala–vmPFC",
    suptitle="Posterior distribution of coupling slopes by group",
    outpath="coupling_slopes_panel_A-F_fixed_xlim.png",
    figsize=(12, 9),
    dpi=600,
)
# %% Compare contrasts of slopes between groups

# Amygdala-vmPFC: combat-exposed without PTSD vs. with PTSD
diff_vmpfc = slope_vmpfc_VCC - slope_vmpfc_VPTSD
pd_diff = float((diff_vmpfc < 0).mean())  # VCC more negative than PTSD
hdi_diff = az.hdi(diff_vmpfc.values.flatten(), hdi_prob=0.89)
print(f"VCC - VPTSD: mean={float(diff_vmpfc.mean()):.3f}, sd = {float(diff_vmpfc.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")

# Amygdala-hippocampus: combat-exposed without PTSD vs. with PTSD
diff_hipp = slope_hipp_VCC - slope_hipp_VPTSD
pd_diff = float((diff_hipp < 0).mean())  # VCC more negative than PTSD
hdi_diff = az.hdi(diff_hipp.values.flatten(), hdi_prob=0.89)
print(f"VCC - VPTSD: mean={float(diff_hipp.mean()):.3f}, sd = {float(diff_hipp.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")
# %% Contrasting HC vs. PTSD
diff_hc_ptsd = slope_hipp_HC - slope_hipp_VPTSD
pd_diff = float((diff_hc_ptsd < 0).mean())  # HC more negative than PTSD
hdi_diff = az.hdi(diff_hc_ptsd.values.flatten(), hdi_prob=0.89)
print(f"HC - VPTSD: mean={float(diff_hc_ptsd.mean()):.3f}, sd = {float(diff_hc_ptsd.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")
# %% Contrasting HC vs. VCC
diff_hc_vcc = slope_hipp_HC - slope_hipp_VCC
pd_diff = float((diff_hc_vcc < 0).mean())  # HC more negative than VCC
hdi_diff = az.hdi(diff_hc_vcc.values.flatten(), hdi_prob=0.89)
print(f"HC - VCC: mean={float(diff_hc_vcc.mean()):.3f}, sd = {float(diff_hc_vcc.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")

# %% compare contrasts of slopes between HC and PTSD/VCC for vmpfc
diff_vmpfc_ptsd = slope_vmpfc_HC - slope_vmpfc_VPTSD
pd_diff = float((diff_vmpfc_ptsd < 0).mean())  # HC more negative than PTSD
hdi_diff = az.hdi(diff_vmpfc_ptsd.values.flatten(), hdi_prob=0.89)
print(f"HC - VPTSD: mean={float(diff_vmpfc_ptsd.mean()):.3f}, sd = {float(diff_vmpfc_ptsd.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")
diff_vmpfc_vcc = slope_vmpfc_HC - slope_vmpfc_VCC
pd_diff = float((diff_vmpfc_vcc < 0).mean())  # HC more negative than VCC
hdi_diff = az.hdi(diff_vmpfc_vcc.values.flatten(), hdi_prob=0.89)
print(f"HC - VCC: mean={float(diff_vmpfc_vcc.mean()):.3f}, sd = {float(diff_vmpfc_vcc.std()):.3f}, pd={pd_diff*100:.1f}%, 89% HDI={hdi_diff}")

# %%
