
# %% Initialise Packages
from DataSelect import vocData, VOCS, unpickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use("Figures/LaTeX.mplstyle")

# initialise vocData
data = vocData()

# %% FIT DATA and EXPORT
for voc in VOCS:
    data.fit(voc=voc, log=True, save=True, groupby='W')

# %% EXTRACT COMPONENTS and MODELS
seasonality = 'yearly'
time_unit = 'H'

component_files = ['Results/'+voc+'_components_fittedby_' +
                   time_unit+'.csv' for voc in VOCS]
component_frames = [pd.read_csv(
    file, index_col=0, parse_dates=[1]) for file in component_files]

model_files = ['Results/'+voc+'_model_fittedby_' +
               time_unit+'.pkl' for voc in VOCS]
models = [unpickle(file) for file in model_files]

# %% CORRELATE from COMPONENTS
daterange = [None, None]
C = np.empty((len(VOCS), len(VOCS)))
for i, df1 in enumerate(component_frames):
    for j, df2 in enumerate(component_frames):
        cols = ['ds', seasonality]
        C[i, j], _ = data.pct_corr(
            df1[cols], df2[cols], log=True, pct_freq='H', daterange=daterange)

# sort rows & columns
sort_index = np.argsort(np.nanmean(C, axis=1)).tolist()
C_sorted = C[sort_index, :][:, sort_index]
VOCS_sorted = np.array(VOCS)[sort_index].tolist()
VOC_labels = [voc.title() for voc in VOCS_sorted]
# sns.heatmap(C, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=VOCS, yticklabels=VOCS)
# plt.figure()
map = sns.heatmap(C_sorted, cmap='coolwarm', vmin=-1, vmax=1, yticklabels=VOC_labels,
                  xticklabels=[], cbar=False, linecolor="face", square=True)
map.figure.tight_layout()
map.set_title((seasonality+" correlations").title())
map.tick_params(length=0)
try:
    map.figure.axes[1].tick_params(length=0)
except:
    pass

# %% SAVE CORRMATS
map.figure.savefig("Figures/pct_corrs/pctcorr_" +
                   seasonality+"by_"+time_unit+"_sorted.pdf")


# %% MAKE PROPHET PLOTS from MODELS
for i, m in enumerate(models):
    title = VOCS[i].title()

    # plot fitted model
    plot_fit = m.plot(component_frames[i], figsize=(5, 3))
    plt.title(title)
    plt.xlabel(r"Time, $t$ years", usetex=True)
    plt.ylabel(r"Density in Air, $\log(\mu g m^{-3})$", usetex=True)
    plt.tight_layout()
    # rasterise scatter points
    scatter_data = plot_fit.axes[0].get_children()[0]
    scatter_data.set_rasterized(True)

    # plot components
    plot_components = m.plot_components(
        component_frames[i], figsize=(6*.8, 7*.8))
    axes = plot_components.axes
    axes[0].set_title(title+" Components")

    week_ticks = axes[1].get_xticks()
    week_ticks = np.linspace(week_ticks[0], week_ticks[-1], 8)
    axes[1].set_xticklabels(
        ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    year_ticks = axes[2].get_xticks()
    year_ticks = np.linspace(year_ticks[0], year_ticks[-1], 13)
    axes[2].set_xticks(year_ticks)
    axes[2].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"])

    axes[0].set_xlabel(r"Time, $t$ years", usetex=True)
    axes[1].set_xlabel(r"Day of Week")
    axes[2].set_xlabel(r"Day of Year")

    try:
        day_ticks = axes[3].get_xticks()
        day_ticks = np.linspace(day_ticks[0], day_ticks[-1], 7)
        axes[3].set_xticks(day_ticks)
        axes[3].set_xlabel(r"Hour of Day")
    except:
        pass

    for ax in axes:
        ax.set_ylabel(ax.get_ylabel().title())
    plt.tight_layout()

    # save plots
    prefix = 'Figures/Regression/'+VOCS[i]
    suffix = '_fittedby_'+time_unit+'.pdf'
    plot_fit.figure.savefig(prefix+'_yhat'+suffix)
    plot_components.figure.savefig(prefix+'_components'+suffix)
