# %% Initialise Packages
from DataSelect import vocData, VOCS, unpickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# initialise vocData
data = vocData()

# %% FIT DATA and EXPORT
for voc in VOCS:
    data.fit(voc=voc, log=True, save=True, groupby='H');


# %% EXTRACT COMPONENTS and MODELS
seasonality = 'daily'
time_unit = 'H'

component_files = ['Results/'+voc+'_components_fittedby_'+time_unit+'.csv' for voc in VOCS]
component_frames = [pd.read_csv(file, index_col=0, parse_dates=[1]) for file in component_files]

model_files = ['Results/'+voc+'_model_fittedby_'+time_unit+'.pkl' for voc in VOCS]
models = [unpickle(file) for file in model_files]

# %% CORRELATE from COMPONENTS
C = np.empty((len(VOCS), len(VOCS)))
for i, df1 in enumerate(component_frames):
    for j, df2 in enumerate(component_frames):
        cols = ['ds', seasonality]
        C[i, j], _ = data.pct_corr(df1[cols], df2[cols], log=True, pct_freq='H')

# sort rows & columns
sort_index = np.argsort(np.nanmean(C, axis=1)).tolist()
C_sorted = C[sort_index, :][:, sort_index]
VOCS_sorted = np.array(VOCS)[sort_index].tolist()
# sns.heatmap(C, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=VOCS, yticklabels=VOCS)
# plt.figure()
sns.heatmap(C_sorted, cmap='coolwarm', vmin=-1, vmax=1, yticklabels=VOCS_sorted, xticklabels=[])
plt.title(seasonality+" correlations")
plt.savefig("Figures/pct_corrs/pctcorr_"+seasonality+"by_"+time_unit+"_sorted.pdf")


# %% MAKE PROPHET PLOTS from MODELS
for i, m in enumerate(models):
    plot_fit = m.plot(component_frames[i])
    plt.title(VOCS[i])
    plt.tight_layout()
    
    plot_components = m.plot_components(component_frames[i])
    plot_components.axes[0].set_title(VOCS[i])
    plt.tight_layout()

    prefix = 'Figures/Regression/'+VOCS[i]
    suffix = '_fittedby_'+time_unit+'.pdf'
    plot_fit.figure.savefig(prefix+'_yhat'+suffix)
    plot_components.figure.savefig(prefix+'_components'+suffix)
