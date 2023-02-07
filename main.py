from DataSelect import vocData, VOCS

import pandas as pd
import numpy as np
import seaborn as sns

# initialise vocData
data = vocData()

# %% FIT DATA and EXPORT
for voc in VOCS:
    data.fit(voc=voc, log=True, save=True, groupby='H');


# %% CORRELATE from COMPONENTS

seasonality = 'daily'
time_unit = 'H'

component_files = ['Results/'+voc+'_components_fittedby_'+time_unit+'.csv' for voc in VOCS]
component_frames = [pd.read_csv(file, index_col=0) for file in component_files]
C = np.empty((len(VOCS), len(VOCS)))
for i, df1 in enumerate(component_frames):
    for j, df2 in enumerate(component_frames):
        cols = ['ds', seasonality]
        C[i, j], _ = data.pct_corr(df1[cols], df2[cols], log=True)

# sort rows & columns
sort_index = np.argsort(np.nanmean(C, axis=1)).tolist()
C_sorted = C[sort_index, :][:, sort_index]
VOCS_sorted = np.array(VOCS)[sort_index].tolist()
# sns.heatmap(C, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=VOCS, yticklabels=VOCS)
# plt.figure()
sns.heatmap(C_sorted, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=VOCS_sorted, yticklabels=VOCS_sorted)
