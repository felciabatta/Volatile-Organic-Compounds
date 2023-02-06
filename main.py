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

component_files = ['Results/'+voc+'_components_fittedby_D.csv' for voc in VOCS]
component_frames = [pd.read_csv(file, index_col=0) for file in component_files]
C = np.empty((len(VOCS), len(VOCS)))
for i, df1 in enumerate(component_frames):
    for j, df2 in enumerate(component_frames):
        cols = ['ds', 'trend']
        C[i, j], _ = data.pct_corr(df1[cols], df2[cols], log=True)

sns.heatmap(C, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=VOCS, yticklabels=VOCS)
