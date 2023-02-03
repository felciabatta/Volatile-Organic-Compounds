from FbprophetFO import FbProphetCorrelation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataSelect import VOCS


from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from DataSelect import vocData
from prophet.plot import plot_yearly, plot_weekly, plot_seasonality
import numpy as np
import seaborn as sns
  



print(VOCS)

Example = ['Nitric oxide',
        'Nitrogen dioxide',
        'Sulphur dioxide',
        'Carbon monoxide',
        'ethane',
        'ethene',
        'ethyne',
        'propane',
        'propene',
        'iso-butane',
        'n-butane',
        '1-butene',
        'trans-2-butene',
        'cis-2-butene',
        'iso-pentane',
        'n-pentane',
        '1,3-butadiene',
        'trans-2-pentene',
        # 'cis-2-pentene (VOC-AIR only)', # ONLY TO 2005
        '2-methylpentane',
        # '3-methylpentane (VOC-AIR only)', # ONLY TO 2005
        'isoprene',
        'n-hexane',
        'n-heptane',
        'toluene',
        'ethylbenzene',
        'm+p-xylene',
        'o-xylene',
        '1,2,4-trimethylbenzene',
        '1,3,5-trimethylbenzene']
BenzeneCorrelations = []
TimeFrame = ['trend','weekly','yearly']

def ListOfCorrelations():
    for i, voc in enumerate(Example):
            CombinationOne = FbProphetCorrelation('benzene',voc)
            m, components = CombinationOne.fit(log=True, plot=False, groupby='D')
            for j, frame in enumerate(TimeFrame):
                BenTolCorr, pchange = CombinationOne.Correlation(components[0][['ds',frame]], components[1][['ds',frame]], log=True, plot=False)
                BenzeneCorrelations.append(BenTolCorr)
    return BenzeneCorrelations

ListOfCorrelations()
print(BenzeneCorrelations)
heatdata = np.array(BenzeneCorrelations)
fixeddata = np.reshape(heatdata,(28,3))
heatmap = sns.heatmap(fixeddata , linewidth = 0.5 , cmap = 'coolwarm',cbar_kws={'label': 'correlation'},yticklabels=Example,xticklabels=TimeFrame)
plt.title( "2-D Heat Map" )
plt.show()

