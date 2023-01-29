import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from DataSelect import vocData
from prophet.plot import plot_yearly, plot_weekly, plot_seasonality
import numpy as np

data = vocData()

class FbProphetCorrelation():
    def __init__(self, VOC1, VOC2, data=data):
        self.VOC1 = VOC1
        self.VOC2 = VOC2
        self.data = data
    
    def fit(self, *dataselect_args, log=False, **dataselect_kwargs):
        VOC = [self.VOC1, self.VOC2]
        frames = np.empty(2, pd.DataFrame) # output data
        m = np.empty(2, Prophet) # model
        components = np.empty(2, pd.DataFrame)
        for i, voc in enumerate(VOC):
            # prepare dataframe
            datasubset = self.data.select(*dataselect_args, **dataselect_kwargs)[voc].to_frame()
            datasubset['ds'] = datasubset.index.to_series()
            datasubset = datasubset.rename({voc: 'y'}, axis='columns')
            datasubset = datasubset[['ds','y']]
            if log:
                datasubset['y'] = np.log(datasubset['y'])
            # fit model in prophet
            m[i] = Prophet(yearly_seasonality=6, weekly_seasonality=3, daily_seasonality=False)
            # doesn't make sense to include monthly
            # m[i].add_seasonality(name='monthly', period=30.5, fourier_order=5)
            if 'groupby' in dataselect_kwargs.keys():
                if dataselect_kwargs['groupby'] == 'H':
                    m[i].add_seasonality(name='daily', period=1, fourier_order=5)
            else:
                m[i].add_seasonality(name='daily', period=1, fourier_order=5)
            m[i].fit(datasubset)
            # extract components
            df = m[i].history
            components[i] = m[i].predict(df)
            m[i].plot(components[i])
            plt.title(voc)
            m[i].plot_components(components[i])
            # plt.title(voc)
            # save df for additional use
            frames[i] = df['y']
        
        frames = pd.concat(frames, axis=1)
        frames.columns = ['col1', 'col2']
        return frames, m, components
            
    def Correlation(self, fitted_data):
        fitted_data = fitted_data.dropna()
        # find percent change
        fitted_data['col1']=fitted_data['col1'].pct_change()
        fitted_data['col2']=fitted_data['col2'].pct_change()
        # plot
        plt.figure()
        plt.scatter(fitted_data['col1'], fitted_data['col2'])
        correlation = fitted_data['col1'].corr(fitted_data['col2'])
        print('correlation is:',correlation)

CombinationOne = FbProphetCorrelation('benzene','toluene')
fitted, m, components = CombinationOne.fit(log=True, groupby='H')
# BenTolCorr = CombinationOne.Correlation(fitted_data)


# print(CombinationOne.VOC1)
# print(CombinationOne.VOC2)
# print(Datasets)
# print(BenTolCorr)



#m = Prophet(weekly_seasonality=False) 
#m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
#future = m.make_future_dataframe(periods=30, freq='M')
#forecast = m.predict(future)
#dataset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#m.plot(dataset)
#m.plot_components(dataset)
#plot_components_plotly(m, dataset)   
#plot_plotly(m, dataset)
#print(datasubset.head(50))
#print(List)



#"--ADDING CHANGEPOINT PARAMETER--"
#"By default, this parameter is set to 0.05"
#"Decreasing parameter will make the trend less flexible, undefit"

#m = Prophet(changepoint_prior_scale=0.5)
#m.fit(datasubset)
#future = m.make_future_dataframe(periods=0, freq='D')
#forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#m.plot(forecast)
#m.plot_components(forecast)
#plot_components_plotly(m, forecast)   
#plot_plotly(m, forecast)



#--ADDING SINGLE SEASONALITY PARAMETER--
#Decreasing parameter will make the trend less flexible, undefit

#Seasonalities are estimated using a partial Fourier sum.
# m = Prophet(yearly_seasonality=0.5) 
# m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# m.fit(df)
# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)
# plot_components_plotly(m, forecast)

# plot_plotly(m, forecast)

# #"--ADDING MULTIPLE SEASONALITY PARAMETER--"
# def season(ds):
#     date = pd.to_datetime(ds)
#     return (date.month > 8 or date.month < 2)

# #"P = 365.25 for yearly data or P = 7 for weekly data, when we scale our time variable in days"
# m = Prophet(weekly_seasonality=False)
# m.add_seasonality(name='weekly_Summer', period=7, fourier_order=3, condition_name='Summer')
# m.add_seasonality(name='weekly_Winter', period=7, fourier_order=3, condition_name='Winter')

# future['Summer'] = future['ds'].apply(season)
# future['Winter'] = ~future['ds'].apply(season)
# m.fit(df)
# forecast = m.predict(future)
# fig = m.plot_components(forecast)
