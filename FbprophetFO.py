import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from DataSelect import vocData
from prophet.plot import plot_yearly, plot_weekly, plot_seasonality

class FbProphetCorrelation():
    data = vocData()
    
    def __init__(self, VOC1, VOC2):
        self.VOC1 = VOC1
        self.VOC2 = VOC2
    
    def DataSubset(self):
        VOC = [self.VOC1,self.VOC2]
        frames = [0, 0]
        for i, voc in enumerate(VOC):
            self.data.data # all the data
            datasubset = self.data.select(["2000-02", "2000-04"])[''+voc+''].to_frame() # selects
            datasubset = self.data._group_by('H', datasubset)
            datasubset['ds'] = datasubset.index.to_series()
            datasubset = datasubset.rename({voc: 'y'}, axis='columns')
            datasubset = datasubset[["ds","y"]]
            m = Prophet(yearly_seasonality=10)
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m.add_seasonality(name='weekly', period=7, fourier_order=3)
            m.add_seasonality(name='daily', period=1, fourier_order=5)
            m.fit(datasubset)
            df = m.history
            plot_yearly(m)
            plot_weekly(m)
            plot_seasonality(m, name='monthly')
            plot_seasonality(m, name='daily')
            print(m)
            
            frames[i] = df['y']
            
        #     if i == VOC[0]:
        #         DataSet1 = df['y']
        #     if i == VOC[1]:
        #         DataSet2 = df['y']
        # frames = [DataSet1,DataSet2]
        result = pd.concat(frames, axis=1)
        result.columns = ['col1', 'col2']
        return result
            
    def Correlation(self):
        Datasets = CombinationOne.DataSubset()
        Datasets = Datasets.dropna()
        Datasets['col1']=Datasets['col1'].pct_change()
        Datasets['col2']=Datasets['col2'].pct_change()
        plt.figure()
        plt.scatter(Datasets['col1'], Datasets['col2'])
        correlation = Datasets['col1'].corr(Datasets['col2'])
        print('correlation is:',correlation)
        
        
        
        
        
        
        
CombinationOne = FbProphetCorrelation('benzene','toluene')
Datasets = CombinationOne.DataSubset()
BenTolCorr = CombinationOne.Correlation()


print(CombinationOne.VOC1)
print(CombinationOne.VOC2)
print(Datasets)
print(BenTolCorr)



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
