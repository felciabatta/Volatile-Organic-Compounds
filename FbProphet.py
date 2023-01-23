"Using fbProphet on hourly Benzene data"

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from DataSelect import vocData


VOCs = ['benzene','toluene']
for i in VOCs:
    data = vocData() #initialise data class
    data.data # all the data
    datasubset = data.select(["2000-02 13", "2020-01"])[i].to_frame() # selects
    datasubset = data._group_by('D', datasubset)
    datasubset['ds'] = datasubset.index.to_series()
    datasubset = datasubset.rename({i: 'y'}, axis='columns')
    datasubset = datasubset[["ds","y"]]
    m = Prophet(changepoint_prior_scale=0.5)
    m.fit(datasubset)
    future = m.make_future_dataframe(periods=0, freq='D')
    forecast = m.predict(future)
    dataset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    m.plot(dataset)
    #m.plot_components(dataset)
    #plot_components_plotly(m, dataset)   
    plot_plotly(m, dataset)
    #print(datasubset.head(50))
    #print(List)
    if i == VOCs[0]:
        dataOne = dataset['yhat']
    if i == VOCs[1]:
        dataTwo = dataset['yhat']


frames = [dataOne,dataTwo]
result = pd.concat(frames, axis=1)
result.columns = ['col1', 'col2']
print(result)
result = result.dropna()
result['col1']=result['col1'].pct_change()
result['col2']=result['col2'].pct_change()
plt.figure()
plt.scatter(result['col1'], result['col2'])
correlation = result['col1'].corr(result['col2'])
print('correlation is:',correlation)





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
