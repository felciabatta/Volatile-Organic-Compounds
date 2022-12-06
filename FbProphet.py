"Using fbProphet on hourly Benzene data"

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from DataSelect import vocData


# df = pd.read_excel(r"C:\Users\vr198\OneDrive\Desktop\MDM3\Chemistry\Volatile-Organic-Compounds\LMR_VOCdata.xlsx", sheet_name="2015",parse_dates=['DateTime'])
# Data = df[["DateTime","benzene"]]
# Data = Data.rename({'DateTime': 'ds', 'benzene': 'y'}, axis='columns')
# Data = pd.DataFrame(Data)
# Data.dropna(
#     axis=0,
#     inplace=True
# )

data = vocData() #initialise data class
data.data # all the data
datasubset = data.select(timerange=None, dow_filter=(4))['benzene'].to_frame() # selects
datasubset = data.group_by('D', datasubset)
datasubset['ds'] = datasubset.index.to_series()
datasubset = datasubset.rename({'benzene': 'y'}, axis='columns')
datasubset = datasubset[["ds","y"]]

print(datasubset.head(50))


#"--ADDING CHANGEPOINT PARAMETER--"
#"By default, this parameter is set to 0.05"
#"Decreasing parameter will make the trend less flexible, undefit"

m = Prophet(changepoint_prior_scale=0.5)
m.fit(datasubset)
future = m.make_future_dataframe(periods=10, freq='D')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)
m.plot_components(forecast)
plot_components_plotly(m, forecast)   
plot_plotly(m, forecast)



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
