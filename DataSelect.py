import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Plot:
  def __init__(self, Year, View):
    self.Year = Year
    self.View = View
    self.df = pd.read_csv("LMR_VOCdata_97-19_DOW.csv", na_values='No data',
                        index_col="Datetime",
                       parse_dates=['Datetime'], infer_datetime_format=1)
    self.df = self.df[self.Year]

  def HourDayWeek(self):
    if self.View == "Hourly":
         self.Hourly()
    elif self.View == "Daily":
         self.Daily()
    elif self.View == "Weekly":
         self.Weekly()
    elif self.View == "Monthly":
         self.Monthly()

  def Hourly(self):
      Data = self.df["benzene"]
      print(Data.head())
      Benzene = self.df["benzene"]
      Date = self.df.index

      # Data.dropna(
      #     axis=0,
      #     inplace=True
      #   )

      plt.scatter(Date, Benzene, s=1)
      plt.title("Hourly Benzene level for" +' '+ self.Year)
      plt.xlabel("Hourly")
      plt.ylabel("Benzene emmision")
      plt.show()

  def Daily(self):
      Data = self.df["benzene"]
      Date = self.df.index
      Benzene = self.df["benzene"]
      # Data.dropna(
      #     axis=0,
      #     inplace=True
      # )
      print(Data.head(50))
      Data = Data.groupby(Data.index.to_frame().Datetime.dt.to_period('D')).agg('sum')
      print(Data.head(50))
      Data.plot()
      plt.title("Daily Benzene level for" +' '+ self.Year)
      plt.xlabel("Daily")
      plt.ylabel("Benzene emmision")
      plt.show()

  def Weekly(self):
      Data = self.df["benzene"]
      Date = self.df.index
      Benzene = self.df["benzene"]
      # Data.dropna(
      #     axis=0,
      #     inplace=True
      # )
      print(Data.head(50))
      Data = Data.groupby(Data.index.to_frame().Datetime.dt.to_period('W')).agg('sum')
      Data.plot()
      plt.title("Weekly Benzene level for" +' '+ self.Year)
      plt.xlabel("Weekly")
      plt.ylabel("Benzene emmision")
      plt.show()

  def Monthly(self):
      Data = self.df["benzene"]
      Date = self.df.index
      Benzene = self.df["benzene"]
      # Data.dropna(
      #     axis=0,
      #     inplace=True
      # )
      print(Data.head(50))
      Data = Data.groupby(Data.index.to_frame().Datetime.dt.to_period('M')).agg('sum')
      Data.plot()
      plt.title("Monthly Benzene level for" +' '+ self.Year)
      plt.xlabel("Month")
      plt.ylabel("Benzene emmision")
      plt.show()

class vocData():
    def __init__(self, csvdata="LMR_VOCdata_97-19_DOW.csv"):
        """
        Time series data object with extended data filtering capabilities.

        Parameters
        ----------
        csvdata : .csv, optional
            VOC/time series data used. Must have 'Datetime' column with
            correctly formatted dates.
            The default is "LMR_VOCdata_97-19_DOW.csv".

        Returns
        -------
        vocData object.

        """
        # TODO: BETTER TO INHERET from pandas dataframe class first
        self.data = pd.read_csv(
            csvdata, na_values='No data', index_col="Datetime",
            parse_dates=['Datetime'], infer_datetime_format=1)

        self.units = ['year', 'month', 'dayofweek', 'hour']

    def get_unit(self, unit, data=None):
        if no(data):
            data = self.data
        unitindexes = getattr(self.data.index, unit)

        if isinstance(unitindexes, type(self.get_unit)):
            return unitindexes()
        else:
            return unitindexes


    def select(self, timerange=None, mon_filter=None, dow_filter=None,
               hr_filter=None):
        """
        Selects

        Parameters
        ----------
        timerange : TYPE, optional
            DESCRIPTION. The default is None.
        mon_filter : TYPE, optional
            DESCRIPTION. The default is None.
        dow_filter : TYPE, optional
            DESCRIPTION. The default is None.
        hr_filter : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dataslice : TYPE
            DESCRIPTION.

        """
        if likelist(timerange):
            if len(timerange)==2:
                dataslice = self.data.loc[timerange[0]:timerange[1]]
            elif len(timerange)==1:
                dataslice = self.data.loc[timerange[0]]
            else:
                print("Range > 2 elements, haven't included functionality yet.")
        elif likestr(timerange):
            dataslice = self.data.loc[timerange]
        else:
            dataslice = self.data

        for i, f in enumerate((mon_filter, dow_filter, hr_filter)):
            if notNon(f):
                dataslice = self._filtbyunit(dataslice, f, self.units[i+1])

        return dataslice

    def group_by(self, unit=None, data=None, aggmethod='mean'):
        """


        Parameters
        ----------
        data : TYPE, optional
            DESCRIPTION. The default is None.
        unit : {'Y', 'M', 'W', 'D'}, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if no(data):
            data = self.data

        groups = data.index.to_series().dt.to_period(unit)
        return data.groupby(groups).agg(aggmethod)

    def _filtbyunit(self, data, unitfilter, unit):
        unitindexes = getattr(data.index, unit)
        if likelist(unitfilter):
            locs = [unitindexes==u for u in unitfilter]
            loc = np.any(locs, 0)
        elif unitfilter is None:
            loc = slice(None)
        else:
            loc = unitindexes==unitfilter

        return data[loc]


def likelist(item):
    return type(item)==tuple or type(item)==list

def likestr(item):
    return type(item)==str

def notNon(item):
    return item is not None

def no(item):
    return item is None


#Enter selected year and time window
# p1 = Plot("2015", "Daily")
# p1.HourDayWeek()
