import pandas as pd
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
      # self.df = pd.read_excel(r"C:\Users\vr198\OneDrive\Desktop\MDM3\Chemistry\LMR_VOCdata_97-19_DOW.xlsx", sheet_name=self.Year)
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
      # self.df = pd.read_excel(r"C:\Users\vr198\OneDrive\Desktop\MDM3\Chemistry\LMR_VOCdata_97-19_DOW.xlsx", sheet_name=self.Year,parse_dates=True)
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
      # self.df = pd.read_excel(r"C:\Users\vr198\OneDrive\Desktop\MDM3\Chemistry\LMR_VOCdata_97-19_DOW.xlsx", sheet_name=self.Year,parse_dates=['Datetime'])
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
      # self.df = pd.read_excel(r"C:\Users\vr198\OneDrive\Desktop\MDM3\Chemistry\LMR_VOCdata_97-19_DOW.xlsx", sheet_name=self.Year,parse_dates=['Datetime'])
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

#Enter selected year and time window
p1 = Plot("2015", "Daily")
p1.HourDayWeek()
