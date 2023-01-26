import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import seaborn as sns
from scipy import stats as st

from typing import List, Union, Literal


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
        plt.title("Hourly Benzene level for" + ' ' + self.Year)
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
        Data = Data.groupby(Data.index.to_frame(
        ).Datetime.dt.to_period('D')).agg('sum')
        print(Data.head(50))
        Data.plot()
        plt.title("Daily Benzene level for" + ' ' + self.Year)
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
        Data = Data.groupby(Data.index.to_frame(
        ).Datetime.dt.to_period('W')).agg('sum')
        Data.plot()
        plt.title("Weekly Benzene level for" + ' ' + self.Year)
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
        Data = Data.groupby(Data.index.to_frame(
        ).Datetime.dt.to_period('M')).agg('sum')
        Data.plot()
        plt.title("Monthly Benzene level for" + ' ' + self.Year)
        plt.xlabel("Month")
        plt.ylabel("Benzene emmision")
        plt.show()


class vocData():
    """
    Time series dataframe object with extended data filtering capabilities.

    Parameters
    ----------
    csvdata : .csv, optional
        VOC/time series data used. Must have 'Datetime' column with
        correctly formatted dates.
        The default is "LMR_VOCdata_97-19_DOW.csv".

    Returns
    -------
    vocData object.

    Examples
    --------

    Initialise using

    >>> data = vocData()

    Select, filter and group using

    >>> data.select(timerange, mon_filter, dow_filter, hr_filter, groupby, aggmethod)

    """

    def __init__(self, csvdata="LMR_VOCdata_97-19_DOW.csv"):
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

    def select(self, timerange: Union[str, List[str]] = None,
               mon_filter: Union[int, List[int]] = None,
               dow_filter: Union[int, List[int]] = None,
               hr_filter: Union[int, List[int]] = None,
               groupby: Literal['Y', 'M', 'D'] = None,
               aggmethod: Literal['mean', 'sum'] = 'mean') -> pd.DataFrame:
        """Select, filter and group data.

        Select data within timerange, subject to filter by month, day of week or hour.
        Can also aggregate data by year, month or day.

        Parameters
        ----------
        timerange : ("YYYY-MM-DD hh:mm:ss", list), optional
            YYYY to select whole year, YYYY-MM to select whole month, etc.
            Use list of two dates to select a range.
            The default is None, which selects entire dataset.
        mon_filter : (int, list), optional
            In addition to time range selection, filter out only specific month.
            e.g. `1` will extract all Januarys only.
            The default is None, extracting all months.
        dow_filter : (int, list), optional
            In addition to time range selection, filter out only specific day of week.
            e.g. `1` will extract all Mondays only.
            The default is None, extracting all dows.
        hr_filter : (int, list), optional
            In addition to time range selection, filter out only specific hour of day.
            e.g. `13` will extract all 1pms only.
            The default is None, extracting all hours.
        groupby : {'Y', 'M', 'D'}, optional
            Aggregates by grouping of year `'Y'`, month `'M'` or day `'D'`, according to `aggmethod`.
            The default is None.
        aggmethod : {'mean', 'sum'}, optional
            Aggregation  method.
            The default is 'mean'.

        Returns
        -------
        dataslice : pandas.dataframe
            Selected data.

        Examples
        --------
        Select 1pm Feb 2003 to Aug 2018, group/aggregated by day:

        >>> data.select(["2003-02 13", "2018-08"], groupby='D')

        Select all Tuesdays in Feb 2003:

        >>> data.select("2003-02", dow_filter=1)

        Select all Tuesdays and Sundays in both Aug and Nov of 2003:

        >>> data.select("2003", mon_filter=[8, 11], dow_filter=[1, 6])

        Select all 12ams and 12pms, and group by month, aggregated with sum

        >>> data.select(hr_filter=[0 12], groupby='M', aggmethod='sum')

        """
        if likelist(timerange):
            if len(timerange) == 2:
                dataslice = self.data.loc[timerange[0]:timerange[1]]
            elif len(timerange) == 1:
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

        if notNon(groupby):
            dataslice = self._group_by(groupby, dataslice, aggmethod)

        return dataslice

    def _group_by(self, unit=None, data=None, aggmethod='mean'):
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

        grouped_data = data.groupby(groups).agg(aggmethod)
        newidx = grouped_data.index.to_timestamp()
        grouped_data = grouped_data.set_index(newidx)

        return grouped_data

    def _filtbyunit(self, data, unitfilter, unit):
        unitindexes = getattr(data.index, unit)
        if likelist(unitfilter):
            locs = [unitindexes == u for u in unitfilter]
            loc = np.any(locs, 0)
        elif unitfilter is None:
            loc = slice(None)
        else:
            loc = unitindexes == unitfilter

        return data[loc]

    def corrmat(self, data=None, H0=0, tail='up', plot=0, sort_index=None):
        if no(data):
            data = self.data
        # correlation matrix
        corr = data[VOCS].corr(method='spearman')
        if no(sort_index):
            # sort by mean r-value
            sort_index = np.argsort(np.mean(corr)).to_list()
        corr = corr[corr.columns[sort_index]].loc[corr.columns[sort_index]]
        # corr, p = st.spearmanr(data[vocs], nan_policy='omit')

        # no. of data points
        n = len(data)
        # null hypothesis: assumed spearman mean
        mu = H0
        # fisher transformation - use abs as interested prescence correlation
        f = np.arctanh(abs(corr))
        muf = np.arctanh(mu)  # fisher trans. mean
        # standard normal
        z = (f - muf)*np.sqrt(n-3)
        # p-values (cumulative probabilities)
        p = st.norm.cdf(z)

        # critical value
        crit = 0.01
        # no. statistical tests (= no. correlations)
        m = (len(corr)-1)*len(corr)/2
        # bonferroni corrected critical value
        critl = crit/m
        critu = 1-crit

        # determine which are statistically significant
        if tail == 'two':
            significant = (p <= critl/2) & ((1-p) <= critl/2)
        elif tail == 'low':
            significant = (p <= critl)
        elif tail == 'up':
            significant = (1-p <= critl)

        if plot:
            # Sample figsize in inches
            fig, ax = plt.subplots(figsize=(10, 7.5))
            sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1,
                        xticklabels=True, yticklabels=True)
            # Sample figsize in inches
            fig, ax = plt.subplots(figsize=(10, 7.5))
            sns.heatmap(significant,
                        xticklabels=True, yticklabels=True)
        return corr, p, significant, sort_index

    def correlate(self, col1, col2, data=None, plot=False):
        if no(data):
            data = self.data
        data1 = data[col1]
        data2 = data[col2]
        corr = data1.corr(data2)

        if plot:
            plt.figure()
            plt.scatter(data1, data2, s=.1)
            plt.xlabel(col1)
            plt.ylabel(col2)
        return corr

    def ratio(self, col1, col2, data=None, inv=False, plot=False):
        if no(data):
            data = self.data
        if inv:
            data1 = data[col2]
            data2 = data[col1]
        else:
            data1 = data[col1]
            data2 = data[col2]
        q = data1/data2
        logq = np.log()

        if plot:
            self.scatter(logq, title=data1.name+':'+data2.name)
        return logq

    def scatter(self, data=None, title=None):
        if no(data):
            data = self.data
        if no(title):
            title = data.name
        dates = data.index
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dates, data, s=1)
        plt.xlabel('Time')
        plt.title(title)


def likelist(item):
    return type(item) == tuple or type(item) == list


def likestr(item):
    return type(item) == str


def notNon(item):
    return item is not None


def no(item):
    return item is None


VOCS = ['Nitric oxide',
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
        'cis-2-pentene (VOC-AIR only)', # ONLY TO 2005
        '2-methylpentane',
        '3-methylpentane (VOC-AIR only)', # ONLY TO 2005
        'isoprene',
        'n-hexane',
        'n-heptane',
        'benzene',
        'toluene',
        'ethylbenzene',
        'm+p-xylene',
        'o-xylene',
        '1,2,4-trimethylbenzene',
        '1,3,5-trimethylbenzene']
# strange anomaly occurs around 2013-2015 at near 0 value (<0.05)
# for almost all VOCS - only top 5 doesn't appear

# Enter selected year and time window
# p1 = Plot("2015", "Daily")
# p1.HourDayWeek()
