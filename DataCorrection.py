#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 00:14:36 2022

@author: felixdubicki-piper
"""

import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# import all spreadsheets separate data frames
data = pd.read_excel("../LMR_VOCdata_97-19_DOW.xlsx", None,
                     na_values='No data', parse_dates=False)
# concatenate dataframes into one
data = [data[k] for k in list(data.keys())]
data = pd.concat(data)
# export to csv
data.to_csv("../LMR_VOCdata_97-19_DOW_TEMP.csv", index=False)
# read from csv
data = pd.read_csv("../LMR_VOCdata_97-19_DOW_TEMP.csv")

# the midnight times are set incorrectly at the following points indexes
midnightErrLoc = (data.Time == "1900-01-01 00:00:00") | (data.Time == "1900-01-02 00:00:00")
# set inccorect times to midnight value
data.Time[midnightErrLoc] = "00:00:00"
# create datetime string column
data["Datetime"] = data.Date + " " + data.Time
# convert datetime strings to datetime format
data.Datetime = pd.to_datetime(data.Datetime)
# add 1 day to each of the inccorect dates (as originally formatted 1 day off)
data.Datetime[midnightErrLoc] = data.Datetime[midnightErrLoc] + pd.Timedelta(days=1)
# remove unneeded columns
data = data.drop(["Date", "Time", "DOW", "Unnamed: 0"], 1)
# rearrange columns
cols = ["Datetime"] + data.columns.to_list()[:-1]
data = data[cols]
# set datetime index
data = data.set_index(['Datetime'])

if 1:
    # strange anomlies occuring at values below 0.05, so remove them
    data[data<0.05] = np.nan

    # export to csv
    data.to_csv("LMR_VOCdata_97-19_DOW_W.csv", na_rep="NaN")
else:
    # export to csv, WITH ANOMALIES
    data.to_csv("LMR_VOCdata_97-19_DOW_W_ANOMALY.csv", na_rep="NaN")
