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

data = pd.read_excel("../LMR_VOCdata_97-19_DOW.xlsx", None,
                     na_values='No data', parse_dates=False)

data = [data[k] for k in list(data.keys())]

data = pd.concat(data)

data.to_csv("../LMR_VOCdata_97-19_DOW_TEMP.csv")

data = pd.read_csv("../LMR_VOCdata_97-19_DOW_TEMP.csv")

# the midnight times are set incorrectly at the following points indexes
midnightErrLoc = (data.Time == "1900-01-01 00:00:00") | (data.Time == "1900-01-02 00:00:00")

data.Time[midnightErrLoc] = "00:00:00"

data.Datetime = data.Date + " " + data.Time

data.Datetime = pd.to_datetime(data.Datetime)

data.Datetime[midnightErrLoc] = data.Datetime[midnightErrLoc] + pd.Timedelta(days=1)

data = data.drop(["Date", "Time", "DOW", "Unnamed: 0"], 1)

cols = ["Datetime"] + data.columns.to_list()[:-1]

data = data[cols]

data = data.set_index(['Datetime'])

data.to_csv("LMR_VOCdata_97-19_DOW.csv", na_rep="NaN")
