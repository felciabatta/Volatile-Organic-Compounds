# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:13:10 2023

@author: Kasper
"""

# importing pandas

  
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


#%%
def combineNew(filenames):
    
    li = []
    
    for filename in fileNames:
        
        data = pd.read_csv(filename, None,
                         na_values='No data', parse_dates=False)
        li.append(data)
    
    data = pd.concat(li,axis=0, ignore_index=True)
    
    data['time'] = data['time'].replace('24:00:00','00:00')
    
    # create datetime string column
    data["Datetime"] = data.Date + " " + data.time
    # convert datetime strings to datetime format
    data.Datetime = pd.to_datetime(data.Datetime)
    # remove unneeded columns
    data = data.drop(["Date", "time", "cis-2-pentene (VOC-AIR only)",
                     "3-methylpentane (VOC-AIR only)","Wind Direction"], 1)
    # rearrange columns
    cols = ["Datetime"] + data.columns.to_list()[:-1]
    data = data[cols]
    # set datetime index
    data.drop(index=data.index[0], axis=0, inplace=True)
    test = data.iloc[:,0]
    print(test)
    data = data.set_index(['Datetime'])
    data.to_csv("LMR_VOCdata_20-23_DOW_TEMP.csv", index=True, na_rep="NaN")
    
    return data

fileNames = ["MY1_2020.csv","MY1_2021.csv","MY1_2022.csv","MY1_2023.csv"]
#data = combineNew(fileNames)



oldData = pd.read_csv("LMR_VOCdata_97-19_DOW_W_ANOMALY.csv")
newData = pd.read_csv("LMR_VOCdata_20-23_DOW_TEMP.csv")



newKeyList = newData.keys()
oldKeyList = oldData.keys()
intersection = list(oldKeyList.intersection(newKeyList))
dataSelect = newData[intersection]

df = pd.concat([oldData, dataSelect], ignore_index=True,)
df.to_csv("LMR_VOCdata_97-23_DOW.csv", index=False, na_rep="NaN")
#%%
import DataSelect


# data = DataSelect.vocData()
# data.scatter(data.data['toluene'])
# data.scatter(data.data['benzene'])
# data.scatter(data.data['Nitrogen dioxide'])
# data.scatter(data.data['Nitric oxide'])
# data.ratio('benzene','toulene')