# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:28:24 2023

@author: vr198
"""

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import JRPComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.settings import JointSettings
from pyrqa.neighbourhood import Unthresholded
from DataSelect import vocData
import pandas as pd
import matplotlib.pyplot as plt

data = vocData()
Selection = data.select(["2000-02 13", "2020-01"], groupby='M')
benzene = Selection['benzene'].dropna()
benzene = list(benzene)
print(benzene)

time_series = TimeSeries(benzene,
                         embedding_dimension=2,
                         time_delay=2)
settings = Settings(time_series,
                    analysis_type=Classic,
                    neighbourhood=FixedRadius(0.65),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)
computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2
print(result)

computation = RPComputation.create(settings)
result = computation.run()
ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    'recurrence_plot.png')
filename = 'recurrence_plot.png'

image = plt.imread(filename)
plt.imshow(filename)

