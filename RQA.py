from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings,JointSettings
from pyrqa.analysis_type import Classic, Cross
from pyrqa.neighbourhood import FixedRadius,RadiusCorridor
from pyrqa.metric import EuclideanMetric,MaximumMetric, TaxicabMetric
from pyrqa.computation import RQAComputation,RPComputation, JRPComputation, JRQAComputation
from pyrqa.image_generator import ImageGenerator
from DataSelect import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rqa(VOC, agrre, type, required, year=None):
### 
    df = vocData().select(groupby=agrre,timerange=year)
    data_points = df[VOC]
    if type == 'recurrence':
        name = VOC + '_' + agrre + '_' + type
        if year != None:
            name = VOC + '_' + agrre + '_' + year[0]+'-'+year[1] + '_' + type
        time_series = TimeSeries(data_points,
                         embedding_dimension=2,
                         time_delay=2)
        settings = Settings(time_series,
                    analysis_type=Classic,
                    neighbourhood=FixedRadius(0.65),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)
        computation = RQAComputation.create(settings,
                                    verbose=True)
        plot_comp = RPComputation.create(settings)
    elif type == 'joint':
        name = VOC[0] + '_' + VOC[1] + '_' + agrre + '_' + type
        if year != None:
            name = VOC[0] + '_' + VOC[1] + '_' + agrre + '_' + year[0]+'-'+year[1] + '_' + type
        data_points_x = data_points[VOC[0]]
        data_points_y = data_points[VOC[1]]
        time_series_1 = TimeSeries(data_points_x,
                           embedding_dimension=2,
                           time_delay=1)
        settings_1 = Settings(time_series_1,
                      analysis_type=Classic,
                      neighbourhood=RadiusCorridor(inner_radius=0.14,
                                                   outer_radius=0.97),
                      similarity_measure=MaximumMetric,
                      theiler_corrector=1)
        time_series_2 = TimeSeries(data_points_y,
                             embedding_dimension=2,
                             time_delay=1)

        settings_2 = Settings(time_series_2,
                      analysis_type=Classic,
                      neighbourhood=RadiusCorridor(inner_radius=0.14,
                                                   outer_radius=0.97),
                      similarity_measure=MaximumMetric,
                      theiler_corrector=1)
        joint_settings = JointSettings(settings_1,
                               settings_2)
        computation = JRQAComputation.create(joint_settings,
                                verbose=True)
        plot_comp = JRPComputation.create(joint_settings)
    elif type == 'cross':
        name = VOC[0] + '_' + VOC[1] + '_' + agrre + '_' + type
        if year != None:
            name = VOC[0] + '_' + VOC[1] + '_' + agrre + '_' + year[0]+'-'+year[1] + '_' + type
        data_points_x = data_points[VOC[0]]
        data_points_y = data_points[VOC[1]]
        time_series_x = TimeSeries(data_points_x,
                           embedding_dimension=2,
                           time_delay=1)
        time_series_y = TimeSeries(data_points_y,
                           embedding_dimension=2,
                           time_delay=2)
        time_series = (time_series_x,
               time_series_y)
        settings = Settings(time_series,
                    analysis_type=Cross,
                    neighbourhood=FixedRadius(0.73),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=0)
        computation = RQAComputation.create(settings,
                                    verbose=True)
        plot_comp = RPComputation.create(settings)

    result = computation.run()
    plot = plot_comp.run()

    if required == 'data':
        rec_data = pd.DataFrame({
        'Recurrence rate':[result.recurrence_rate],
        'Determinism':[result.determinism],
        'Average diagonal line length (L)': [result.average_diagonal_line],
        'Longest diagonal line length (L_max)': [result.longest_diagonal_line],
        'Divergence (DIV)': [result.divergence],
        'Entropy diagonal lines (L_entr)': [result.entropy_diagonal_lines],
        'Laminarity (LAM)': [result.laminarity],
        'Trapping time (TT)':[result.trapping_time],
        'Longest vertical line length (V_max)': [result.longest_vertical_line],
        'Entropy vertical lines (V_entr)': [result.entropy_vertical_lines],
        'Average white vertical line length (W)': [result.average_white_vertical_line],
        'Longest white vertical line length (W_max)': [result.longest_white_vertical_line],
        'Longest white vertical line length inverse (W_div)': [result.longest_white_vertical_line_inverse],
        'Entropy white vertical lines (W_entr)': [result.entropy_white_vertical_lines],
        'Ratio determinism / recurrence rate (DET/RR)': [result.ratio_determinism_recurrence_rate],
        'Ratio laminarity / determinism (LAM/DET)': [result.ratio_laminarity_determinism]
    })
        return name, rec_data
    elif required == 'plot':
        print(name)
        ImageGenerator.save_recurrence_plot(plot.recurrence_matrix_reverse,
                                    'recurrence_plots/'+name+'.png')
        return

# plots = [['benzene','D','recurrence','plot'],
#         ['benzene','M','recurrence','plot'],
#         ['benzene','Y','recurrence','plot'],
#         ['toluene','D','recurrence','plot'],
#         ['toluene','M','recurrence','plot'],
#         ['toluene','Y','recurrence','plot'],
#         [['benzene','toluene'],'D','joint','plot'],
#         [['benzene','toluene'],'M','joint','plot'],
#         [['benzene','toluene'],'Y','joint','plot'],
#         [['benzene','toluene'],'D','cross','plot'],
#         [['benzene','toluene'],'M','cross','plot'],
#         [['benzene','toluene'],'Y','cross','plot']
# ]
# for p in plots:
#     rqa(p[0],p[1],p[2],p[3])

# recs = [['benzene','D','recurrence','data'],
#         ['benzene','M','recurrence','data'],
#         ['benzene','Y','recurrence','data'],
#         ['toluene','D','recurrence','data'],
#         ['toluene','M','recurrence','data'],
#         ['toluene','Y','recurrence','data'],
#         [['benzene','toluene'],'D','joint','data'],
#         [['benzene','toluene'],'M','joint','data'],
#         [['benzene','toluene'],'Y','joint','data'],
#         [['benzene','toluene'],'D','cross','data'],
#         [['benzene','toluene'],'M','cross','data'],
#         [['benzene','toluene'],'Y','cross','data']
# ]

# data_dict = {}
# for r in recs:
#     name, data = rqa(r[0],r[1],r[2],r[3])
#     data_dict.update({name:data})

# pd.DataFrame.from_dict(data_dict, orient='columns')
# pd.DataFrame.from_dict({(i,j): data_dict[i][j] 
#                            for i in data_dict.keys() 
#                            for j in data_dict[i].keys()},
#                        orient='index').T
