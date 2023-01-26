from DataSelect import vocData
import pandas as pd
import matplotlib.pyplot as plt


data = vocData()
#data.scatter(data.data['benzene'])
Selection = data.select(["2000-02 13", "2020-01"], groupby='M')
benzene = Selection['benzene']
toluene = Selection['toluene']
frames = [benzene, toluene]
result = pd.concat(frames, axis=1)
print(result)
result = result.dropna()
result['benzene']=result['benzene'].pct_change()
result['toluene']=result['toluene'].pct_change()
print(result['benzene'])

plt.scatter(result['benzene'], result['toluene'])
correlation = result['toluene'].corr(result['benzene'])
print('correlation is:',correlation)


class trendcomparison():
    def __init__(self, emission1,emmision2):
        # TODO: BETTER TO INHERET from pandas dataframe class first
        self.emission1 = emission1
        self.emmision2 = emmision2

    
    def correlation(emmision1,emmision2):
        data = vocData()
        
        #data.scatter(data.data['benzene'])
        Selection = data.select(["2000-02 13", "2020-01"], groupby='M')
        emmision1 = Selection[emmision1]
        emmision2 = Selection[emmision2]
        frames = [emmision1, emmision2]
        result = pd.concat(frames, axis=1)
        print(result)
        result = result.dropna()
        result[emmision1]=result[emmision1].pct_change()
        result[emmision2]=result[emmision2].pct_change()
        plt.scatter(result[emmision1], result[emmision2])
        correlation = result[emmision1].corr(result[emmision2])
        print('correlation is:',correlation)