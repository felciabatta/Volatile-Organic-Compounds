from DataSelect import vocData, VOCS

# initialise vocData
data = vocData()

# %% FIT DATA and EXPORT
for voc in VOCS:
    data.fit(voc=voc, log=True, save=True, groupby='D');


