import numpy as np
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl').transpose()
print(data.shape)
box_df = data.transpose()

print('----- Preparing first subplot -----')
fig, ax = plt.subplots(nrows=1, ncols=2)
x = box_df.iloc[5]
sns.boxplot(x=x, ax=ax[0])

print('----- Reducing features with zscores by gene -----')
scaler = StandardScaler()
scaler.fit(data.transpose())
normalized = scaler.transform(data.transpose())
print(normalized.mean(axis=0))
zscore_matrix = abs(normalized.transpose())
filtered = []
for list in zscore_matrix:
    filtered.append(np.all(list < 3+np.mean(list)))
removed = data[filtered].transpose()

print('----- New Dataset Shape: -----')
print(removed)
# removed.to_pickle('Datasets/ZscoreRemoved_ByGene.pkl')
print('----- Preparing second subplot -----')
x = removed.iloc[5]
sns.boxplot(x=x, ax=ax[1])
plt.show()
