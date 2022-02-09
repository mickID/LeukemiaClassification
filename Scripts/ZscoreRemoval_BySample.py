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

print('----- Reducing features with zscores by sample -----')
scaler = StandardScaler()
scaler.fit(data)
normalized = scaler.transform(data)
zscore_matrix = abs(normalized)
filtered = []
for list in zscore_matrix:
    filtered.append(np.all(list < 3))
removed = data[filtered].transpose()
print('----- New Dataset Shape: -----')
print(removed.shape)
# removed.to_pickle('Datasets/ZscoreRemoved_BySample.pkl')
print('----- Preparing second subplot -----')
x = removed.iloc[5]
sns.boxplot(x=x, ax=ax[1])
plt.show()