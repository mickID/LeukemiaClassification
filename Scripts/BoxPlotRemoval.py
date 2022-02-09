import pandas
import seaborn as sns
from matplotlib import pyplot as plt

box_df = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl')
data = box_df.transpose()
print(data.shape)

print('----- Preparing plotting for row 2 -----')
fig, ax = plt.subplots(nrows=1, ncols=2)
x = box_df.iloc[2]
sns.boxplot(x=x, ax=ax[0])

print('----- Calculating filtered data with IQR -----')
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1
print(Q1-1.5*IQR)
data = data[~((data < (Q1-1.5*IQR)) | (data > (Q3+1.5*IQR))).any(axis=1)]
box_df = data.transpose()
print('----- New Dataset Shape -----')
print(box_df.shape)
# data.to_pickle('Datasets/IQRremoved.pkl')

print('----- Preparing plotting for row 2 after removing with IQR -----')
x = box_df.iloc[2]
sns.boxplot(x=x, ax=ax[1])

plt.show()
