import pandas
from numpy import random

print('----- Importing Dataset -----')
data = pandas.read_table("Datasets/Dataset_Leukemia_17996.csv", sep=';', decimal='.')
X = data.to_numpy()
data_columns = data.transpose().columns.values
print('----- Importing Data Labels -----')
data_labels = pandas.read_table("Datasets/label_data.csv", sep=';')
y = data_labels.to_numpy().reshape(556, )

print('----- Replacing , with . -----')
for single_list in X:
    for i, item in enumerate(single_list):
        item = str(item)
        single_list[i] = float(item.replace(',', '.'))

print('----- Shuffling Samples -----')
X = X.transpose()
p = random.permutation(len(y))
X = X[p, :]
y = y[p]
print(y)

data = pandas.DataFrame(X, columns=data_columns)
print(data)
data_labels = pandas.DataFrame(y)
data_labels.to_pickle('Datasets/Data_labels.pkl')
data.to_pickle('Datasets/Dataset_Leukemia.pkl')
