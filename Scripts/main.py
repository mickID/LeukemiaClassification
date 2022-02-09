import pandas
from sklearn.preprocessing import quantile_transform
from matplotlib import pyplot as plt

data = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl')
data_columns = data.columns.values
data_qn = quantile_transform(data.to_numpy(), axis=1)
data = pandas.DataFrame(data_qn, columns=data_columns)
data.to_pickle('Datasets/QN_S_Dataset_Leukemia.pkl')


