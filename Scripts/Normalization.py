import pandas
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

""" Normalizing Datasets reduced with Zscore matrix to 5769 genes """
print('##### Importing ZscoreRemoved_ByGene Dataset #####')
data = pandas.read_pickle('Datasets/ZscoreRemoved_ByGene.pkl')
X = data.to_numpy()
print(data)
data_columns = data.columns.values
print(data_columns)

# Grafico prima della normalizzazione
plt.hist(X[:, 1], bins=500, rwidth=0.8, density=True)
plt.xlabel('Gene expression')
plt.ylabel('Count')
plt.show()

print('----- Gene Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.transform(X)
normalized_df = pandas.DataFrame(normalized, columns=data_columns)
print(normalized_df)
# normalized_df.to_pickle('Datasets/ZS_5769_Normalized_By_Gene.pkl')

# Grafico dopo la normalizzazione
plt.hist(normalized[:, 1], bins=500, rwidth=0.8, density=True)
plt.xlabel('Gene expression')
plt.ylabel('Count')
plt.show()

print('##### Importing ZscoreRemoved_ByGene Dataset #####')
data = pandas.read_pickle('Datasets/ZscoreRemoved_ByGene.pkl')
X = data.to_numpy().transpose()
print(data)
data_columns = data.columns.values
print(data_columns)

print('----- Sample Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.transform(X)
normalized_df = pandas.DataFrame(normalized.transpose(), columns=data_columns)
print(normalized_df)
normalized_df.to_pickle('Datasets/ZS_5769_Normalized_By_Sample.pkl')


""" Normalizing Datasets reduced with Zscore matrix to 16408 genes """
print('##### Importing ZscoreRemoved_BySample Dataset #####')
data = pandas.read_pickle('Datasets/ZscoreRemoved_BySample.pkl')
data_columns = data.columns.values
X = data.to_numpy().transpose()

print('----- Sample Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.fit_transform(X)
data = pandas.DataFrame(normalized.transpose(), columns=data_columns)
print(data)
# data.to_pickle('Datasets/ZS_16408_Normalized_By_Sample.pkl')

print('##### Importing ZscoreRemoved_BySample Dataset #####')
data = pandas.read_pickle('Datasets/ZscoreRemoved_BySample.pkl')
data_columns = data.columns.values
X = data.to_numpy()

print('----- Gene Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.fit_transform(X)
data = pandas.DataFrame(normalized, columns=data_columns)
print(data)
data.to_pickle('Datasets/ZS_16408_Normalized_By_Gene.pkl')

""" Normalizing Full Dataset of 17996 features """
print('##### Importing Leukemia Full Dataset #####')
data = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl')
X = data.to_numpy()
print(data)
data_columns = data.columns.values
print(data_columns)

print('----- Gene Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.transform(X)
data = pandas.DataFrame(normalized, columns=data_columns)
print(data)
# data.to_pickle('Datasets/Full_Normalized_By_Gene.pkl')

print('##### Importing Leukemia Full Dataset #####')
data = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl')
X = data.to_numpy().transpose()
print(data)
data_columns = data.columns.values
print(data_columns)

print('----- Sample Normalization with MinMax Scaler -----')
sc = MinMaxScaler()
sc.fit(X)
normalized = sc.fit_transform(X)
data = pandas.DataFrame(normalized.transpose(), columns=data_columns)
print(data)
# data.to_pickle('Datasets/Full_Normalized_By_Sample.pkl')
