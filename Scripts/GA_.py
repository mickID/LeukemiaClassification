from genetic_selection import GeneticSelectionCV
import numpy as np
import pandas
from sklearn import svm

data = pandas.read_pickle('Datasets/N_GA_ZS_5769_Normalized_By_Gene_4.pkl')
print(data)
data_columns = data.columns.values
X = data.to_numpy()
data_labels = pandas.read_pickle('Datasets/Data_labels.pkl')
y = data_labels.to_numpy()
y = np.ravel(y)

print('----- Dataset -----')
print(X.shape)
print('----- Labels -----')
print(y)
print('##### Running GA #####')
classifier = svm.SVC()
selector = GeneticSelectionCV(estimator=classifier,
                              cv=5,
                              n_population=100,
                              n_generations=100,
                              crossover_proba=0.8,
                              mutation_proba=0.1,
                              crossover_independent_proba=0.8,
                              mutation_independent_proba=0.08,
                              tournament_size=3,
                              caching=True,
                              n_jobs=1,
                              verbose=1,
                              scoring='accuracy')
selector.fit(data, y)
print('Number of selected features: ' + str(selector.n_features_))
print('Generations Scores')
print(selector.generation_scores_)
score = selector.score(data, y)
print(score)
indices = selector.get_support(True)
data_columns = data_columns[indices]
print(data_columns)
data = data[data_columns]
print(data.shape)
print(data_columns.shape)
data.to_pickle('Datasets/N_GA_ZS_5769_Normalized_By_Gene_5.pkl')

