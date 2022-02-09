import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from sklearn import svm
from matplotlib import pyplot as plt
from yellowbrick.features import Manifold
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from sklearn.preprocessing import MinMaxScaler

data = pandas.read_pickle('Datasets/GA_Full_Normalized_By_Gene_1.pkl')
selected = data.columns.values
labels = pandas.read_pickle('Datasets/Data_labels.pkl')
y = labels.to_numpy()
y = np.ravel(y)

df = pandas.read_pickle('Datasets/Dataset_Leukemia.pkl')
data = df[selected]
le = LabelEncoder()
le.fit(['AML', 'ALL'])
target = le.transform(y)
viz = Manifold(manifold='tsne', classes=['AML', 'ALL'])
viz.fit_transform(data, target)
viz.show()

sc = MinMaxScaler()
X = sc.fit_transform(data.to_numpy())
#X = X.transpose()

print('----- Using 5-Fold CrossValidation -----')
classifier = svm.SVC()
y_pred = cross_val_predict(classifier, X, y, cv=5)
tot = y.shape
correct_count = np.sum(y_pred == y)
print('----- Accuracy -----')
print(correct_count/tot)
print('----- Plotting Confusion Matrix -----')
mat = confusion_matrix(y, y_pred)
plot_confusion_matrix(y, y_pred)
plt.show()
print('----- Using Yellobrick -----')
classifier = svm.SVC()
cv = StratifiedKFold(n_splits=5)
visualizer = CVScores(model=classifier, cv=cv, scoring='accuracy')
visualizer.fit(X, y)
visualizer.show()
