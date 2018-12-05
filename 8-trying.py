import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

l1 = [0, 1, 2]
def rename(S):
	l2 = []
	for i in S:
		if i not in l2:
			l2.append(i)
	for i in S:
		pos = l2.index(i)
		i = l1[pos]
	return S

iris = load_iris()
print('Data', iris.data)
print('Target names:', iris.target_names)
print('Target:', iris.target)

x = pd.DataFrame(iris.data)
x.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters = 3)
model.fit(x)

plt.figure(figsize = (14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(x.PetalLength, x.PetalWidth, c = colormap[y.Targets], s = 40)
plt.title('Real Classification')
plt.subplot(1, 2, 2)
plt.scatter(x.PetalLength, x.PetalWidth, c = colormap[model.labels_], s = 40)
plt.title('KMeans Classification')
plt.show()

km = rename(model.labels_)
print('What KMeans thought:', km)
print('Accuracy score of KMeans:', accuracy_score(y, km))
print('Confusion matris of KMeans:', confusion_matrix(y, km))

scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns = x.columns)
print('\n', xs.sample(5))

gmm = GaussianMixture(n_components = 3)
gmm.fit(xs)

y_cluster_gmm = gmm.predict(xs)

plt.subplot(1, 2, 1)
plt.scatter(x.PetalLength, x.PetalWidth, c = colormap[y_cluster_gmm], s = 40)
plt.title('GMM Classification')
plt.show()

em = rename(y_cluster_gmm)
print('What EM thought:', km)
print('Accuracy score of EM:', accuracy_score(y, em))
print('Confusion matris of EM:', confusion_matrix(y, em))
