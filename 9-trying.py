from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()
print('IRIS FEATURES | TARGET NAMES:', dataset.target_names)
print('\nData:', dataset["data"])
print('\nTarget', dataset["target"])

xtrain, xtest, ytrain, ytest = train_test_split(dataset["data"], dataset["target"], random_state = 0)

print("\nX TRAIN \n", xtrain)
print("\nX TEST \n", xtest)
print("\nY TRAIN \n", ytrain)
print("\nY TEST \n", ytest)

kn = KNeighborsClassifier(n_neighbors = 1)
kn.fit(xtrain, ytrain)

predictions = kn.predict(xtest)
for i in range(len(xtest)):
    print("\nActual: {0} {1} \nPredicted: {2} {3}".format(ytest[i], dataset["target_names"][ytest[i]], predictions, dataset["target_names"][predictions]))
print("\nTEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(xtest, ytest)))
