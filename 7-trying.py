import pandas as pd
import numpy as np
import pgmpy
from urllib.request import urlopen
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
np.set_printoptions(threshold = np.nan)
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']

heart = pd.read_csv(urlopen(url), names = names)
print(heart.head())

del heart['oldpeak']
del heart['slope']
del heart['ca']
del heart['thal']

heart = heart.replace('?', np.nan)
print(heart.dtypes)

model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), ('exang', 'trestbps'), ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'), ('heartdisease', 'thalach'), ('heartdisease', 'chol')])
model.fit(heart, estimator = MaximumLikelihoodEstimator)

print(model.get_cpds('age'))
print(model.get_cpds('sex'))
print(model.get_cpds('chol'))

model.get_independencies()
inference = VariableElimination(model)

q = inference.query(variables = ['heartdisease'], evidence = {'age': 28})
print(q['heartdisease'])

q = inference.query(variables = ['heartdisease'], evidence = {'chol': 100})
print(q['heartdisease'])
