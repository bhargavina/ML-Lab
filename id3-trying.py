import math
from collections import Counter
from pprint import pprint
from pandas import DataFrame

tennis = DataFrame.from_csv('id3-training-examples.csv')
print('PlayTennis dataset:', tennis)

def entropyOfList(aList):
	cnt = Counter(aList)
	probs = [x / len(aList) for x in cnt.values()]
	entropy = sum([-prob * math.log(prob, 2) for prob in probs])
	return entropy

print('\nEntropy of PlayTennis dataset: %.4f' % entropyOfList(tennis['PlayTennis']))

def informationGain(tennis, splitAttributeName, targetAttributeName):
	split = tennis.groupby(splitAttributeName)
	agg_ent = split.agg({targetAttributeName: [entropyOfList, lambda x: len(x) / len(tennis)]})
	agg_ent.columns = ['Entropy', 'PropObservations']
	newEntropy = sum(agg_ent['Entropy'] * agg_ent['PropObservations'])
	oldEntropy = entropyOfList(tennis[targetAttributeName])
	return oldEntropy - newEntropy

print('\nInformation gain for Outlook: %.4f' % informationGain(tennis, 'Outlook', 'PlayTennis'))
print('Information gain for Temperature: %.4f' % informationGain(tennis, 'Temperature', 'PlayTennis'))
print('Information gain for Humidity: %.4f' % informationGain(tennis, 'Humidity', 'PlayTennis'))
print('Information gain for Wind: %.4f' % informationGain(tennis, 'Wind', 'PlayTennis'))

def id3(tennis, targetAttributeName, attributeNames, defaultClass = None):
	cnt = Counter(tennis[targetAttributeName])
	if len(cnt) == 1:
		return next(iter(cnt))
	elif tennis.empty or (not attributeNames):
		return defaultClass
	else:
		defaultClass = max(cnt.keys())
		gains = [informationGain(tennis, attr, targetAttributeName) for attr in attributeNames]
		best = attributeNames[gains.index(max(gains))]
		tree = {best: {}}
		remainingAttributeNames = [i for i in attributeNames if i != best]
		for attr, subset in tennis.groupby(best):
			tree[best][attr] = id3(subset, targetAttributeName, remainingAttributeNames, defaultClass)
		return tree

attributeNames = list(tennis.columns)
print('\nList of attributes:', attributeNames)
attributeNames.remove('PlayTennis')
print('Predicting attributes:', attributeNames)

tree = id3(tennis, 'PlayTennis', attributeNames)
print('\nThe resultant decision tree is:')
pprint(tree)

attribute = next(iter(tree))
def classify(instance, tree, default = None):
	attribute = next(iter(tree))
	if instance[attribute] in tree[attribute].keys():
		result = tree[attribute][instance[attribute]]
		if isinstance(result, dict):
			return classify(instance, result)
		else:
			return result
	else:
		return default

trainData = tennis.iloc[1: -4]
testData = tennis.iloc[-4: ]
trainTree = id3(trainData, 'PlayTennis', attributeNames)

testData['predicted2'] = testData.apply(classify, axis = 1, args = (trainTree, 'Yes'))
print('\nPredicted values for sample data:\n', testData['predicted2'])
print('Accuracy: ', sum(testData['predicted2'] == testData['PlayTennis']) / len(testData.index))
