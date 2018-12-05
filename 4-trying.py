import math
import random

def initializeNetwork(nInputs, nHidden, nOutputs):
	network = []
	hiddenLayer = [{'weights': [random.uniform(-.5, .5) for i in range(nInputs + 1)]} for i in  range(nHidden)]
	network.append(hiddenLayer)
	outputLayer = [{'weights': [random.uniform(-.5, .5) for i in range(nInputs + 1)]} for i in  range(nOutputs)]
	network.append(outputLayer)
	print('The initial neural network is')
	for i, layer in zip(range(1, len(network) + 1), network):
		for j, neuron in zip(range(1, len(layer) + 1), layer):
			print('Layer[%d] Node[%d]: ' % (i, j), neuron)
	return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights) - 1):
		activation += weights[i] * inputs[i]
	return activation

def forwardPropagate(network, row):
	inputs = row
	for layer in network:
		newInputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = 1 / (1 + math.exp(-activation))
			newInputs.append(neuron['output'])
		inputs = newInputs
	return inputs

def backwardPropagateError(network, expected):
	for i in range(len(network) - 1, -1, -1):
		layer = network[i]
		errors = []
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0
				for neuron in network[i + 1]:
					error += neuron['weights'][j] * neuron['delta']
				errors.append(error)
		else:
			for j in range(len(layer)):
				errors.append(expected[j] - layer[j]['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * neuron['output'] * (1 - neuron['output'])

def updateWeights(network, row, lRate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
			for neuron in network[i]:
				for j in range(len(inputs)):
					neuron['weights'][j] += inputs[j] * neuron['delta'] * lRate
				neuron['weights'][-1] += neuron['delta'] * lRate

def trainNetwork(network, dataset, lRate, nIter, nOutputs):
	for iter in range(nIter):
		sumOfErrors = 0
		for row in dataset:
			outputs = forwardPropagate(network, row)
			expected = [0 for i in range(nOutputs)]
			expected[row[-1]] = 1
			sumOfErrors += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
			backwardPropagateError(network, expected)
			updateWeights(network, row, lRate)
		print(iter, 'Error = ', sumOfErrors)

random.seed()
dataset = [ [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
#nInputs = len(dataset[0]) - 1
#nOutputs = len(set([row[-1] for row in dataset]))
network = initializeNetwork(2, 2, 2)
trainNetwork(network, dataset, 0.5, 20, 2)
print('The final neural network is')
for i, layer in zip(range(1, len(network) + 1), network):
	for j, neuron in zip(range(1, len(layer) + 1), layer):
		print('Layer[%d] Node[%d]: ' % (i, j), neuron)
