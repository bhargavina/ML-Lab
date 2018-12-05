import csv
import math

def mean(numbers):
	return sum(numbers) / len(numbers)

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x - avg, 2) for x in numbers]) / (len(numbers) - 1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def calcProb(summary, item):
	prob = 1
	for i in range(len(summary)):
		x = item[i]
		mean, stdev = summary[i]
		exponent = math.exp(-pow(x - mean, 2) / (2 * stdev ** 2))
		final = exponent / (math.sqrt(2 * math.pi) * stdev)
		prob *= final
	return prob

with open('naive-bayes-training-examples.csv') as file:
	data = [line for line in csv.reader(file)]
for i in range(len(data)):
	data[i] = [float(x) for x in data[i]]

split = int(0.90 * len(data))
train = data[: split]
test = data[split: ]

print('\nTotal number of hypotheses:', len(data))
print('Number of hypotheses in training data:', len(train))
print('Number of hypotheses in test data:', len(test))
print("\nThe values assumed for the concept learning attributes are:")
print("OUTLOOK: Sunny = 1, Overcast = 2 and Rain = 3\nTEMPERATURE: Hot = 1, Mild = 2 and Cool = 3\nHUMIDITY: High = 1 and Normal = 2\nWIND: Weak = 1 and Strong = 2")
print("TARGET CONCEPT: PlayTennis where Yes = 10 and No = 5")

print("\nTraining dataset:")
for x in train:
	print(x)
print("\nTest dataset:")
for x in test:
	print(x)

yes = []
no = []
for i in range(len(train)):
	if data[i][-1] == 5.0:
		no.append(data[i])
	else:
		yes.append(data[i])

yes = summarize(yes)
no = summarize(no)

predictions = []
for item in test:
	yesProb = calcProb(yes, item)
	noProb = calcProb(no, item)
	predictions.append(10.0 if yesProb > noProb else 5.0)

correct = 0
for i in range(len(test)):
	if(test[i][-1] == predictions[i]):
		correct += 1

print("\nActual values are:")
for i in range(len(test)):
	print(test[i][-1], end = " ")
print("\nPredicted values are:")
for i in range(len(predictions)):
	print(predictions[i], end = " ")
print("\nAccuracy is %.1f%%" % (correct / len(test) * 100))
