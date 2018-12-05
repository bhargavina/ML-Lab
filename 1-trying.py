import csv

with open('find-s-training-examples.csv') as file:
	data = [line for line in csv.reader(file) if line[-1] == 'Y']

print('The positive training examples are: \n{}'.format(data))

S = ['$'] * len(data[0])

print('The output at each step is \n{}'.format(S))

for example in data:
	i = 0
	for feature in example:
		S[i] = feature if S[i] == '$' or S[i] == feature else '?'
		i += 1
	print(S)