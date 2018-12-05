from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

train = fetch_20newsgroups(subset = 'train', shuffle = True)
print('The categories of 20NewsGroups are:')
for cat in train.target_names:
	print(cat)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True)
test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True)

countVectorizer = CountVectorizer()
traintf = countVectorizer.fit_transform(train.data)
print('\ntf train count:', traintf.shape)
testtf = countVectorizer.transform(test.data)
print('tf test count:', testtf.shape)

tfidftransformer = TfidfTransformer()
traintfidf = tfidftransformer.fit_transform(traintf)
print('\ntf train count:', traintf.shape)
testtfidf = tfidftransformer.transform(testtf)
print('tf test count:', testtf.shape)

model = MultinomialNB()
model.fit(traintfidf, train.target)
predicted = model.predict(testtfidf)

print('Accuracy score:', accuracy_score(test.target, predicted))
print(classification_report(test.target, predicted, target_names = test.target_names))
print('Confusion Matrix: ', confusion_matrix(test.target, predicted))
