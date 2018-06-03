# Bag of Words model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('labeledTrainData.tsv', delimiter = '\t', header = 0, quoting = 3)
test_dataset = pd.read_csv('testData.tsv', delimiter = '\t', header = 0)


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
test_corpus = []
for i in range(0, 25000):
	review = BeautifulSoup(dataset['review'][i], "html5lib").get_text()
	test_review = BeautifulSoup(test_dataset['review'][i], "html5lib").get_text()
	review = re.sub('[^a-zA-Z]', ' ', review)
	test_review = re.sub('[^a-zA-Z]', ' ', test_review)
	review = review.lower()
	test_review = test_review.lower()
	review = review.split()
	test_review = test_review.split()
	ps     = PorterStemmer()
	stops  = set(stopwords.words('english'))
	review = [ps.stem(word) for word in review if not word in stops]
	test_review = [ps.stem(word) for word in test_review if not word in stops]
	review = ' '.join(review)
	test_review = ' '.join(test_review)
	corpus.append(review)
	test_corpus.append(test_review)



# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
X_test = cv.transform(test_corpus).toarray()


# Using xgboost model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, dataset['sentiment'])


# Prediction
result = classifier.predict(X_test)

output = pd.DataFrame(data={"id": test_dataset["id"], "sentiment": result})
output.to_csv("submission.csv", index = false, quoting = 3)



