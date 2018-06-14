# Word2vVec Code

# Importing the Libraries
import re
import nltk
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import nltk.data
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from bs4 import BeautifulSoup
from gensim.models import word2vec

# Importing data from files

def import_data():

	labeled_training_data = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
	unlabeled_training_data = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
	test_data = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting =3)

	return labeledTrainData, unlabeledTrainData, test_data

# Function parses sentence and returns a list of words

def sentence_parser(sentence_data):

	sentence_data = BeautifulSoup(sentence_data).get_text()
	sentence_data = re.sub("[^a-zA-Z]"," ", sentence_data)
	word_list = sentence_data.lower().split()
	return word_list


def text_parser(text_data):

	nltk.download()
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	raw_sentences = tokenizer.tokenize(review.strip())	
	sentences = []

	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(sentence_parser(raw_sentence))

	return sentences

def input_data(labeled_training_data, unlabeled_training_data):

	sentences = []

	for review in labeled_training_data["review"]:
		sentences += text_parser(review)

	for review in unlabeled_training_data["review"]:
		sentences += text_parser(review)

	return sentences

def train_model(input_data, hyper_parameters):

	model = word2vec.Word2Vec(input_data, workers=hyper_parameters[2], min_count=hyper_parameters[1], size=hyper_parameters[0], window = hyper_parameters[3], sample = hyper_parameters[4])
	model.init_sims(replace = True)
	model_name = "word2vec_model"
	model.save(model_name)
	

