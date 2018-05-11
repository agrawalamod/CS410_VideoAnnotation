import csv
import sys
import numpy as np
import time
from collections import Counter
import os
import nltk
from math import log
import operator
import glob
import os.path, time
import csv
import numpy as np
import time
from collections import Counter
import glob
import os.path, time
import json
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from scipy import spatial
from nltk.tag import StanfordNERTagger
from math import *


def pre_process(doc):

	doc = doc.replace('"', '')
	tokens = tokenizer.tokenize(doc)
	stemmed_doc = []
	for word in tokens:
		if word not in stopwords.words('english'):
			try:
				stemmed_doc.append(stemmer.stem(word))
			except Exception, e:
				print e
				stemmed_doc.append(word)

	return stemmed_doc

def add_to_inverted_index(doc_num, doc):
	for word in doc:
		if(word in inverted_index):
			curr_list = inverted_index[word]
			all_docs = [i[0] for i in curr_list]
			if(doc_num not in all_docs):
				curr_list.append([doc_num, doc.count(word)])
				inverted_index[word] = curr_list
		else:
			inverted_index[word] = [[doc_num, doc.count(word)]]


def make_inverted_index(data):
	for line in data:
		doc_num = int(line[0])
		doc = line[1]
		stemmed_doc = pre_process(doc)
		for word in doc:
			add_to_inverted_index(doc_num, stemmed_doc)

def mainit():

	global stemmer
	global tokenizer
	global data 
	global inverted_index

	stemmer = PorterStemmer()
	tokenizer = RegexpTokenizer(r'\w+')

	data = []

	f = open('time_index.csv')
	csv_f = csv.reader(f)
	for row in csv_f:
		data.append([row[0], row[3]])

#print data

	inverted_index = {}

	make_inverted_index(data)
	print inverted_index
	with open("inverted_index.csv", "w") as csv_file:
		  writer = csv.writer(csv_file, delimiter=',')
		  for key in inverted_index.keys():
		  	line = [key, inverted_index[key]]
			writer.writerow(line)


 
