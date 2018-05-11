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
import ast
import collections

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

inverted_index = {}
f = open('inverted_index.csv')
csv_f = csv.reader(f)
for row in csv_f:
	inverted_index[row[0]] = ast.literal_eval(row[1])
f.close()


documents = {}
doc_time = {}
f = open('time_index.csv')
csv_f = csv.reader(f)
for row in csv_f:
	documents[int(row[0])] = ast.literal_eval(row[3])
	doc_time[int(row[0])] = row[1]
f.close()



print inverted_index

df = {}

for word in inverted_index.keys():
	df[word] = len(inverted_index[word])

M = len(documents.keys())

query = "smoothing in naive bayes"#
query = raw_input()


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

stemmed_query = pre_process(query)


rel_docs = {}

for query_word in stemmed_query:
	if query_word in inverted_index:
		docs = inverted_index[query_word]
		print docs
		for doc in docs:
			if doc[0] not in rel_docs.keys():
				rel_docs[doc[0]] = []
				t_doc = rel_docs[doc[0]]
				t_doc.append(doc[1])
				rel_docs[doc[0]] = t_doc
			else:
				t_doc = rel_docs[doc[0]]
				t_doc.append(doc[1])
				rel_docs[doc[0]] = t_doc

print rel_docs

print "----------"
sorted_rel = sorted(rel_docs.items(), key=operator.itemgetter(0), reverse=True)
print sorted_rel[0:5]



score = {}

# for doc_num in sorted_rel:
# 	dnum = doc_num[0]
# 	freq = doc_num[1]
# 	t_score = 0
# 	for i in range(dnum-2, dnum+3):
# 		if(i in rel_docs):
# 			t_score = t_score + sum(rel_docs[i])	
# 	score[dnum] = t_score

# #print score

# sorted_score = sorted(score.items(), key=operator.itemgetter(1))

print "-----"
#print sorted_score


def bm_25(word, doc):
	global k
	global M
	k = 1.2

	c_wd = doc.count(word)
	return ((k+1)*(c_wd))/(c_wd + k)*log((M+1)/df[word])


for doc in documents.keys():
	current_doc = documents[doc]
	bm_score = 0
	for query_word in stemmed_query:
		if(query_word in current_doc):
			bm_score = bm_score + bm_25(query_word, current_doc)
	score[doc] = bm_score

sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)

print sorted_score[0:5]



print "--------------"
#print documents

print "--------------"
new_score = {}

for doc_id in range(1, M+1):
	#print "doc_id", doc_id
	t_score = 0.00
	for i in range(0, 3):
		if(doc_id+i in documents):
			#print "doc_id + i", doc_id+i
			t_score = t_score + score[doc_id+i]/(float(i)+1)
	new_score[doc_id] = t_score


#print new_score

new_sorted_score = sorted(new_score.items(), key=operator.itemgetter(1), reverse=True)
print "-------"
print new_sorted_score[0:5]

for p in new_sorted_score[0:5]:
	print "doc id: ", p[0], " time: ", doc_time[int(p[0])], " rel: ", p[1] 









