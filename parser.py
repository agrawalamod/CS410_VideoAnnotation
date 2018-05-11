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

def read_file(filename):
	with open(filename) as f:
		 content = f.readlines()
	return content



def parse(filename):
	subs = read_file(filename)
	#for index, line in enumerate(subs):
		#print index, line
	#print subs
	print "-----------------------"
	time_index = {}

	flag = 0
	for line in subs:
		line = line.strip('\n')
		if('-->' in line):
			if(flag==0):
				#start recording
				flag = 1
				timestamps = line.split('-->')
				timestamps[0]=timestamps[0].strip(' ').replace(' ','')
				timestamps[1]=timestamps[1].strip('\n').replace(' ','')
				content = ""
			elif(flag==1):
				#end recording, start new
				time_index[(timestamps[0],timestamps[1])] = str(content)
				#print time_index[(timestamps[0],timestamps[1])]
				timestamps = line.split('-->')
				timestamps[0]=timestamps[0].strip(' ').replace(' ','')
				timestamps[1]=timestamps[1].strip('\n').replace(' ','')
				content = ""
			else:
				print "something went very wrong"
		else:
			if(flag==1):
				#add to record
				print line
				if(line != "[SOUND]" and not line.isdigit() and line != ''):
					content = content + ' ' + line.replace('\n','').replace('"', '')
				else:
					print line, "useless"
			else:
				#ignore out
				print line, "ignored"	


	print "---------------"
	print time_index


	sorted_time = sorted(time_index.items(), key=operator.itemgetter(0))


	print "------------------ Done -------------------"
	for i in sorted_time:
		print i

	with open("time_index.csv", "w") as csv_file:
			  writer = csv.writer(csv_file, delimiter=',')
			  index = 1

			  for i in sorted_time:
			  	line = [index, i[0][0], i[0][1], pre_process(i[1])]
			  #for key in time_index.keys():
			  	#line = [index, key[0].split('.')[0], key[1].split('.')[0], time_index[key]]
				writer.writerow(line)
				index = index + 1


