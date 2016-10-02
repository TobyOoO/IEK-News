# coding=UTF8
# encoding=utf8
import csv
import json
import sklearn.preprocessing
import numpy as np

word2id={}
result=[]
keyword=[]
normalword=[]
field='society-2'
with open('plot/%s.csv'%field, 'rb') as f:
	counter = 0
	reader = csv.reader(f)
	for row in reader:
		if(float(row[1])==100.0):
			word = row[0]
			word2id[word] = counter
			keyword.append(word)
			counter+=1
	f.seek(0)
	for row in reader:
		if(float(row[1])<100):
			word = row[0]
			word2id[word] = counter
			normalword.append(word)
			counter+=1

	for i in range(counter):
		result.append([ 0 for j in range(counter) ])

	i=0
	f.seek(0)
	for row in reader:
		val = float(row[1])
		if(val==100.0):
			i=word2id[row[0]]
		if(val<100):
			j=word2id[row[0]]
			result[i][j]=val

np_result = sklearn.preprocessing.normalize(np.array(result))

with open('plot/%s_words.csv'%field, 'wb') as f:
	f.write('name,color\n')
	for word in keyword:
		f.write('{0},{1}\n'.format(word,'#E41A1C'))
	for word in normalword:
		f.write('{0},{1}\n'.format(word,'#FFFF33'))

with open('plot/%s_matrix.json'%field, 'wb') as f:
	f.write(json.dumps(np_result.tolist()))