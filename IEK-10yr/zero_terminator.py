'''
import sys

reload(sys)
sys.setdefaultencoding('utf8')
'''

import csv
import numpy as np
import sklearn.preprocessing

filename = '大於50-常態化-20161002-IEK-偷比-word2vec-後期-matrix-R-termMatrix-M-1-長資料-m.csv'

header = []
output = []
scaled_weight = {}

np.set_printoptions(precision=4)

with open(filename, 'r', encoding='big5') as f:
	reader = csv.reader(f)
	header = next(reader, None)
	for i, row in enumerate(reader):
		if(i%100000==0):print('dealing %i'%i)
		if(row[3] != '0'):
			try:
				scaled_weight[row[2]].append(row[3])
			except:
				scaled_weight[row[2]] = [row[3]]
			output.append(row[:4])

for var, weight_list in scaled_weight.items():
	scalar = sklearn.preprocessing.MinMaxScaler(feature_range=(1,100))
	scalar.fit(np.array(weight_list).reshape(-1,1))
	scaled_weight[var] = scalar.transform(np.array(weight_list).reshape(-1,1)).tolist()

with open('再常態化-'+filename, 'w', encoding='big5') as f:
	writer = csv.writer(f)
	writer.writerow(header+['scaled_weight'])
	j = 0
	for i, row in enumerate(output):
		var = row[2]
		writer.writerow(row+scaled_weight[var][j])
		if j+1 == len(scaled_weight[var]):
			j = 0
		else:
			j += 1
