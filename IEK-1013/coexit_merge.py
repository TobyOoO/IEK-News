# -*- coding: utf-8 -*-
import os
import csv

folder = 'export_coexist/'
output_0 = []
output_1 = []
header = []
for file in os.listdir(folder):
	rw_freq = {}
	rw_list = []
	result = []
	if file.endswith(".csv"):
		with open(folder+file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = reader.next()
			#print(reader)
			if file.endswith('_0.csv'):
				target = output_0
			else:
				target = output_1
			for row in reader:
				target.append(row)


for i in range(2):
	with open('output_%i.csv'%i, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(header)
		if i == 0:
			writer.writerows(output_0)
		else:
			writer.writerows(output_1)
