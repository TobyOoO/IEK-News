import os
import csv

folder = 'export/'
for file in os.listdir(folder):
	rw_freq = {}
	rw_list = []
	result = []
	if file.endswith(".csv"):
		with open(folder+file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = reader.next()

			for row in reader:
				rw = row[3]
				try:
					rw_freq[rw] += 1
				except:
					rw_freq[rw] = 1

		for rw, freq in rw_freq.iteritems():
			if freq > 1:
				rw_list.append(rw)

		with open(folder+file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			result.append(reader.next())

			for i, row in enumerate(reader):
				if i%500==0: print('dealing %i'%i)
				rw = row[3]
				if rw in rw_list and rw != 'UNK':
					result.append(row)

		with open('export_coexist/'+file, 'w') as output:
			writer = csv.writer(output)
			writer.writerows(result)
