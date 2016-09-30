import csv
import os
from openpyxl import Workbook

folder = 'export_coexist/'
wb = Workbook()
dest_filename = r"export_coexist.xlsx"

for i, file in enumerate(sorted(os.listdir(folder))):
	rw_freq = {}
	rw_list = []
	result = []
	if file.endswith(".csv"):
		with open(folder+file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			ws = wb.create_sheet(file.replace('wordlist_','').replace('.csv',''))

			for row_index, row in enumerate(reader):
				ws.append(row)

wb.save(filename = dest_filename)