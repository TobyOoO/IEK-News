import csv
import sqlite3

conn = sqlite3.connect('data/database.db')
c = conn.cursor()

with open('data/bigfile/news_raw_data.csv', 'r') as f:
	reader = csv.reader(f)
	next(reader, None)
	for i, row in enumerate(reader):
		c.execute('INSERT INTO News VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', [None]+row)
		if(i%1000==0):
			print('inserting %i\r'%i)
			conn.commit()
conn.close()
