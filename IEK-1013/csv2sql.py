import csv
import sqlite3

conn = sqlite3.connect('data/database.db')
c = conn.cursor()

with open('data/20161013-IEK-時報-更正-年度月日排序.csv', 'r') as f:
	reader = csv.reader(f)
	next(reader, None)
	for i, row in enumerate(reader):
		c.execute('INSERT INTO News VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [None]+row[0:12])
		if(i%1000==0):
			print('inserting %i\r'%i)
			conn.commit()
conn.close()