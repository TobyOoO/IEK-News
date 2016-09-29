import csv
import sqlite3

conn = sqlite3.connect('data/database.db')
c = conn.cursor()

with open('data/20160929-news.csv', 'rb') as f:
	reader = csv.reader(f)
	next(reader, None)
	for i, row in enumerate(reader):
		c.execute('INSERT INTO News VALUES (?,?,?,?,?,?,?,?,?,?)', row)
		if(i%1000==0):print('inserting %i\r'%i)