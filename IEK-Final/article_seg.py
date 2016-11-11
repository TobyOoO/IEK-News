# coding=UTF8
# encoding=utf8

import jieba
import sqlite3
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')

conn = sqlite3.connect('data/bigfile/database.db')
c = conn.cursor()

def read_distinct_fields():
	c.execute('SELECT distinct Topic FROM News')
	return c.fetchall()

def read_data():
	jieba.set_dictionary('dict.txt.big')
	stop1 = u"[，。、「」（）\(\)\.【】『』：；・．～？＝＼／！＠\@＄\$％＆\&\%\-\/\\\>\<\~\:\,\[\]\?\!\=\+＋\*＊]*"
	stop2 = "http[a-zA-Z\.\:\/\-\?\#]*"
	stop3 = "[0-9\.]*"
	

	#read article
	topic = read_distinct_fields()

	for t in topic:
		t = t[0]
		result = []
		print('dealing topic %s'%t)
		for half in range(2):
			if half == 0:
				query=c.execute(u'SELECT Title FROM News WHERE Title not null and Topic=? and Period=前期', (t,))
			else:
				query=c.execute(u'SELECT Title FROM News WHERE Title not null and Topic=? and Period=後期', (t,))
			for i, article in enumerate(query):
				if(i%1000==0):print('reading %i' % i)
				result += ['UNK']
				text = article[0].replace('  ', '')
				text = re.sub(stop1, "", text)
				text = re.sub(stop2, "_URL_", text)
				text = re.sub(stop3, "", text)
				seg_list = jieba.lcut(text, cut_all=False)
				word_list = []
				for seg in seg_list:
					if(len(seg) == 1):
						continue
					word_list.append(seg)
				result += word_list
			with open('seg/article_seg_%s_%i.txt'%(t.replace('/','_'),half), 'wb') as f:
				f.write(' '.join(result))

read_data()
conn.close()