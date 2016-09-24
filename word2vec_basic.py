# coding=UTF8
# encoding=utf8

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
# import tensorflow.python.platform
import collections
import math
import numpy as np
# import os
import random
# from six.moves import urllib
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import codecs
import jieba.analyse
import sys
import re
import sklearn.preprocessing
from numpy import array
import chroma
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cmx

import sqlite3

reload(sys)
sys.setdefaultencoding('utf8')

#const
top_n_articles = 20
vocabulary_size = 3000 #top n words to be trained
#num_steps = 10001 #training steps, ex. 100001
num_steps = 100001
plot_only = 500 #ploting data pts
output_model = "model/tars-mode-c-v4"
output_filename = "img/tsne-c4.png"

# Read the data into a string.
def read_data():
	jieba.set_dictionary('dict.txt.big')
	stop1 = u"[，。、「」（）\(\)\.【】『』：；・．～？＝＼／！＠\@＄\$％＆\&\%\-\/\\\>\<\~\:\,\[\]\?\!\=\+＋\*＊]*"
	stop2 = "http[a-zA-Z\.\:\/\-\?\#]*"
	stop3 = "[0-9\.]*"
	result = []

	word_cate = {}
	color_dict = {}
	
	'''
	#read color dict
	header = True
	f = codecs.open('category_color.csv', 'r', encoding='utf8')
	for row in f:
		if(header):
			header = False
			continue
		row = row.replace('"', '').split(',')
		color_dict[row[0]] = None

	arr = color_dict.keys()
	cNorm  = colors.Normalize(vmin=-0.1, vmax=1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('nipy_spectral') )
	for i in range(len(arr)):
		color = scalarMap.to_rgba(i/float(len(arr)))
		color_dict[arr[i]] = chroma.Color(color, format="RGB")
		print(arr[i]+', '+str(color_dict[arr[i]]))
	'''

	#read article
	conn = sqlite3.connect('ptt_course.db')
	c = conn.cursor()


	for i, article in enumerate(c.execute('SELECT Content FROM Article_Directory WHERE Category = "評價" AND Content not null')):
		print('reading %i' % i)
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
	'''
	for partition in range(0, 10000, 1000):
		count = 0
		filename = 'article_v2_%s_%s.arff' % (str(partition), str(partition+1000))
		print(filename)
		f = codecs.open(filename, 'r', encoding='utf8')
		for row in f:
			if(row[0]=='@'): continue
			#if(count > 100): break
			if(count > top_n_articles): break
			print("T: "+row.split(',')[4].replace('"', ''))
			temp_cate_list = row.split(',')[1].replace('"', '').split(';')
			cate_list = []
			for cate in temp_cate_list:
				if cate in color_dict.keys():
					cate_list.append(cate)
			text = row.split(',')[5].replace('"', '')
			#print(text)
			text = re.sub(stop1, "", text)
			text = re.sub(stop2, "_URL_", text)
			text = re.sub(stop3, "", text)
			seg_list = jieba.lcut(text, cut_all=False)
			word_list = []
			for seg in seg_list:
				if(len(seg) == 1):
					continue
				word_list.append(seg)
			#print("/ ".join(seg_list))

			score_list = jieba.analyse.textrank(text, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
			#here to normalize
			key_word_list = []
			norm_score_list = []

			for word, score in score_list:
				if (len(word) == 1):
					continue
				key_word_list.append(word)
				norm_score_list.append([score])

			a = array(norm_score_list, dtype='f')
			if(len(norm_score_list)>0):
				norm_score_list = sklearn.preprocessing.normalize(a, axis=0)

			for i in range(len(score_list)):
				word = key_word_list[i]
				score = norm_score_list[i][0]
				for cate in cate_list:
					try:
						word_cate[word][cate] += [score]
					except:
						try:
							word_cate[word][cate] = [score]
						except:
							word_cate[word] = {}
							word_cate[word][cate] = [score]

			result = result + word_list
			count += 1

	word_color = {}
	for word, cate_list in word_cate.iteritems():

		color_list = []
		for cate, score_list in cate_list.iteritems():
			score = sum(score_list) / float(len(score_list))
			#color = chroma.Color(color_dict[cate])
			color = color_dict[cate]
			color.alpha = score
			color_list.append(color)

		word_color[word] = chroma.Color('#000000')
		for color in color_list:
			word_color[word] += color

		if(word_color[word]==chroma.Color('#000000')):
			word_color[word] = chroma.Color('#FFFFFF')
	'''

	return result

words = read_data()
print('Data size', len(words))
# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = 1000

def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary) - 1
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0	# dictionary['UNK']
			unk_count = unk_count + 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words	# Hint to reduce memory.
print('Most common words (+UNK)', count[:10])
print('Sample data', data[:20])
data_index = 0
# Step 4: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window	# target label at the center of the buffer
		targets_to_avoid = [ skip_window ]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
	print(batch[i], '->', labels[i, 0])
	print(reverse_dictionary[batch[i]])
	print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])
# Step 5: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128	# Dimension of the embedding vector.
skip_window = 1			 # How many words to consider left and right.
num_skips = 2				 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16		 # Random set of words to evaluate similarity on.
valid_window = 100	# Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
num_sampled = 64		# Number of negative examples to sample.
graph = tf.Graph()
with graph.as_default():
	# Input data.
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device('/gpu:0'):
		# Look up embeddings for inputs.
		embeddings = tf.Variable(
				tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(
				tf.truncated_normal([vocabulary_size, embedding_size],
														stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	loss = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
										 num_sampled, vocabulary_size))
	# Construct the SGD optimizer using a learning rate of 1.0.
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
	# Compute the cosine similarity between minibatch examples and all embeddings.
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(
			normalized_embeddings, valid_dataset)
	similarity = tf.matmul(
			valid_embeddings, normalized_embeddings, transpose_b=True)
# Step 6: Begin training
#num_steps = 100001


with tf.Session(graph=graph) as session:
	# We must initialize all variables before we use them.
	tf.initialize_all_variables().run()
	saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
	print("Initialized")
	average_loss = 0
	for step in xrange(num_steps):
		batch_inputs, batch_labels = generate_batch(
				batch_size, num_skips, skip_window)
		feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
		# We perform one update step by evaluating the optimizer op (including it
		# in the list of returned values for session.run()
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val
		if step % 2000 == 0:
			if step > 0:
				average_loss = average_loss / 2000
			# The average loss is an estimate of the loss over the last 2000 batches.
			print("Average loss at step ", step, ": ", average_loss)
			saver.save(session, output_model, global_step=step)
			average_loss = 0
		# note that this is expensive (~20% slowdown if computed every 500 steps)
		if step % 10000 == 0:
			sim = similarity.eval()
			for i in xrange(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8 # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[1:top_k+1]
				log_str = "Nearest to %s:" % valid_word
				for k in xrange(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = "%s %s," % (log_str, close_word)
				print(log_str)
	final_embeddings = normalized_embeddings.eval()
# Step 7: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename=output_filename):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	font = FontProperties(fname="NotoSansCJKtc-Medium.otf", size=14) 
	plt.figure(figsize=(32, 32))	#in inches
	plt.axis([-30, 30, -30, 30])
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		kwcolor = 'black'
		plt.scatter(x, y)
		plt.annotate(label,
								 xy=(x, y),
								 xytext=(5, 2),
								 textcoords='offset points',
								 ha='right',
								 va='bottom',
								 fontproperties=font,
								 color = kwcolor)
	'''
	count = 0
	for word, color in color_dict.iteritems():
		plt.text(-29, 25-1*count, word, ha='left', va='bottom', fontproperties=font, color=str(color)[:7])
		count += 1
	'''
	plt.savefig(filename)
try:
	from sklearn.manifold import TSNE
	from matplotlib.font_manager import FontProperties
	import matplotlib
	matplotlib.use('Agg') #no display
	import matplotlib.pyplot as plt
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
	labels = [reverse_dictionary[i] for i in xrange(plot_only)]
	plot_with_labels(low_dim_embs, labels)
except ImportError:
	print("Please install sklearn and matplotlib to visualize embeddings.")