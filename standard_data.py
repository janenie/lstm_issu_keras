import sys
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense 
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM

from keras.optimizers import SGD, Adadelta, Adagrad 
from data_utils import utils as du
from numpy import *
import numpy as np 

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN, device=gpu, floatX=float32"

import theano
import keras


def getVocab(fname):
	vocab = []
	words = []
	count = []
	with open(fname, 'r') as fin:
		for line in fin:
			tuples = line.split('\t')
			vocab.append((tuples[0], int(tuples[1]), float(tuples[2])))
			words.append(str(tuples[0]))
			count.append(int(tuples[1]))
    	fin.close()
	return vocab, words, count 

def len_argsort(seq):
	return sorted(range(len(seq)), key= lambda x: len(seq[x]))

def splitData(data_, pices):
	total_len = len(data_)
	batches = total_len / pices

	ret = []
	st = 0
	for i in xrange(pices-1):
		ret.append(data_[st:st+batches])
		st += batches
	ret.append(data_[st:])
	return ret 

def preprocessData(x_train, y_train, inputdim, maxLen=100, batch_size=10):
	length = len(x_train)
	pad_x_train = sequence.pad_sequences(x_train, maxlen=maxLen, padding='pre', value=0)
	pad_y_train = sequence.pad_sequences(y_train, maxlen=maxLen, padding='pre', value=0)
	y_ret = np.zeros((length, maxLen, inputdim))
	for i in xrange(length):
		for t, char in enumerate(pad_y_train[i]):
			y_ret[i, t, char] = 1
	print "the shape of the preprocessd matrix is " + str(np.array(pad_x_train).shape)
	return np.array(pad_x_train), y_ret
	# x_input = np.zeros((length, maxLen, inputdim))
	# y_input = np.zeros((length, maxLen, inputdim))

	# for i in xrange(length):
	# 	for t, char in enumerate(x_train[i]):
	# 		x_input[i, t, char] = 1
	# 	for t2, char2 in enumerate(y_train[i]):
	# 		y_input[i, t2, char2] = 1

	# return x_input, y_input

class LSTMLM:
	def __init__(self, input_len, hidden_len, output_len, return_sequences=True):
		self.input_len = input_len
		self.hidden_len = hidden_len
		self.output_len = output_len
		self.seq = return_sequences
		self.model = Sequential()

	#simple LSTM layer
	def build(self, maxlen=50, dropout=0.2):
		self.model.add(Embedding(self.input_len, self.hidden_len, input_length=maxlen))
		self.model.add(LSTM(output_dim=self.hidden_len, return_sequences=True))
		#self.model.add(Dropout(dropout))
		self.model.add(TimeDistributedDense(self.output_len))
		self.model.add(Activation('softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	def train(self, train_x, train_y, dev_x, dev_y, batchsize=16, epoch=1):
		hist = self.model.fit(train_x, train_y, batch_size=batchsize, nb_epoch=1, \
						show_accuracy=True, validation_data=(dev_x, dev_y))
		print hist.history

	def saveModel(self, save_file):
		self.model.save_weights(save_file)

	def computePPL(self, test_x, test_y, maxLen=100):
		predictions = self.model.predict(test_x)
		#self.model.evaluate(test_x, test_y)
		#test_y = sequence.pad_sequences(test_y, maxlen=maxLen, padding='pre', value=0)
		#print predictions.shape

		test_num = len(test_y)
		maxlen = predictions.shape[1]
		print "the total probability for vocab size is " + str(sum(predictions[0][0]))
		ppl = 0.0
		total_words = 0
		# st = True
		for i in xrange(test_num):
			pred_idx = test_y[i]
			#print np.max(pred_idx)
			sent_len = len(test_y[i])
			padding = maxlen - sent_len
			# if st :
			# 	print predictions[i][0][pred_idx[0]]
			# 	st = False
			for j in xrange(sent_len):
				prob_ = predictions[i, j+padding, pred_idx[j]]
				ppl -= np.log(prob_)
			total_words += sent_len
			#print ppl
		# print ppl
		return np.exp(ppl)/(float(total_words))
		#return ppl, total_words


if __name__ == '__main__':
	folder = 'data/'
	vocab_file = 'data/vocab_ptb.txt'
	train_file = folder + 'ptb.train2.txt'
	test_file = folder + 'ptb.test2.txt'
	valid_file = folder + 'ptb.valid2.txt'

	weight_file = folder + 'train_model_weight.h5'

	vocab, words, count = getVocab(vocab_file)
	vocabsize = 10000
	num_to_word = dict(enumerate(words[:vocabsize]))
	word_to_num = du.invert_dict(num_to_word)

	unknown_words = word_to_num.get('UUUNKKK')

	docs = du.load_modified_dataset(train_file)
	S_train = du.docs_to_index(docs, word_to_num)
	X_train, Y_train = du.seqs_to_lmXY(S_train)

	test_x = np.array(X_train)
	lengths = [len(s) for s in X_train]

	#train_idf = du.get_tfidf(vocabsize, vocab, S_train)
	#print train_idf[:20]

	# Load the dev set (for tuning hyperparameters)
	docs = du.load_modified_dataset(valid_file)
	S_dev = du.docs_to_index(docs, word_to_num)
	X_dev, Y_dev = du.seqs_to_lmXY(S_dev)

	# Load the test set (final evaluation only)
	docs = du.load_modified_dataset(test_file)
	S_test = du.docs_to_index(docs, word_to_num)
	X_test, Y_test = du.seqs_to_lmXY(S_test)

	#sort data by length
	sorted_indexs = len_argsort(X_train)
	X_train = [X_train[i] for i in sorted_indexs]
	Y_train = [Y_train[i] for i in sorted_indexs]

	sorted_indexs = len_argsort(X_dev)
	X_dev = [X_dev[i] for i in sorted_indexs]
	Y_dev = [Y_dev[i] for i in sorted_indexs]

	sorted_indexs = len_argsort(X_test)
	X_test = [X_test[i] for i in sorted_indexs]
	Y_test = [Y_test[i] for i in sorted_indexs] 

	lengths = [len(s) for s in X_train]
	maxlen = np.max(lengths)
	#print "longest sentence is %d" %(maxlen)
	samples = len(S_train)

	x_train_split = splitData(X_train, 10)
	y_train_split = splitData(Y_train, 10)
	x_dev_split = splitData(X_dev, 10)
	y_dev_split = splitData(Y_dev, 10)
	x_test_split = splitData(X_test, 10)
	y_test_split = splitData(Y_test, 10)

	data_dim = 100
	dropout = 0.5

	print 'Start building...'
	lstm = LSTMLM(vocabsize, data_dim, vocabsize, return_sequences=True)
	lstm.build(maxlen=maxlen, dropout=0.5)

	# lstm.train(x_train_lm, y_train_lm, x_dev_lm, y_dev_lm)
	# lstm.save_file(weight_file)
	# print lstm.computePPL(x_test_lm, y_test_lm)
	x_test_lm, y_test_lm = preprocessData(x_test_split[0], y_test_split[0], vocabsize, maxLen=maxlen)
	print "test y shape is " + str(y_test_lm.shape)
	for i in xrange(10):
		x_train_lm, y_train_lm = preprocessData(x_train_split[i], y_train_split[i], vocabsize, maxLen=maxlen) 
		x_dev_lm, y_dev_lm = preprocessData(x_dev_split[i], y_dev_split[i], vocabsize, maxLen=maxlen)
		x_test_lm, y_test_lm = preprocessData(x_test_split[i], y_test_split[i], vocabsize, maxLen=maxlen)
		lstm.train(x_train_lm, y_train_lm, x_dev_lm, y_dev_lm)
		print lstm.computePPL(x_test_lm, y_test_lm, maxlen)

	lstm.save_file(weight_file)

	x_test_lm, y_test_lm = preprocessData(X_test, Y_test, vocabsize, maxLen=maxlen)
	print lstm.computePPL(x_test_lm, y_test_split[i], maxlen)

	# x_train_lm, y_train_lm = preprocessData(X_train, Y_train, vocabsize, maxLen=maxlen)
	# x_dev_lm, y_dev_lm = preprocessData(X_dev, Y_dev, vocabsize, maxLen=maxlen)
	# x_test_lm, y_test_lm = preprocessData(X_test, Y_test, vocabsize, maxLen=maxlen)

	#bptt_steps = 10

	# train_model = Sequential()
	# #train_model.add(Embedding(vocabsize, data_dim, input_length=maxlen))
	# train_model.add(LSTM(output_dim=data_dim, return_sequences=True, activation='sigmoid',\
	# 		 inner_activation='hard_sigmoid', input_shape=(maxlen, vocabsize)))
	# #train_model.add(Dropout(dropout))
	# train_model.add(TimeDistributedDense(output_dim=vocabsize))
	# train_model.add(Activation('softmax'))
	# Ada = Adagrad(lr=0.1, epsilon=1e-6)
	# train_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	# hist = train_model.fit(x_train_lm, y_train_lm, batch_size=10, nb_epoch=10, show_accuarcy=True, validation_data=(x_dev_lm, y_dev_lm))
	# print (hist.history)

	# train_model.save_weights(weight_file)

	#train_model.evaluate()
	print 'training finished...'







