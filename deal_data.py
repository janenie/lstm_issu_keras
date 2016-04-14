from numpy import *
import pandas as pd 
import operator
import fileinput
import sys
import re

def replaceUnknownWords(fname, fname_convert, unknown_st, unknown_ed):
	f = open(fname).read()
	fout = open(fname_convert, "w")

	fout.write(re.sub(unknown_st, unknown_ed, f))

	fout.close()


def readVocabFromFile(fname, fout_name):
	total_words = dict()
	total_sum = 0
	line_num = 0
	#total_words['</s>'] = 0
	with open(fname, 'r') as fin:
		for line in fin:
			words = line.strip().split(' ')
			
			total_sum += len(words)
			for word in words:
				if total_words.has_key(word):
					total_words[word] += 1
				else:
					total_words[word] = 1
			line_num += 1
		fin.close()

	sorted_words = sorted(total_words.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	sorted_words.insert(0, ("</s>", line_num))
	sorted_words.insert(0, ("<s>", line_num))

	with open(fout_name, 'w') as fout:
		for (k, v) in sorted_words:
			fout.write(k + "\t" + str(v) + "\t" + str(float(v)/float(total_sum))+"\n")
		fout.close()
	
	freq = [float(v)/(float(total_sum)) for (e, v) in sorted_words]
	#print sorted_words[0]
	df = pd.DataFrame(data = sorted_words, columns=['word', 'count'])
	df['freq'] = freq 
	#df.to_csv(fout_name, header=None, sep='\t', index_col=0, )

	# for line in fileinput.input([fname], inplace=True):
	# 	sys.stdout.write('{l}'.format(l=line))

if __name__ == '__main__':
	fname = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.train.txt'
	fname2 = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.train2.txt'
	fout_name = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/vocab_ptb.txt'
	replaceUnknownWords(fname, fname2, "<unk>", "UUUNKKK")
	readVocabFromFile(fname2, fout_name)
	fname_valid = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.valid.txt'
	fname_valid2 = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.valid2.txt'
	fname_test = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.test.txt'
	fname_test2 = '/Users/jan/course/cs224d/assignments/assignment2/data/lm/ptb.test2.txt'
	replaceUnknownWords(fname_valid, fname_valid2, '<unk>', 'UUUNKKK')
	replaceUnknownWords(fname_test, fname_test2, '<unk>', 'UUUNKKK')
		










