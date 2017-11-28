import json
import numpy as np 
#import nltk
import string
import numpy as np 
import math
import os.path


REBUILD_DATA = True
REVERSE = True
VOCABULARY_SIZE = 10000
FULL_VSIZE = VOCABULARY_SIZE+4

START_SYMBOL = '_START'
STOP_SYMBOL = '_STOP'
PAD_SYMBOL = '_PAD'
UNK_SYMBOL = '_UNK'

# returns a list of dictionaries
# each dictionary has three keys: date, author, text
# each entry in the list corresponds to a line in the conversation
def load_conversation(fname = './data/full_conversation.json'):
	with open(fname, 'r') as f:
		return json.load(f)

# conversation is a list of dictionaries, each dictionary containing
# a conversation line
def clean_text(conversation):
	for item in conversation:
		if item:
			# get the text to be preprocessed
			text = item['text']
			# strip away punctuation
			text = ''.join(c for c in text if c not in string.punctuation)
			#strip away digits
			text = ''.join(c for c in text if c not in '0123456789')
			text = text.lower()
			item['text'] = text 
		else:
			conversation.remove(item)
	return conversation

def build_vocabulary(conversation):
	vocabulary = {}
	for item in conversation:
		if item:
			text = item['text'].split()
			for word in text:
				if word in vocabulary.keys():
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1

	return vocabulary

def cut_vocabulary(vocabulary, vocab_size = VOCABULARY_SIZE):
	words = [[x, vocabulary[x]] for x in vocabulary.keys()]
	words = sorted(words, key = lambda x: x[1], reverse = True)
	words = words[:vocab_size]
	words = [[words[i][0],i+1] for i in range(vocab_size)]
	words.append([UNK_SYMBOL, vocab_size+1])
	words.append([START_SYMBOL, vocab_size+2])
	words.append([STOP_SYMBOL, vocab_size+3])
	words.append([PAD_SYMBOL, 0])
	vocabulary = dict(words)
	with open('./data/vocabulary.json', 'w') as f:
			json.dump(vocabulary, f)
	vocab_size = VOCABULARY_SIZE + 4
	return vocabulary

def numify_sentence(sentence, vocabulary):
	words = sentence.split()
	n_words = [vocabulary[x] if x in vocabulary.keys() else vocabulary[UNK_SYMBOL] for x in words]

	return n_words

def build_numeric_sentences(conversation, vocabulary):
	ready_to_process_sentences = []
	for item in conversation:
		if item:
			text = item['text']
			if len(text) == 0:
				continue
			n_text =  numify_sentence(text, vocabulary)
			if len(n_text)==0:
				continue
			item['n_text'] = n_text

			ready_to_process_sentences.append(item)

	with open('./data/full_conversation.json', 'w') as f:
			json.dump(ready_to_process_sentences, f)
	return ready_to_process_sentences

def find_min_max_len(conversation):

	ls = [len(x['n_text']) for x in conversation]

	return (min(ls),max(ls))



def create_q_and_a(conversation, vocabulary):
	questions = {}
	answers = {}

	index = 0
	q_index = 0
	a_index = 0
	for item in conversation:
		if index%2 == 0:
			question = item['n_text'].copy()
			if REVERSE:
				question = list(reversed(question))
			question.extend([vocabulary[STOP_SYMBOL]])
			questions[q_index] = question
			q_index+=1
		else:
			answer = item['n_text'].copy()
			answer.insert(0,vocabulary[START_SYMBOL])
			answer.extend([vocabulary[STOP_SYMBOL]])
			answers[a_index] = answer
			a_index+=1
		index+=1
	print(len(questions))
	with open('./data/questions.txt', 'w') as file:
		for i in range(q_index-1):
			print(questions[i], file = file)
	with open('./data/answers.txt', 'w') as file:
		for i in range(a_index-1):
			print(answers[i], file = file)

	return (questions, answers)

	
def create_text_q_and_a(questions, answers, rev_vocabulary):
	q = [' '.join(rev_vocabulary[word_id] for word_id in questions[i]) 
		for i in range(len(questions))]

	a = [' '.join(rev_vocabulary[word_id] for word_id in answers[i]) 
		for i in range(len(answers))]

	with open('./data/questions.txt', 'w') as file:
		for i in range(len(q)):
			print(q[i], file = file)
	with open('./data/answers.txt', 'w') as file:
		for i in range(len(a)):
			print(a[i], file = file)

def build_target_sentences(answers, rev_vocabulary):
	a = [' '.join(rev_vocabulary[word_id] for word_id in answers[i] 
		if rev_vocabulary[word_id] != START_SYMBOL) 
		for i in range(len(answers))]
	a = [x + STOP_SYMBOL for x in a]
	with open('./data/targets.txt', 'w') as file:
		for i in range(len(a)):
			print(a[i], file = file)





def build_q_a_dictionary_list(questions, answers):
	q_a_dictionary_list = [{'q': questions[i], 
		'a': answers[i],
		'length': (len(questions[i]),len(answers[i]))} 
		for i in range(len(questions))]

	with open('./data/q_a_dictionary_list.json', 'w') as f:
			json.dump(q_a_dictionary_list, f)
	return q_a_dictionary_list


def build_reverse_dictionary(vocabulary):
	r_dict = dict([[vocabulary[x],x] for x in vocabulary.keys()])
	return r_dict

def write_plain_vocabulary(vocabulary):
	keys = [[x, vocabulary[x]] for x in vocabulary.keys()]
	keys = sorted(keys, key = lambda x: x[1])
	with open('./data/plain_vocabulary.txt', 'w') as f:
		for key in keys:
			print(key[0], file = f)

# This function one hot encodes a sentence given a fixed vocabulary 
# size. The encoding is returned in time major, i.e.,
# [[w_0][w_1]...[w_n]] where w_i are column vectors of one hot encoded words
def one_hot_encode(sentence, vocab_size):
	encoded_sentence = np.zeros((vocab_size, len(sentence)))
	j = 0
	for i in sentence:
		encoded_sentence[i,j] = 1
		j+=1
	return encoded_sentence



def load_q_a_dictionary_list(fname = './data/q_a_dictionary_list.json'):
	with open(fname, 'r') as f:
		return	json.load(f)

def load_vocabulary(fname = './data/vocabulary.json'):
	with open(fname, 'r') as f:
		return json.load(f)

def load_full_conversation(fname = './data/full_conversation.json'):
	with open(fname, 'r') as f:
		return json.load(f)

def initialize_data_utils():

	if os.path.isfile('./data/vocabulary.json') and \
		os.path.isfile('./data/full_conversation.json') and \
		not REBUILD_DATA:

		vocabulary = load_vocabulary()
		full_conversation = load_full_conversation()
	else:
		conv = load_conversation()
		conv = clean_text(conv)
		vocabulary = build_vocabulary(conv)
		vocabulary = cut_vocabulary(vocabulary)
		full_conversation = build_numeric_sentences(conv, vocabulary)

	r_vocabulary = build_reverse_dictionary(vocabulary)
	
	if os.path.isfile('./data/q_a_dictionary_list.json') and \
		not REBUILD_DATA:

		print('Loading padded questions and answers structure')
		q_a_dictionary_list = load_q_a_dictionary_list()
	else:
		min_len, max_len = find_min_max_len(full_conversation)
		q,a = create_q_and_a(full_conversation, vocabulary)
		q_a_dictionary_list = build_q_a_dictionary_list(q, a)

	questions = [x['q'] for x in q_a_dictionary_list]
	answers = [x['a'] for x in q_a_dictionary_list]
	create_text_q_and_a(questions,answers,r_vocabulary)
	build_target_sentences(answers, r_vocabulary)
	write_plain_vocabulary(vocabulary)

	return (vocabulary, r_vocabulary, q_a_dictionary_list)


if __name__ == '__main__':

	(vocabulary, rev_vocabulary, q_a_dictionary_list) = initialize_data_utils()
	#a = np.array([x['q'] for x in q_a_dictionary_list])
	#print(a)
