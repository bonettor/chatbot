import numpy as np
import random
import sys

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
 
import data_utils as du

# I HAVE 58508 TRAINING Qs AND As


# can be Train or Infer
EPOCHS = 100
TRAINING_SIZE = 29282
BATCH_SIZE = 32

#STEPS = 300

STEPS = (TRAINING_SIZE//BATCH_SIZE)*EPOCHS

EMBEDDING_SIZE = 512
ENCODER_NUM_UNITS = EMBEDDING_SIZE
ENCODER_NUM_LAYERS = 4

ATTENTION = False
ATTENTION_SIZE = 1024

DECODER_NUM_UNITS = EMBEDDING_SIZE
DECODER_NUM_LAYERS = 4

DROPOUT = 0.2

MAX_ITER = 50


max_gradient_norm = 1
starting_learning_rate = 1e0
decay_steps = 16000
decay_rate = 0.5

def build_vocabularies():
	vocabulary = du.load_vocabulary()
	rev_vocabulary = du.build_reverse_vocabulary(vocabulary)
	return (vocabulary, rev_vocabulary)

def get_eos_sos_ids(vocabulary):
	eos_id = vocabulary[du.STOP_SYMBOL]
	sos_id = vocabulary[du.START_SYMBOL]
	return (sos_id, eos_id)

def build_training_iterators(eos_id):
	# Create the lookup table (this is the same as building the vocabulary) 
	lookup_table = \
		tf.contrib.lookup.index_table_from_file('./data/plain_vocabulary.txt')

	# Load questions and anwers datasets
	train_questions = tf.data.TextLineDataset('./data/questions.txt')
	train_answers = tf.data.TextLineDataset('./data/answers.txt')

	# split each string in the qestuions
	train_questions = train_questions.map(
		lambda string: tf.string_split([string]).values)

	# for each question, create a tuple (question_word_list, length)
	train_questions = train_questions.map(
		lambda words: (words, tf.size(words)))

	# for each tuple, substitute the first element with its numerical 
	# representation
	train_questions = train_questions.map(
		lambda words, size: (lookup_table.lookup(words), size))

	# Same as above for the answers
	train_answers = train_answers.map(
		lambda string: tf.string_split([string]).values)
	train_answers = train_answers.map(
		lambda words: (words, tf.size(words)))
	train_answers = train_answers.map(
		lambda words, size: (lookup_table.lookup(words), size))

	# Merge the two datasets into one that returns a tuple of tuples,
	# namely: 
	# ((question, question_length), (answer, answer_length)) 
	q_a_dataset = tf.data.Dataset.zip((train_questions, train_answers))

	# Batch the data and pad with EOS according to the 
	# max length of the batch
	batched_dataset = q_a_dataset.padded_batch(
		BATCH_SIZE,
		padded_shapes = 
			((tf.TensorShape([None]), tf.TensorShape([])),
			(tf.TensorShape([None]), tf.TensorShape([]))),
		padding_values = 
			((tf.cast(eos_id, dtype = tf. int64), 0),
			(tf.cast(eos_id, dtype = tf.int64), 0)))

	# Create the iterator to extract data from the dataset
	batched_iterator = batched_dataset.make_initializable_iterator()

	return batched_iterator

def build_embedding_encoder():
#	source = tf.transpose(source)
	embedding_encoder = tf.get_variable('embedding_encoder',
		[du.FULL_VSIZE, EMBEDDING_SIZE])
	return embedding_encoder

def build_embedding_decoder():
#	target = tf.transpose(target)
	embedding_decoder = tf.get_variable('embedding_decoder',
		[du.FULL_VSIZE, EMBEDDING_SIZE])
	return embedding_decoder

def embed_input(sequence, embedding_callable):
	sequence = tf.transpose(sequence)
	embedded_sequence = tf.nn.embedding_lookup(
		embedding_callable,
		sequence)

	return embedded_sequence


def build_encoder(encoder_embedded_input, source_lengths, keep_prob):
	# Build the encoder RNN
	e_single_cell = tf.contrib.rnn.GRUCell(
		ENCODER_NUM_UNITS)
	e_single_cell = tf.contrib.rnn.DropoutWrapper(
		e_single_cell,
		input_keep_prob = keep_prob)
	encoder_cell = tf.contrib.rnn.MultiRNNCell(
		[e_single_cell for _ in range(ENCODER_NUM_LAYERS)])

	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
		encoder_cell,
		encoder_embedded_input,
		sequence_length = source_lengths,
		time_major = True,
		dtype = tf.float32)
	return (encoder_outputs, encoder_state)

def build_projection_layer():
	projection_layer =  layers_core.Dense(
		du.FULL_VSIZE,
		use_bias = False)
	return projection_layer



def build_decoder_cell(keep_prob):
	de_single_cell = tf.contrib.rnn.GRUCell(
		DECODER_NUM_UNITS)

	de_single_cell = tf.contrib.rnn.DropoutWrapper(
		de_single_cell,
		input_keep_prob = keep_prob)

	decoder_cell = tf.contrib.rnn.MultiRNNCell(
		[de_single_cell for _ in range(DECODER_NUM_LAYERS)])
	return decoder_cell

def build_training_decoder(decoder_embedded_input, 
	target_lengths, encoder_state, projection_layer, decoder_cell):

	helper = tf.contrib.seq2seq.TrainingHelper(
	    decoder_embedded_input, target_lengths, time_major=True)

	decoder = tf.contrib.seq2seq.BasicDecoder(
	    decoder_cell, 
	    helper, 
	    encoder_state,
	    output_layer=projection_layer)

	# Dynamic decoding
	decoder_outputs, _, _= tf.contrib.seq2seq.dynamic_decode(decoder, 
		output_time_major = True)
	return decoder_outputs

def build_infer_decoder(embedding_decoder, 
	encoder_state, 
	projection_layer, 
	decoder_cell, 
	sos_id, eos_id):

	helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
		embedding_decoder,
		sos_id, eos_id) 
	decoder = tf.contrib.seq2seq.BasicDecoder(
	    decoder_cell, 
	    helper, 
	    encoder_state,
	    output_layer=projection_layer)

	outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    	decoder, maximum_iterations=MAX_ITER)
	sequence = outputs.sample_id
	return sequence

def build_training_model(batched_iterator,
	encoder_embedded_input,
	encoder_outputs, encoder_state,
	decoder_embedded_input,
	projection_layer,
	decoder_cell):


	decoder_outputs = build_training_decoder(decoder_embedded_input, 
		target_lengths, encoder_state, projection_layer, decoder_cell)
	
	logits = decoder_outputs.rnn_output
	return (target, target_lengths, logits)

def build_inference_model():
	pass


def build_training_loss(target, target_lengths, logits, global_step):
	target = tf.transpose(target)
	learning_rate = tf.train.exponential_decay(starting_learning_rate, 
		global_step = global_step,
		decay_steps = decay_steps, 
		decay_rate = decay_rate, 
		staircase=True,
		name = 'exp_lr_decay')



	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels = target, logits = logits)

	# target_weights is a zero-one matrix of the same size as decoder_outputs. 
	# It masks padding positions outside of the target sequence lengths with values 0.

	target_weights = tf.sequence_mask(
	        target_lengths, target.shape[0].value, dtype = tf.float32)
	# because it is time major
	target_weights = tf.transpose(target_weights)

	train_loss = tf.reduce_sum((crossent*target_weights)/BATCH_SIZE)

	params = tf.trainable_variables()
	gradients = tf.gradients(train_loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(
		gradients, max_gradient_norm)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	update_step = optimizer.apply_gradients(
		zip(clipped_gradients, params),
		global_step = global_step)
	return (train_loss, update_step, learning_rate)


if __name__ == '__main__':

	lines = open('./data/questions.txt').read().splitlines()

	train_graph = tf.Graph()
	test_graph = tf.Graph()

	tain_sess = tf.Session(graph = train_graph)
	infer_sess = tf.Session(graph = test_graph)
	
	vocabulary, rev_vocabulary = build_vocabularies()
	eos_id, sos_id = get_eos_sos_ids(vocabulary)	

	with train_graph.as_default():
		keep_prob = tf.placeholder_with_default(0.8, shape=())		
		batched_iterator = build_training_iterators(eos_id)


		embedding_encoder = build_embedding_encoder()
		((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()
		encoder_embedded_input = embed_input(source, embedding_encoder)

		encoder_outputs, encoder_state = build_encoder(encoder_embedded_input, 
			source_lengths, keep_prob)

		embedding_decoder = build_embedding_decoder()
		projection_layer = build_projection_layer()
		decoder_cell = build_decoder_cell(keep_prob)
		decoder_embedded_input = embed_input(target, embedding_decoder)
		target, target_lengths, logits = build_training_model(batched_iterator,
											encoder_embedded_input,
											encoder_outputs, encoder_state,
											decoder_embedded_input,
											projection_layer,
											decoder_cell)

		global_step = tf.Variable(0, trainable=False)

		train_loss, update_step, learning_rate = \
			build_training_loss(target,
								target_lengths, 
								logits,
								global_step)
		init_table = tf.tables_initializer()
		init = tf.global_variables_initializer()
		train_saver = tf.train.Saver()

	with test_graph.as_default():
		source = tf.placeholder(dtype = tf.int64, shape = [1, None])
		source_lengths = tf.placeholder(dtype = tf.int32, shape = 1)
		keep_prob = tf.placeholder_with_default(0.8, shape=())		

		embedding_encoder = build_embedding_encoder()
		encoder_embedded_input = embed_input(source, embedding_encoder)

		encoder_outputs, encoder_state = build_encoder(encoder_embedded_input, 
			source_lengths, keep_prob)

		embedding_decoder = build_embedding_decoder()
		projection_layer = build_projection_layer()
		decoder_cell = build_decoder_cell(keep_prob)
		sequence = build_infer_decoder(embedding_decoder, 
			encoder_state, 
			projection_layer, 
			decoder_cell, 
			[vocabulary[du.START_SYMBOL]], 
			vocabulary[du.STOP_SYMBOL])
		test_saver = tf.train.Saver()
		


	# HERE STARTS THE TRAINING 
	train_sess = tf.Session(graph=train_graph)
	test_sess = tf.Session(graph=test_graph)

	train_sess.run(init)
	train_sess.run(init_table)
	train_sess.run(batched_iterator.initializer)
	epoch = 1
	for step in range(STEPS):
		if step%50 == 0 and step > 0:
			train_saver.save(train_sess, 
				'./checkpoints/basic_model.ckpt',
				write_meta_graph=False,
    			write_state=False,
    			global_step = step+1)
			saved_ckpt_fname = './checkpoints/basic_model.ckpt-%d' % (step+1) 
			test_saver.restore(test_sess,
				saved_ckpt_fname)
			#line = random.choice(lines)
			line = 'figata che germania la visto hai va come merda ciao'
			line = line.split()
			line = [vocabulary[x] 
				if x in vocabulary.keys() else vocabulary[du.UNK_SYMBOL] 
				for x in line]
			line = np.array(line)
			line_len = np.array([line.shape[0]])
			line = line.reshape((1,len(line)))
			feed_dict = {source: line, source_lengths: line_len, keep_prob: 1.0}
			s = test_sess.run(sequence, feed_dict = feed_dict)
			s = s[0]
			line = [x for x in line[0]]
			line[:-1] = list(reversed(line[:-1]))
			print('epoch: ', epoch, 'step: ', step, 'loss: ', l)
			print('User:-->', ' '.join(rev_vocabulary[x] for x in line))
			print('BOT:-->', ' '.join(rev_vocabulary[x] for x in s))

		if step == (TRAINING_SIZE//BATCH_SIZE)*epoch:
			print('DONE EPOCH %d' % epoch)
			train_sess.run(batched_iterator.initializer)
			epoch+=1

		if tf.train.global_step(train_sess, global_step) % decay_steps == 0:
			print('learning_rate: ', train_sess.run(learning_rate))
		l, _ = train_sess.run([train_loss,update_step])
			
	train_saver.save(train_sess, 
				'./checkpoints/basic_model.ckpt')	

	
