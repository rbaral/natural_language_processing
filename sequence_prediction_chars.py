from __future__ import absolute_import, division, print_function, unicode_literals

'''
character embedding for next character prediction

Ref: https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
'''

from numpy import array
import os
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


data_dir = "/Users/dur-rbaral-m/projects/test_projects/data/"

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def prepare_data_lstm():
	raw_text = load_doc(os.path.join(data_dir, 'rhyme.txt'))
	print(raw_text)

	# clean
	tokens = raw_text.split()
	raw_text = ' '.join(tokens)

	# organize into sequences of characters
	length = 10
	sequences = list()
	for i in range(length, len(raw_text)):
		# select sequence of tokens
		seq = raw_text[i - length:i + 1]
		# store
		sequences.append(seq)
	print('Total Sequences: %d' % len(sequences))

	# save sequences to file
	out_filename = os.path.join(data_dir, 'char_sequences.txt')
	save_doc(sequences, out_filename)

	# load
	in_filename = os.path.join(data_dir, 'char_sequences.txt')
	raw_text = load_doc(in_filename)
	lines = raw_text.split('\n')

	# integer encode sequences of characters
	chars = sorted(list(set(raw_text)))
	mapping = dict((c, i) for i, c in enumerate(chars))
	sequences = list()
	for line in lines:
		# integer encode line
		encoded_seq = [mapping[char] for char in line]
		# store
		sequences.append(encoded_seq)

	# vocabulary size
	vocab_size = len(mapping)
	print('Vocabulary Size: %d' % vocab_size)

	# separate into input and output
	sequences = array(sequences)
	X, y = sequences[:, :-1], sequences[:, -1]
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
	X = array(sequences)
	y = to_categorical(y, num_classes=vocab_size)
	return X, y, vocab_size, mapping



# define model
def create_model_lstm():
	X, y, vocab_size, mapping = prepare_data_lstm()
	model = Sequential()
	model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(X, y, epochs=100, verbose=2)

	# save the model to file
	model.save(os.path.join(data_dir, 'model.h5'))
	# save the mapping
	dump(mapping, open(os.path.join(data_dir,'mapping.pkl'), 'wb'))



# generate a sequence of characters with a language model
def generate_seq_lstm(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		#encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

def test_model_lstm():
	# load the model
	model = load_model(os.path.join(data_dir, 'model.h5'))
	# load the mapping
	mapping = load(open(os.path.join(data_dir, 'mapping.pkl'), 'rb'))

	# test start of rhyme
	print(generate_seq_lstm(model, mapping, 10, 'Sing a son', 20))
	# test mid-line
	print(generate_seq_lstm(model, mapping, 10, 'king was i', 20))
	# test not in original
	print(generate_seq_lstm(model, mapping, 10, 'hello worl', 20))
	print(generate_seq_lstm(model, mapping, 10, 'When the p', 20))


#Using core RNN and tensorflow

#imports

import tensorflow as tf
import numpy as np
import os
import time
tf.enable_eager_execution()

def prepare_data_rnn():
	path_to_file = tf.keras.utils.get_file('shakespeare.txt',
										   'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
	# Read, then decode for py2 compat.
	text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
	# length of text is the number of characters in it
	print('Length of text: {} characters'.format(len(text)))
	# Take a look at the first 250 characters in text
	print(text[:250])
	# The unique characters in the file
	vocab = sorted(set(text))
	print('{} unique characters'.format(len(vocab)))
	#map chars to numbers/index for processing
	# Creating a mapping from unique characters to indices
	char2idx = {u: i for i, u in enumerate(vocab)}
	idx2char = np.array(vocab)

	text_as_int = np.array([char2idx[c] for c in text])
	print('{')
	for char, _ in zip(char2idx, range(20)):
		print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
	print('  ...\n}')
	# Show how the first 13 characters from the text are mapped to integers
	print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

	# The maximum length sentence we want for a single input in characters
	seq_length = 100
	examples_per_epoch = len(text) // seq_length

	# Create training examples / targets
	char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
	if tf.executing_eagerly():
		for i in char_dataset.take(5):
			print(idx2char[i.numpy()])

	sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

	for item in sequences.take(5):
		print(repr(''.join(idx2char[item.numpy()])))

	def split_input_target(chunk):
		input_text = chunk[:-1]
		target_text = chunk[1:]
		return input_text, target_text

	dataset = sequences.map(split_input_target)

	for input_example, target_example in dataset.take(1):
		print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
		print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

	for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
		print("Step {:4d}".format(i))
		print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
		print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

	return dataset, vocab, idx2char, char2idx


def train_model_rnn(dataset, vocab, idx2char, char2idx):

	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

	model = build_model_rnn(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE)

	EPOCHS = 30
	#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
	optimizer = tf.keras.optimizers.Adam()

	@tf.function
	def train_step(inp, target):
		with tf.GradientTape() as tape:
			predictions = model(inp)
			loss = tf.reduce_mean(
				tf.keras.losses.sparse_categorical_crossentropy(
					target, predictions, from_logits=True))
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		return loss

	for epoch in range(EPOCHS):
		start = time.time()

		# initializing the hidden state at the start of every epoch
		# initally hidden is None
		hidden = model.reset_states()

		for (batch_n, (inp, target)) in enumerate(dataset):
			loss = train_step(inp, target)

			if batch_n % 100 == 0:
				template = 'Epoch {} Batch {} Loss {}'
				print(template.format(epoch + 1, batch_n, loss))

		# saving (checkpoint) the model every 5 epochs
		if (epoch + 1) % 5 == 0:
			model.save_weights(checkpoint_prefix.format(epoch=epoch))

		print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
		print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	model.save_weights(checkpoint_prefix.format(epoch=epoch))
	return model


def build_model_rnn(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim,
								  batch_input_shape=[batch_size, None]),
		tf.keras.layers.LSTM(rnn_units,
							return_sequences=True,
							stateful=True,
							recurrent_initializer='glorot_uniform'),
		tf.keras.layers.Dense(vocab_size)
	  ])
	return model


def generate_seq_rnn(model, start_string):
	# Evaluation step (generating text using the learned model)

	# Number of characters to generate
	num_generate = 1000

	# Converting our start string to numbers (vectorizing)
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)

	# Empty string to store our results
	text_generated = []

	# Low temperatures results in more predictable text.
	# Higher temperatures results in more surprising text.
	# Experiment to find the best setting.
	temperature = 1.0

	# Here batch size == 1
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		# remove the batch dimension
		predictions = tf.squeeze(predictions, 0)

		# using a categorical distribution to predict the word returned by the model
		predictions = predictions / temperature
		predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

		# We pass the predicted word as the next input to the model
		# along with the previous hidden state
		input_eval = tf.expand_dims([predicted_id], 0)

		text_generated.append(idx2char[predicted_id])

	return (start_string + ''.join(text_generated))


def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# Directory where the checkpoints will be saved
checkpoint_dir = os.path.join(data_dir, 'seq_prediction_training_checkpoints')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
dataset, vocab, idx2char, char2idx = prepare_data_rnn()
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def test_model_rnn():
	model = build_model_rnn(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE)

	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

	model.build(tf.TensorShape([1, None]))
	print(model.summary())
	print(generate_seq_rnn(model, start_string=u"ROMEO: "))


if __name__=="__main__":
	print("main method started")
	#create_model_lstm()
	#test_model_lstm()
	train_model_rnn(dataset, vocab, idx2char, char2idx)
	test_model_rnn()
