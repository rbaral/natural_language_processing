'''
REf: https://www.kaggle.com/carlosaguayo/deep-learning-for-text-classification/data
'''

import os
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.base import get_data_home
from keras.metrics import categorical_accuracy

data_home = get_data_home()
twenty_home = os.path.join(data_home, "20news_home")

if not os.path.exists(data_home):
    os.makedirs(data_home)

if not os.path.exists(twenty_home):
    os.makedirs(twenty_home)

data_dir = "/Users/dur-rbaral-m/projects/project_68/datagen/current/rovi_all_unbound_fetch_data"

#!cp.. /input/20-newsgroup-sklearn/20news-bydate_py3* /tmp/scikit_learn_data



# http://qwone.com/~jason/20Newsgroups/
dataset = fetch_20newsgroups(subset='all', shuffle=True, download_if_missing=True)

texts = dataset.data # Extract text
target = dataset.target # Extract target

vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences

print (tokenizer.texts_to_sequences(['Hello King, how are you?']))

#print (len(sequences))
#print (len(sequences[0]))
#print (sequences[0])

word_index = tokenizer.word_index
#print('Found {:,} unique words.'.format(len(word_index)))

# Create inverse index mapping numbers to words
inv_index = {v: k for k, v in tokenizer.word_index.items()}

# Print out text again
#for w in sequences[0]:
#    x = inv_index.get(w)
#    print(x,end = ' ')

# Get the average length of a text
avg = sum(map(len, sequences)) / len(sequences)

# Get the standard deviation of the sequence length
std = np.sqrt(sum(map(lambda x: (len(x) - avg)**2, sequences)) / len(sequences))

#print(avg,std)

print(pad_sequences([[1,2,3]], maxlen=5))
print(pad_sequences([[1,2,3,4,5,6]], maxlen=5))

max_length = 100
data = pad_sequences(sequences, maxlen=max_length)

'''
Turning labels into One-Hot encodings
Labels can quickly be encoded into one-hot vectors with Keras:

'''
from keras.utils import to_categorical
labels = to_categorical(np.asarray(target))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

print (target[0])
print (labels[0])

embeddings_index = {} # We create a dictionary of word -> embedding

with open(os.path.join(data_dir, 'glove', 'glove.840B.300d.txt')) as f: #'glove.6B.100d.txt'
    for line in f:
        values = line.split()
        word = values[0] # The first value is the word, the rest are the values of the embedding
        #print(word)
        #print(values[1:])
        embedding = np.asarray(values[-300:], dtype='float32') # Load embedding
        embeddings_index[word] = embedding # Add embedding to our embedding dictionary

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

#print (embeddings_index['frog'])
#print (len(embeddings_index['frog']))

print (np.linalg.norm(embeddings_index['man'] - embeddings_index['woman']))
print (np.linalg.norm(embeddings_index['man'] - embeddings_index['cat']))

embedding_dim = 300 # We use 300 dimensional glove vectors

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # How many words are there actually

embedding_matrix = np.zeros((nb_words, embedding_dim))

# The vectors need to be in the same position as their index.
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= vocab_size:
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dim,
                    input_length=max_length,
                    weights = [embedding_matrix],
                    trainable = False))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

model.fit(data, labels, validation_split=0.2, epochs=10)

example = data[400] # get the tokens
print (texts[400])

# Print tokens as text
print("tokens are:\n")
for w in example:
    x = inv_index.get(w)
    print(x,end = ' ')

# Get prediction
pred = model.predict(example.reshape(1,100))

# Output predicted category
print("predicted result is:")
print(dataset.target_names[np.argmax(pred)])