import os
import sys
import numpy as np
import string
from nltk.corpus import stopwords

print("Loading embeddings...")
embeddings_index = {}
f = open('/tmp/glove.6B.100d.txt', encoding = 'utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

#print('%d vectors' % len(embeddings_index))

sw = set(stopwords.words('english'))
def clean_review(text):
	text = text.lower().translate(str.maketrans('','',string.punctuation))
	text = " ".join( [w for w in text.split(" ") if (w.isalpha() and not w in sw) ] )
	return text

print('Loading input.csv')
input_rows = open("input.csv", "r").readlines()
input_x = []
input_y = []
import re
print('Removing ratings < 3 words')
for row in input_rows:
	if(len(row.strip().rstrip()) < 4):
		continue
	text = clean_review(row[:-2])
	r = re.search('^.*,(\d)$' ,row)
	if r is None:
		continue
	rating = r.group(1)
	wc = len(text.split(" "))
	if wc > 2:
		input_x.append(text)
		input_y.append( (float(rating) - 1.0)/4.0 )
print("%d texts %d ratings" % ( len(input_x), len(input_y) ) )
max_length = max([len(s.split()) for s in input_x])
print("max_length = %d" % max_length)
#print(input_x.shape)

print("Loading communication.csv")
test_rows = open("communication.csv", "r").readlines()
test_x = []
test_y = []
import re
print('Removing ratings < 3 words')
for row in test_rows:
        if(len(row.strip().rstrip()) < 4):
                continue
        text = clean_review(row[:-2])
        r = re.search('^.*,(\d)$' ,row)
        if r is None:
                continue
        rating = r.group(1)
        wc = len(text.split(" "))
        if wc > 2:
                test_x.append(text)
                test_y.append( (float(rating) - 1.0)/4.0 )
print("%d texts %d ratings" % ( len(test_x), len(test_y) ) )

#quit()

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer()
input_x = np.array(input_x)
tokenizer.fit_on_texts(input_x)
sequences = tokenizer.texts_to_sequences(input_x)
word_index = tokenizer.word_index

review_pad = pad_sequences(sequences, maxlen=max_length)
test_pad = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=max_length)

#ratings = np.array(input_y)/5.0
ratings = np.array(input_y)
test_ratings = np.array(test_y)

inv_index = {}
for w, i in word_index.items():
	inv_index[i]=w
#ratings = to_categorical(ratings)

print("%d tokens, reviews shape: %s, ratings shape: %s" % (len(word_index), review_pad.shape, ratings.shape))
#tokenizer.fit_on_texts()

num_words = len(word_index) + 1
emb_matrix = np.zeros((num_words, 100))

for word, i in word_index.items():
	if i > num_words:
		continue
	emb_vec = embeddings_index.get(word)
	if emb_vec is not None:
		emb_matrix[i] = emb_vec

print(emb_matrix.shape)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

gru_units=100
m_clipnorm=1.0
m_clipvalue=0.55

model = Sequential()
embedding_layer = Embedding(num_words, 100, embeddings_initializer = Constant(emb_matrix), input_length = max_length, trainable=False)
model.add(embedding_layer)
model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = 'relu')) #for regression
#model.add(Dense(6, activation = 'softmax')) #for categorical
lr = float(sys.argv[2]) if len(sys.argv)>=3 else 0.0005
from keras.optimizers import Adam
adam = Adam(learning_rate=lr, clipvalue=m_clipvalue, clipnorm=m_clipnorm)
model.compile(loss = 'mean_absolute_error', optimizer=adam, metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

from datetime import datetime

import keras.callbacks
if len(sys.argv)>=2:
	model.load_weights(sys.argv[1])
else:
	print("Please specify checkpoint")
	quit()

h = model.evaluate(x = test_pad, y = test_ratings, batch_size=256, verbose=2)
print(model.metrics_names)
print(h)
TEST_PREDICT = 20
tt = model.predict(x = test_pad[:TEST_PREDICT], batch_size=16, verbose=2)
for i in range(TEST_PREDICT):
        print("#%d: tested = %.1f -- actual = %.1f -- error: %.2f" % (i, tt[i],test_ratings[i], test_ratings[i]-tt[i]))
