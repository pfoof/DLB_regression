import os
import sys
import numpy as np
import string
from nltk.corpus import stopwords
from keras.utils.vis_utils import plot_model

if len(sys.argv) < 4:
	print("Usage test_regr.py gru100|doublegru|gru32 .ckpt review")
	quit()
test_review = sys.argv[3]

def get_model(version="gru32", num_words=100, emb_matrix=None, max_length=213):
	m_clipvalue = 0.5
	m_clipnorm = 1.0
	model = Sequential()
	embedding_layer = Embedding(num_words, 100, embeddings_initializer = Constant(emb_matrix), input_length = max_length, trainable=False)
	model.add(embedding_layer)
	if version == "doublegru":
		gru_units=32
		model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
		model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2))
	elif version == "gru100":
		gru_units=100
		model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2))
	else:
		gru_units=32
		model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2))

	model.add(Dense(1, activation = 'relu')) #for regression
	lr = 0.0005
	from keras.optimizers import Adam
	adam = Adam(learning_rate=lr, clipvalue=m_clipvalue, clipnorm=m_clipnorm)
	model.compile(loss = 'mean_absolute_error', optimizer=adam, metrics=['accuracy'])
	return model

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
print('Cleaning test review')
test_review = clean_review(test_review)

max_length = max([len(s.split()) for s in input_x])

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer()
input_x = np.array(input_x)
tokenizer.fit_on_texts(input_x)
sequences = tokenizer.texts_to_sequences(input_x)
test_sequences = tokenizer.texts_to_sequences([test_review])
word_index = tokenizer.word_index

review_pad = pad_sequences(sequences, maxlen=max_length)
test_pad = pad_sequences(test_sequences, maxlen=max_length)
ratings = np.array(input_y)

inv_index = {}
for w, i in word_index.items():
	inv_index[i]=w

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


model = get_model(sys.argv[1], num_words, emb_matrix, max_length)

#print(model.to_yaml())
#plot_model(model, to_file="%s_model.png" % sys.argv[1], show_shapes=True, show_layer_names=True)

model.summary()

#quit()

from datetime import datetime

if len(sys.argv)>=2:
	model.load_weights(sys.argv[2])
else:
	print("Please specify the checkpoint!")
	quit()

h = model.predict(test_pad)
stars = h[0] * 4.0 + 1.0
from math import floor, ceil
print( "%.3f = %d or %d stars" % (h[0], floor(stars), ceil(stars)) )

