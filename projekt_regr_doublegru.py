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
#ratings = np.array(input_y)/5.0
ratings = np.array(input_y)

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

gru_units=32
m_clipnorm=1.0
m_clipvalue=0.55

model = Sequential()
embedding_layer = Embedding(num_words, 100, embeddings_initializer = Constant(emb_matrix), input_length = max_length, trainable=False)
model.add(embedding_layer)
model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
model.add(GRU(units=gru_units, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = 'relu')) #for regression
#model.add(Dense(6, activation = 'softmax')) #for categorical
lr = float(sys.argv[2]) if len(sys.argv)>=3 else 0.0005
from keras.optimizers import Adam
adam = Adam(learning_rate=lr, clipvalue=m_clipvalue, clipnorm=m_clipnorm)
model.compile(loss = 'mean_absolute_error', optimizer=adam, metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

VAL_SPLIT = 0.2
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
ratings = ratings[indices]
num_val_samp = int(VAL_SPLIT * review_pad.shape[0])

X_train = review_pad[:-num_val_samp]
Y_train = ratings[:-num_val_samp]
X_test = review_pad[-num_val_samp:]
Y_test = ratings[-num_val_samp:]

print("Review: %s " % review_pad[0])
print("Review: %s" % "".join(inv_index[x] if x > 0 and inv_index[x] is not None else "" for x in review_pad[0]))
print("Rating %s\n\n" % ratings[0])

ep = int(sys.argv[3]) if len(sys.argv)>=4 else 20
from datetime import datetime

project_name = "grux2_cn%.2fv%.2fu%dl%f_e%dt%s" % (m_clipnorm, m_clipvalue, gru_units, lr, ep, datetime.now().strftime("%m%d_%Hh%M%S"))
print("Project name: %s" % project_name)

import keras.callbacks
log_path = "logs/%s" % project_name
cp_path = "ckpt/%s.ckpt" % project_name
cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True,verbose=1, period = 10)
tb_callback = keras.callbacks.TensorBoard(log_dir = log_path, histogram_freq = 5)
if len(sys.argv)>=2:
	if not sys.argv[1] == '-':
		model.load_weights(sys.argv[1])
h = model.fit(X_train, Y_train, batch_size = 256, epochs = ep, validation_data = (X_test, Y_test), callbacks=[tb_callback, cp_callback], verbose=2)


import json
with open('history.json', 'w') as f:
	json.dump(h.history, f)
