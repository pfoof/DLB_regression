import os
import sys
import numpy as np
import string
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))
def clean_review(text):
	text = text.lower().translate(str.maketrans('','',string.punctuation))
	text = " ".join( [w for w in text.split(" ") if (w.isalpha() and not w in sw) ] )
	return text

print('Loading input.csv')
input_rows = open("input.csv", "r").readlines()
print('Loading communication.csv')
test_rows = open("communication.csv","r").readlines()
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

test_x = []
test_y = []

print('Removing test ratings < 3 words')
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
from collections import Counter

tokenizer = Tokenizer()
input_x = np.array(input_x)
test_x = np.array(test_x)
tokenizer.fit_on_texts(input_x)

Xtrain = pad_sequences(tokenizer.texts_to_sequences(input_x), maxlen = max_length)
Xtest = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=max_length)
ratings = np.array(input_y)
test_ratings = np.array(test_y)

print("Reviews shape: %s, ratings shape: %s" % (Xtest.shape, test_ratings.shape))
#tokenizer.fit_on_texts()

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

m_clipnorm=1.0
m_clipvalue=0.55

model = Sequential()
model.add(Dense(1, activation = 'relu', input_shape=(213,))) #for regression
#model.add(Dense(6, activation = 'softmax')) #for categorical
lr = float(sys.argv[2]) if len(sys.argv)>=3 else 0.0005
from keras.optimizers import Adam
adam = Adam(learning_rate=lr, clipvalue=m_clipvalue, clipnorm=m_clipnorm)
model.compile(loss = 'mean_absolute_error', optimizer=adam, metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

VAL_SPLIT = 0.2
indices = np.arange(Xtrain.shape[0])
np.random.shuffle(indices)
Xtrain = Xtrain[indices]
ratings = ratings[indices]
num_val_samp = int(VAL_SPLIT * Xtrain.shape[0])

X_train = Xtrain[:-num_val_samp]
Y_train = ratings[:-num_val_samp]
X_test = Xtrain[-num_val_samp:]
Y_test = ratings[-num_val_samp:]

ep = int(sys.argv[3]) if len(sys.argv)>=4 else 20
from datetime import datetime

project_name = "dense_l%f_e%dt%s" % (lr, ep, datetime.now().strftime("%m%d_%Hh%M%S"))
print("Project name: %s" % project_name)

import keras.callbacks
log_path = "logs/%s" % project_name
cp_path = "ckpt/%s.ckpt" % project_name
cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True,verbose=1, period = 10)
tb_callback = keras.callbacks.TensorBoard(log_dir = log_path, histogram_freq = 5)
if len(sys.argv)>=2:
	if not sys.argv[1] == '-':
		model.load_weights(sys.argv[1])
#h = model.fit(X_train, Y_train, batch_size = 256, epochs = ep, validation_data = (X_test, Y_test), callbacks=[tb_callback, cp_callback], verbose=2)
ev = model.evaluate(Xtest, test_ratings)
print(model.metrics_names)
print(ev)
quit()

model.summary()

import json
with open('history.json', 'w') as f:
	json.dump(h.history, f)
