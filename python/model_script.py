import csv
import re
import string
from timeit import default_timer as timer
import gensim
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, Input, Dropout, Bidirectional, GaussianNoise
from keras.layers.recurrent import LSTM, GRU

import matplotlib.pyplot as plt
import pandas as pd


#######################################################
'''
Edit these constants to change important things in code below
'''
NOTEEVENTS = "./data/NOTEEVENTS.csv"
DIAGNOSES_ICD = "./data/DIAGNOSES_ICD.csv"
EMBEDDING_MODEL = 'w2v' # 'w2v' or 'ft'
ICD_CODES = ["4019","4280","42731", "41401", "5849"] # ICD codes of interest
SEQ_LEN = 3000 # Max length of note
MAX_WORDS = 50000

SUBSAMPLE_SIZE = 10000
TEST_SET_FRACTION = 0.2
BATCH_SIZE = 128
TRAINING_EPOCHS = 5
MODEL_SAVE_PATH = './models/model_20180501_liam' # Path to save trained model to
#######################################################


def clean_string1(text):
    """
    """
    text = text.strip().lower().replace('-', '_').replace('.', '_').replace(' ', '_').rstrip('_')
    return text


def preprocess_text2(query):
    """
    """
    query = re.sub('\d+|\d+\.\d+', '[NUM]', query)
    query = re.sub('(\[\*\*.*?\*\*\])', '[PHI]', query)
    query = query.strip('"').strip('?').strip("'").strip('(').strip(')').strip(':')
    query = re.sub('['+'!"#$%&\'()*+,-./:;<=>?@\\^`{|}~'+']', '', query)
    word_list = query.split()
    word_list = [clean_string1(word) for word in word_list]
    return word_list


if EMBEDDING_MODEL == 'w2v':
    word_vectors = Word2Vec.load('./data/w2v_embeddings')
else:
    word_vectors = FastText.load('./data/fasttext_embeddings')
embedding_dim = word_vectors.wv.vectors.shape[1]


df1 = pd.read_csv(DIAGNOSES_ICD)
df2 = pd.read_csv(NOTEEVENTS)
# Translate ICD codes to indicator variables
dummy = pd.get_dummies(df1['ICD9_CODE'])[ICD_CODES]
# Append indicator var columnns to original ICD df
dummy_combined = pd.concat([df1, dummy], axis=1)
# Combine by HADM_ID and drop columns
dummy_combined = dummy_combined.groupby(['HADM_ID'], as_index=False).sum().drop(['ROW_ID','SUBJECT_ID','SEQ_NUM'], axis = 1)
#now join the two tables together
df_final = pd.merge(df2, dummy_combined,left_on="HADM_ID", right_on='HADM_ID', how='left')
# Filter by discharge summary
df_final = df_final[df_final['CATEGORY'] == 'Discharge summary']
# removed any hadmid that have more than one entry in database
df_final = df_final.drop_duplicates(subset = "HADM_ID", keep = False)
df_final = df_final.drop(['ROW_ID', 'SUBJECT_ID', 'CHARTDATE',
                          'CHARTTIME', 'STORETIME', 'CATEGORY',
                         'DESCRIPTION', 'CGID', 'ISERROR'], axis = 1)
#random sample of the data
sub_df_final = df_final.sample(SUBSAMPLE_SIZE)

# Read in and process data
input_notes = []
hadm_id = []
y = []
for i in range(len(sub_df_final)):
    y.append(list(map(int, sub_df_final.iloc[i, 2:].tolist())))
    input_notes.append(preprocess_text2(sub_df_final.iloc[i, 1]))
    hadm_id.append(sub_df_final.iloc[i, 0])
y = np.array(y)

'''
Train/test split
'''
x_train, x_test, y_train, y_test = train_test_split(input_notes, y, test_size=TEST_SET_FRACTION)

from keras.preprocessing.text import Tokenizer
'''
Takes words and converts to indices (which then correspond to vectors in
the embedding models)
'''
#max_words = len(word_vectors.wv.vocab)
max_words = MAX_WORDS
token = Tokenizer(max_words)
token.fit_on_texts(input_notes)
vocab_size = max_words + 1

sequences = token.texts_to_sequences(x_train)
test_sequences = token.texts_to_sequences(x_test)

'''
Convert to padded sequences
'''
from keras.preprocessing.sequence import pad_sequences
seq_len = SEQ_LEN
X = pad_sequences(sequences, maxlen=seq_len)
X_test = pad_sequences(test_sequences, maxlen=seq_len)

embeddings_index = {}
vocab = token.word_index.keys()
for word in vocab:
    if word in word_vectors.wv.vocab:
      coefs = np.asarray(word_vectors.wv[word], dtype='float32')
      embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

word_index = token.word_index
embedding_matrix = np.zeros((vocab_size , embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector

num_classes = y_train.shape[1]

import keras.backend as K

# config = K.tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = K.tf.Session(config=config)

from keras.layers import Lambda
ClipLayer = Lambda(lambda x: K.clip(x, min_value=0.01, max_value=0.99))

## Build the model ##
input = Input(shape=(seq_len,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(input)
x = GaussianNoise(0.75)(x)
x = Bidirectional(GRU(units=128, recurrent_dropout=0.2, dropout=0.2, activation = 'relu', return_sequences=True))(x)
x = Bidirectional(GRU(units=128, recurrent_dropout=0.2, dropout=0.2, activation = 'relu'))(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='sigmoid')(x)
x = ClipLayer(x)
model = Model(input, x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y_train, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE)

model.save(MODEL_SAVE_PATH)

model.predict(X_test, y_test, batch_size=256)


