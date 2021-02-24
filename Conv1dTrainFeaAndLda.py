from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing
import pandas as pd
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
import os
TEXT_DATA_DIR = 'E:/Data/20_Newsgroups/20newsgroup'
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        if label_id == 2:
            break
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath,'r',encoding='latin-1')
                texts.append(f.read().strip())
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))
print(texts[0])
print(labels[0])


######，我们可以新闻样本转化为神经网络训练所用的张量。
# 所用到的Keras库是keras.preprocessing.text.Tokenizer和keras.preprocessing.sequence.pad_sequences。代码如下所示
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import ExtractFea
import LdaFea
import tensorflow as tf

wordEmbedding_fea = np.load('wordEmbedding.npy', allow_pickle=True)
LDA_fea = np.load('topic_matrix.npy', allow_pickle=True)


texts_matrix = np.concatenate((wordEmbedding_fea, LDA_fea),axis=1)
# texts_matrix = tf.expand_dims(texts_matrix, -1)
y_train = np.array(labels)
# y_train = tf.expand_dims(y_train, -1)
# y_train = tf.expand_dims(y_train, -1)

# trainDF = pd.DataFrame()
# trainDF['text'] = texts_matrix
# trainDF['label'] = labels

# x_train, x_val, y_train, y_val= model_selection.train_test_split(trainDF['text'], trainDF['label'])
# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit_transform(trainDF['label'])
# y_val = encoder.fit_transform(y_val)

# x_train = np.array(x_train)
# x_val = np.array(x_val)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(trainDF['text'])
# x_seq = tokenizer.texts_to_sequences(x_train)
# x_train = sequence.pad_sequences(, maxlen=9927)
# x_val = sequence.pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=9927)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
embeddings_index = {}
f = open(os.path.join('E:\Data\glove.6B', 'glove.6B.100d.txt'),'r',encoding='utf-8')
for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



from keras.models import *
from keras.layers import *

# embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=10036, trainable=False)
embedding_layer = Embedding(10000, 24, input_length=10036)
sequence_input = Input(shape=(148,), dtype='int32') #shape=10036/9927
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu',name='conv1')(embedded_sequences)
x = MaxPooling1D(5, name='pool1')(x)
x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(24)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# happy learning!
model.fit(texts_matrix, y_train, validation_data=(texts_matrix, y_train),epochs=4, batch_size=128)
model.save('E:/Code/ELMo_CNN/ELMo_CNN.h5')

