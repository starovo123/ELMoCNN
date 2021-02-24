from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing
import pandas as pd
import os
import scipy.io as scio

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
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
import LdaFea
# trainMAT = scio.loadmat('E:/Code/ELMo_CNN/lda.mat')
# trainDF = pd.DataFrame()
# trainDF['text'] = texts
# texts_vec = trainMAT['text']
# T_vec = []
# for v in texts_vec:
#     vv = v.split(' ')
#     T_vec.append(vv)
# trainDF['text_topic_vec'] = T_vec

topic_text_matrix = LdaFea.getLdaMatrix()
trainDF = pd.DataFrame()
trainDF['text']=texts
trainDF['label']=labels
trainDF['topic_matrix']=topic_text_matrix

x_train, x_val, y_train, y_val, x_topic_train, x_topic_val = model_selection.train_test_split(trainDF['text'], trainDF['label'],trainDF['topic_matrix'])
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.fit_transform(y_val)
x_train = np.array(x_train)
x_val = np.array(x_val)

x_train_seq = sequence.pad_sequences(x_topic_train, maxlen=70)
x_val_seq = sequence.pad_sequences(x_topic_val, maxlen=70)
# x_train_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=9927)
# x_val_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=9927)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainDF['text'])
word_index = tokenizer.word_index

####读取词向量

embeddings_index = {}
f = open(os.path.join('E:\Data\glove.6B', 'glove.6B.100d.txt'),'r',encoding='utf-8')
for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#############我们可以根据得到的字典生成上文所定义的词向量矩阵
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)
#########我们将这个词向量矩阵加载到Embedding层中，注意，我们设置trainable=False使得这个编码层不可再训练。
from keras.layers import Embedding

# embedding_matrix = np.append(embedding_matrix,x_train_tfidf)

embedding_num = len(word_index)+1
embedding_dim = 100
embedding_layer = Embedding(embedding_num, embedding_dim, weights=[embedding_matrix], input_length=10036, trainable=False)

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
sequence_input = Input(shape=(70,), dtype='int32') #shape=10036/9927
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# happy learning!
model.fit(x_topic, y_train, validation_data=(y_topic, y_val),epochs=4, batch_size=128)
model.save('E:/Code/ELMo_CNN/ELMo_CNN.h5')