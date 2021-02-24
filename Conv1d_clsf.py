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
tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)

# matrixs = tokenizer.texts_to_matrix(texts2d)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

# data = pad_sequences(sequences)
# from keras.utils import np_utils
# labels = np_utils.to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)

# # split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels_new = []
# for i in indices:
#     labels_new.append(labels[i])
#
# nb_validation_samples = int(0.8 * data.shape[0])

trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

x_train, x_val, y_train, y_val= model_selection.train_test_split(trainDF['text'], trainDF['label'])
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.fit_transform(y_val)

# x_train = data[:-nb_validation_samples]
# y_train = np.array(labels_new[:-nb_validation_samples])
# x_val = data[-nb_validation_samples:]
# y_val = np.array(labels_new[-nb_validation_samples:])
# print(x_train[0])

# tf-idf 特征
# tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=9927)
# tfidf_vect.fit(texts)
# xtrain_tfidf = tfidf_vect.transform(x_train.tolist())
# xvalid_tfidf = tfidf_vect.transform(x_val.tolist())

x_train = np.array(x_train)
x_val = np.array(x_val)

tokenizer.fit_on_texts(trainDF['text'])
word_index = tokenizer.word_index
x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=148)
x_val = sequence.pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=148)


###############读取词向量

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

embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=10036, trainable=False)
# embedding_layer = Embedding(10000, 24, input_length=10036)
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
sequence_input = Input(shape=(148,), dtype='int32') #shape=10036/9927
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(24)(x)  # global max pooling
x = Flatten()(x)
fea_x = Dense(128, activation='relu', name='fea_x')(x)
preds = Dense(1, activation='sigmoid')(fea_x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=4, batch_size=128)
model.save('E:/Code/ELMo_CNN/ELMo_CNN.h5')

# layer_name = 'fea_x'
# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('fea_x').output)
# intermediate_output = intermediate_layer_model.predict(x_val)
# for i in intermediate_output:
#     print(i)