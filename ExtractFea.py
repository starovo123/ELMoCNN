from sklearn import preprocessing
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras.models import *
from keras.layers import *

def WordEmbeddingFea():
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
    tokenizer = Tokenizer()
    trainDF = pd.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    x_train = np.array(trainDF['text'])
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(trainDF['label'])
    tokenizer.fit_on_texts(trainDF['text'])
    word_index = tokenizer.word_index
    x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=9927)

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

    #########我们将这个词向量矩阵加载到Embedding层中，注意，我们设置trainable=False使得这个编码层不可再训练。
    embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=10036, trainable=False)
    sequence_input = Input(shape=(9927,), dtype='int32') #shape=10036/9927
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    fea_x = Dense(128, activation='relu', name='fea_x')(x)
    preds = Dense(1, activation='sigmoid')(fea_x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_train, y_train),epochs=4, batch_size=128)
    model.save('E:/Code/ELMo_CNN/ELMo_CNN.h5')

    layer_name = 'fea_x'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('fea_x').output)
    intermediate_output = intermediate_layer_model.predict(x_train)
    return intermediate_output

if __name__ == '__main__':
    matrix = WordEmbeddingFea()
    np.save('wordEmbedding.npy', matrix)
    # m = np.load('wordEmbedding.npy', allow_pickle=True)
    # print(m)