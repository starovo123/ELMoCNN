import os
import numpy as np
from gensim import corpora, models, similarities
import time
import scipy.io as scio

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_stopword():
    '''
    加载停用词表
    :return: 返回停用词的列表
    '''
    f_stop = open('E:/Data/stopwords/en_stopwords.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


def create_corpusList():
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
                    f = open(fpath, 'r', encoding='latin-1')
                    texts.append(f.read().strip())
                    f.close()
                    labels.append(label_id)
    return texts
    # f = open('E:/Code/ELMo_CNN/LdaCorpus.txt', 'w', encoding='utf-8')
    # for t in texts:
    #     f.write(t+'\n')
    # f.close()

def getLdaMatrix():

    print('1.初始化停止词列表 ------')
    # 开始的时间
    t_start = time.time()
    # 加载停用词表
    stop_words = load_stopword()

    print('2.开始读入语料数据 ------ ')
    # 读入语料库
    # create_corpusTXT()
    # f = open('E:/Code/ELMo_CNN/LdaCorpus.txt','r')
    # 语料库分词并去停用词
    texts = create_corpusList()
    texts = [[word for word in OneText.strip().lower().split() if word not in stop_words] for OneText in texts]
    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))
    # f.close()
    M = len(texts)
    print('文本数目：%d个' % M)

    print('3.正在建立词典 ------')
    # 建立字典
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)

    print('4.正在计算文本向量 ------')
    # 转换文本数据为索引，并计数
    corpus = [dictionary.doc2bow(text) for text in texts]

    print('5.正在计算文档TF-IDF ------')
    t_start = time.time()
    # 计算tf-idf值
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))

    print('6.LDA模型拟合推断 ------')
    # 训练模型
    num_topics = 20
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                      alpha=0.01, eta=0.01, minimum_probability=0.001,
                      update_every=1, chunksize=100, passes=1)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))

    # matrix(1772, 20) 打印1772个文档的20个主题
    topic_text_matrix = []
    print('7.结果：1772个文档的主题分布：--')
    doc_topics = lda.get_document_topics(corpus_tfidf)
    idx = np.arange(M)
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[0:20,1])
        topic_idx = topic_distribute.argsort()[:-num_topics-1:-1]
        topic_text_matrix.append(topic_distribute[topic_idx])
    return topic_text_matrix

if __name__ == '__main__':
    # topic_matrix= getLdaMatrix()
    # np.save('topic_matrix.npy',topic_matrix)
    m = np.load('topic_matrix.npy',allow_pickle=True)
    # print(m)
    # path = 'E:/Code/ELMo_CNN/lda.mat'
    # scio.savemat(path, {'text':topic_matrix})
    # print(topic_matrix[:5])


    # 随机打印某10个文档的主题
    # num_show_topic = 10  # 每个文档显示前几个主题
    # print('7.结果：10个文档的主题分布：--')
    # doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    # idx = np.arange(M)
    # np.random.shuffle(idx)
    # idx = idx[:10]
    # for i in idx:
    #     topic = np.array(doc_topics[i])
    #     topic_distribute = np.array(topic[:, 1])
    #     # print topic_distribute
    #     topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
    #     print('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx
    #     print(topic_distribute[topic_idx])
    # #
    # num_show_term = 10  # 每个主题显示几个词
    # print('8.结果：每个主题的词分布：--')
    # for topic_id in range(num_topics):
    #     print('主题#%d：\t' % topic_id)
    #     term_distribute_all = lda.get_topic_terms(topicid=topic_id)
    #     term_distribute = term_distribute_all[:num_show_term]
    #     term_distribute = np.array(term_distribute)
    #     term_id = term_distribute[:, 0].astype(np.int)
    #     print('词：\t', )
    #     for t in term_id:
    #         print(dictionary.id2token[t], )
    #     print('\n概率：\t', term_distribute[:, 1])

