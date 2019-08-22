
"分数：74.45(2) ——>75.03(3) ——>75.79（现1）——>75.83"
from math import log
import numpy as np
import matplotlib.pyplot as plt
import operator
from gensim import corpora, models
import jieba
import re
from pprint import pprint
import os
from gensim import corpora
import jieba
import xml.dom.minidom
import time


def read_xml(path):
    dom = xml.dom.minidom.parse(path)
    # dom = dom.documentElement
    root = dom.documentElement  # 看不了
    bs = root.getElementsByTagName("Doc")
    st0 = ""
    listm = []
    listl = []
    lm = []
    for bd in bs:
        bb = bd.getElementsByTagName("Sentence")
        l = len(bb)

        for i in range(0, l):
            if bb[i].hasAttribute("label"):
                # print ("+++++")
                if bb[i].getAttribute("label") == "0":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    listl.append("0")

                if bb[i].getAttribute("label") == "1":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    listl.append("1")

                if bb[i].getAttribute("label") == "2":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    listl.append("2")
    list0 = st0.split("\n")[:-1]
    return list0, listl,lm


def convert_doc_to_wordlist(str_doc, cut_all):
    #print(str_doc)
    # 分词的主要方法
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all)) for part in sent_list]  # 分词
    word_list = sum(word_2dlist, [])
    return word_list

def rm_char(text):
    text = re.sub('\u3000', '', text)
    return text

def get_stop_words(path='./stopwork4.txt'):
    # stop_words中，每行放一个停用词，以\n分隔
    file = open(path, 'rb').read().decode('utf8').split('\n')
    return set(file)


def rm_tokens(words):  # 去掉一些停用次和数字
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:  # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list



#list0, listl = read_xml()


#生成字典！！！！

def dict_te(list0):
    dictionary = corpora.Dictionary()
    for file in list0:
        file = convert_doc_to_wordlist(file, cut_all=False)
        dictionary.add_documents([file])    #参数是表中表[[]]
    return dictionary
#dictionary.filter_n_most_frequent(10)
# dictionary.filter_extremes(no_below=3)
# p=list(dictionary.values())
# print("处理词典: ",p)
# # dictionary.compacity()
# o = input()

# #

#dictionary = dict_te(list0)

def de_dictionary(dictionary):
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 2 ]
    dictionary.filter_tokens(small_freq_ids)
    dictionary.compactify()
    print(dictionary[len(dictionary)-1])
    return dictionary
    # print(dictionary[7207])


def dict_xy(list0,dictionary):#将文本转化成词向量(稀疏向量)
    count = 0
    bow = []
    for file in list0:
        count += 1
        word_list = convert_doc_to_wordlist(file, cut_all=False)
        word_bow = dictionary.doc2bow(word_list)
        bow.append(word_bow)
    return bow


def xy_tfidf(bow):#生成tf-idf向量，但却是稀疏向量！！！！！
    tfidf_model = models.TfidfModel(bow)
    corpus_tfidf = tfidf_model[bow]
    return corpus_tfidf
#
# lsi_model = models.LsiModel(corpus = corpus_tfidf,
#                             id2word = dictionary,
#                             num_topics=50)
# corpus_lsi = lsi_model[bow]
# corpus_tfidf = corpus_lsi
# print(len(corpus_tfidf))
# print(len(list(corpus_tfidf)[0]))


#转化成密集向量
from scipy.sparse import csr_matrix
def xisu_miji(corpus_tfidf,x,y):#将稀疏向量转化成密集向量
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_tfidf:  # lsi_corpus_total 是之前由gensim生成的lsi向量
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    tfi_sparse_matrix = csr_matrix((data, (rows, cols)),shape=(x,y))  # 稀疏向量
    tfi_matrix = tfi_sparse_matrix.toarray()  # 密集向量
    return tfi_matrix
#m = preprocessing.scale(m)

#m = lsi_matrix
from sklearn import preprocessing
#m = preprocessing.scale(m)

def clf_train(m,listl):
    print("样本数： ", len(m))
    print("开始生成模型")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='liblinear', C=1.6, multi_class='ovr')
    print(clf.get_params(clf))
    print("开始训练")
    clf = clf.fit(m, listl)
    print("训练完毕，开始解析dev")
    return clf

def show(x_text,liy,clf):

    print("开始给分")
    fen = clf.score(x_text, liy)                                #开始评分
    print("得分： ", fen)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    y_pred = clf.predict(x_text)
    print("准确率： ", accuracy_score(liy, y_pred))  # 预测正确的/样本总数
    print("精确率： ", precision_score(liy, y_pred, average=None))  # 预测类别为“正确”中的实际正确的样本数/预测类别为”正确“的样本数
    print("回归率: ", recall_score(liy, y_pred, average=None))  # 预测类别为“正确”且实际也正确的样本数/总样本中实际正确的样本数
    print("F1 score: ", f1_score(liy, y_pred, average=None))  # F1 = 2 * (precision * recall) / (precision + recall)


def main():
    list0, listl,l1 = read_xml("SMP2019_ECISA_Train.xml")   #解析xml,这里主要取list0，list1,list0是一个列表，里面是句子文本[txt1,txt2,txt3...]

    dictionary = dict_te(list0)                             #将文本转化成词典
    dictionary = de_dictionary(dictionary)

    bow = dict_xy(list0,dictionary)                 #将文本转化为与字典对应的词频向量
    corpus_tfidf = xy_tfidf(bow)                    #将文本词频向量转化为对应的tf-idf稀疏向量
    m =xisu_miji(corpus_tfidf,len(corpus_tfidf),len(dictionary))    #将tf-idf稀疏向量转化成对应的tf-idf密集向量

    clf = clf_train(m,listl)                                         #建立并训练分类器
    ss,liy,lm= read_xml("SMP2019_ECISA_Dev.xml")      #读取dev测试集，这里主要取liy，lm，lm为表中表，且文本已经分词[[t1],[t2],[t3]...]

    corpus = [dictionary.doc2bow(text) for text in lm]      #将验证文本转化为对应的稀疏词频向量

    corpus_tfidf=xy_tfidf(corpus)                           #将验证稀疏词频向量转化为对应的tf-idf稀疏向量

    x_text = xisu_miji(corpus_tfidf,len(lm),len(dictionary))    #将验证tf-idf稀疏向量转化为对应的密集向量


    show(x_text,liy,clf)
    #x_text = preprocessing.scale(x_text)

if __name__ == '__main__':
    main()

