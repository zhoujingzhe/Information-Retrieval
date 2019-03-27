from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import editdistance
import re
import numpy as np
import operator
import pandas as pd
import collections
import time
import itertools

epsilon = np.finfo(float).eps
def get_cosine_score(q_tfidf, tf_idf):
    matrix = q_tfidf * tf_idf
    matrix = np.sum(a=matrix, axis=0)
    q_matrix = np.sqrt(np.sum(a=np.square(q_tfidf), axis=0))
    d_matrix = np.sqrt(np.sum(a=np.square(tf_idf), axis=0))
    result = matrix / (q_matrix * d_matrix)
    index = np.argsort(a=result, axis=-1)[::-1]
    index = index[:10:]
    return index + 1
def get_score(q_tfidf, tf_idf):
    q_word_matrix = q_tfidf.copy()
    word_matrix = tf_idf.copy()
    q_word_matrix[q_word_matrix > 0] = 1
    word_matrix[word_matrix > 0] = 1
    matrix = word_matrix * q_word_matrix
    index = np.argsort(a=np.sum(a=matrix * tf_idf, axis=0), axis=-1)[::-1]
    index = index[:10:]
    return index + 1
def generateTokens(contents, filePath):
    tokens = re.sub(r"[^a-zA-Z0-9\s]|[\x00-\x08\x0b-\x1f\x7f-\xff]+", '', contents)
    tokens = re.sub(r"\b\w{,1}\b", '', tokens).lower().split()
    return np.array(list(zip(tokens, itertools.repeat(filePath))))
def getsimilarword(word, dictionary):
    distance = []
    for i in range(len(dictionary)):
        distance.append(editdistance.eval(word, dictionary[i]))
    index = np.argmin(distance)
    return dictionary[index]

def generate_frequency_matrix(x, index, word_document_matrix, idf_matrix, document_num):
    counter = collections.Counter(x)
    word_document_matrix[index, np.array(list(map(int, map(float, np.unique(x))))) - 1] = np.array(list(counter.values()))
    idf_matrix[index] = np.log10(document_num / float(np.count_nonzero(word_document_matrix[index])))
    return word_document_matrix, idf_matrix

def pre_processing_line(line):
    line = line.strip()
    line = line.lower()
    words = re.split(r'[\s]', line)
    words = [re.sub(r'[^\w\s]', '', temp) for temp in words]
    words = list(filter(None, words))
    return words

def sorting_tokens(tokens):
    temp = tokens[:].copy()
    tokens = sorted(tokens, key=operator.itemgetter(0, 1))
    temp = list(zip(temp[:, 0], temp[:, 1]))
    tmp = sorted(temp)
    if np.array_equal(a1=tmp, a2=tokens) is False:
        print('exception on sorting!')
    return tokens

def Posting_list_merge(query_words, inverted_list):
    document_list = None
    operator_list = []
    for word in query_words:
        if word not in ['and', 'or', 'not']:
            document_ids = inverted_list.get(word)
            if document_ids is None:
                print('the word:'+str(word)+'is not in database!')
                word = getsimilarword(word=word, dictionary=inverted_list.index)
                document_ids = inverted_list.get(word)
                print('This word is replaced by '+str(word))
            document_ids = np.reshape(a=document_ids, newshape=(-1, ))
            document_ids = list(map(float, document_ids))
            document_ids = list(map(int, document_ids))
            document_array = np.zeros((document_num, ))
            document_ids = np.array(document_ids)
            document_ids = document_ids - 1
            document_array[document_ids] = 1
            if document_list is None:
                document_list = document_array
            operator_list = operator_list[::-1]
            for operator in operator_list:
                if operator == 'and':
                    document_list = document_list * document_array
                elif operator == 'or':
                    document_list = document_list + document_array
                    document_list[document_list > 1] = 1
                elif operator == 'not':
                    document_array = 1 - document_array
                    if len(operator_list) == 1:
                        document_list = document_array
            operator_list.clear()
        else:
            operator_list.append(word)
    return document_list

import sys
if __name__ == "__main__":
    tokens = None
    document_num = 7945
    # document_sizes = np.zeros((document_num,))
    start1 = time.time()
    dictionary = dict()
    snowstemmer = SnowballStemmer("english")
    for i in range(document_num):
        filename = './data/HillaryEmails/' + str(i + 1) + '.txt'
        print('the file:', i+1)
        f = open(file=filename, mode='r', encoding='UTF-8')
        content = f.read()
        f.close()
        token = generateTokens(contents=content, filePath=i+1)
        for j in range(len(token)):
            tmp = snowstemmer.stem(token[j, 0])
            if dictionary.get(tmp) is None:
                dictionary[tmp] = [token[j, 1]]
            else:
                dictionary[tmp].append(token[j, 1])
    print('the size of dictionary:', sys.getsizeof(dictionary))
    print('the time on creating index:', time.time() - start1)
    start2 = time.time()
    # step5: inputting query word
    query = 'Department and youth'
    query_words = pre_processing_line(query)
    query_words = [snowstemmer.stem(plural) for plural in query_words]
    query_words = np.reshape(a=query_words, newshape=(-1, ))
    print('query words:', query_words)

    # step6: information retrieving
    document_list = Posting_list_merge(query_words=query_words, inverted_list=dictionary)
    print('The searching result is:', np.nonzero(document_list)[0] + 1)
    print('the time consumed on retrieving', time.time()-start2)














    # calculate tf-idf

    # tf_matrix = np.zeros((len(inverted_list.index), document_num))
    # idf_matrix = np.zeros((len(inverted_list.index)))
    # d_f = np.reshape(a=inverted_list.values, newshape=(-1,))
    # index = 0
    # for y in d_f:
    #     tf_matrix, idf_matrix = generate_frequency_matrix(x=y, index=index, word_document_matrix=tf_matrix,
    #                                                       idf_matrix=idf_matrix, document_num=document_num)
    #     index = index + 1
    # document_sizes = np.reshape(a=document_sizes, newshape=(1, -1))
    # tf = np.log(1 + np.divide(tf_matrix, document_sizes+epsilon))
    # idf = np.reshape(a=idf_matrix, newshape=(-1, 1))
    # tf_idf = tf * idf
    # print('the size of tf-idf:', sys.getsizeof(tf_idf))
    # word_indexes = list(inverted_list_snow.index)
    # query1 = 'department youth'
    # query1 = pre_processing_line(line=query1)
    # query1 = [snowstemmer.stem(plural) for plural in query1]
    # qword_list = []
    # for i in range(len(query1)):
    #     try:
    #         qword_list.append(word_indexes.index(query1[i]))
    #     except ValueError:
    #         word = getsimilarword(word=query1[i], dictionary=inverted_list.index)
    #         qword_list.append(word_indexes.index(word))
    #
    # counter = collections.Counter(qword_list)
    # q_tf = np.zeros((len(inverted_list.index)))
    # q_tf[np.array(list(map(int, map(float, np.unique(qword_list)))))] = np.array(list(counter.values()))
    # q_tf = np.log(1 + np.divide(q_tf, len(query1)))
    # q_tf = np.reshape(a=q_tf, newshape=(-1, 1))
    # q_tfidf = q_tf * idf
    # end2 = time.time()
    # get_cosine_score(q_tfidf=q_tfidf, tf_idf=tf_idf)
    # print('the time spent on retrieval is:', time.time()-end2)