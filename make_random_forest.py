# -*- coding: utf-8 -*-
#

from sklearn.ensemble import RandomForestClassifier
import MeCab
import sys
from gensim import corpora, matutils
from gensim import models

# 結局使わないことになったけど、pythonのyieldの使い方を覚えた
def makeWakatiData(mecab,sentence):

    print("---------------------")
    node = mecab.parse(sentence)
    print(sentence)
    words = []

    for chunk in node.splitlines()[:-1]:
        nodeParts = []
        (surface, feature) = chunk.split('\t')

        #if surface not in words:
        #    words.append(surface)
        if surface != '':
            words.append(surface)

    return words

#def get_words_main(mecab,content):
#    '''
#    一つの記事を形態素解析して返す
#    '''
#    return token for token in makeWakatiData(mecab,content)

if __name__ == '__main__':
    
    args = sys.argv

    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    words = []

    # ファイルから読み込んで文章取得
    # 形態素解析して、それをファイルに書き込み
    with open(args[1]+'.txt','r') as f:

        lines = f.readlines()

        for line in lines:
            words.append(makeWakatiData(mecab,line.strip()))
    
    print(words)
    dictionary = corpora.Dictionary.load_from_text(args[1]+'.dict')

    # BoW
    corpus = [dictionary.doc2bow(text) for text in words]
    
    aryDense = []
    
    # ベクトルを作成
    for c in corpus:
        dense = list(matutils.corpus2dense([c], num_terms=len(dictionary)).T[0])
        aryDense.append(dense) 

    # 正解ラベルを定義
    aryAnswer = [1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,1 \
                ,2 \
                ,2 \
                ,2 \
                ,2 \
                ,1 \
                ,2 \
                ,2 \
                ,1 \
                ,1 \
                ,1 \
                ,2 \
                ,1 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ,3 \
                ]

    estimator = RandomForestClassifier()

    # 学習させる
    estimator.fit(aryDense, aryAnswer)

    # 予測
    label_predict = estimator.predict(aryDense)
    print(label_predict)
