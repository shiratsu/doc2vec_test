# -*- coding: utf-8 -*-
#
# pickleで保存したものが、果たして利用できるのか確認する。

import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier
import make_random_forest
import sys
import MeCab
from gensim import corpora, matutils

# 予測を行う
def predictData(strDictName,strModelName,sentence,mecab):

    words = []

    # モデルをロード
    with open(strModelName, 'rb') as f:
        model = cPickle.load(f)

    # 引数に入ってきた文字列を分解して入れる
    words.append(make_random_forest.makeWakatiData(mecab,sentence))

    #print(words)
    dictionary = corpora.Dictionary.load_from_text(strDictName)

    # BoW
    corpus = [dictionary.doc2bow(text) for text in words]
    
    aryDense = []
    
    # ベクトルを作成
    for c in corpus:
        dense = list(matutils.corpus2dense([c], num_terms=len(dictionary)).T[0])
        print(dense)
        aryDense.append(dense) 
    
    result = model.predict(aryDense)

    return result

if __name__ == '__main__':

    args = sys.argv

    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    
    result = predictData(args[1],args[2],args[3],mecab)

    print(result)
