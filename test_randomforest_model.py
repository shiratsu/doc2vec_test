# -*- coding: utf-8 -*-
#
# pickleで保存したものが、果たして利用できるのか確認する。

import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import Doc2Vec
import doc2vec_test_label
import sys
import MeCab
from gensim import corpora, matutils

# 予測を行う
def predictData(strModelName,sentence,mecab):

    words = []

    # モデルをロード
    with open(strModelName+'.pickle', 'rb') as f:
        forest = cPickle.load(f)

    # 引数に入ってきた文字列を分解して入れる
    sentences = doc2vec_test_label.makeWakatiData(mecab,args[2])

    model = Doc2Vec.load(strModelName+'.model')
    infered_vecor = model.infer_vector(sentences)
    print(infered_vecor)
    result = model.docvecs.most_similar([infered_vecor])
    print(result)
    vector = model.docvecs[result[0][0]]
    print(vector)

    result2 = forest.predict([vector])

    return result2

if __name__ == '__main__':

    args = sys.argv

    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    
    result = predictData(args[1],args[2],mecab)

    print(result)
