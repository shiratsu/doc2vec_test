# -*- coding: utf-8 -*-
#
# yes no判定の処理をrandomforestで行えるように

import MeCab
from gensim import corpora
import sys
import test_doc2vec_model
from gensim import corpora

# 結局使わないことになったけど、pythonのyieldの使い方を覚えた
#def makeWakatiData(mecab,sentence):
#
#    node = mecab.parse(sentence)
#
#    for chunk in node.splitlines()[:-1]:
#        nodeParts = []
#        (surface, feature) = chunk.split('\t')
#        yield surface


if __name__ == '__main__':

    args = sys.argv

    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    words = []

    # ファイルから読み込んで文章取得
    # 形態素解析して、それをファイルに書き込み
    with open(args[1]+'.txt','r') as f:

        lines = f.readlines()

        for line in lines:
            words.append(test_doc2vec_model.makeWakatiData(mecab,line))

    dictionary = corpora.Dictionary(words)

    print("-------------------token-------------------")
    print(words)
    print("-------------------token_id-------------------")
    print(dictionary.token2id)
    dictionary.save_as_text(args[1]+'.dict')
    
