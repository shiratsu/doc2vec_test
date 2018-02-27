# -*- coding: utf-8 -*-
#

from sklearn.ensemble import RandomForestClassifier
import MeCab
import sys
from gensim import corpora

# 結局使わないことになったけど、pythonのyieldの使い方を覚えた
def makeWakatiData(mecab,sentence,words):

    print("---------------------")
    node = mecab.parse(sentence)
    print(sentence)

    for chunk in node.splitlines()[:-1]:
        nodeParts = []
        (surface, feature) = chunk.split('\t')

        if surface not in words:
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
            words = makeWakatiData(mecab,line,words)
    
    print(words)
    dictionary = corpora.Dictionary.load_from_text(args[1]+'.dict')

    # BoW
    vec = dictionary.doc2bow(words)
    print(vec)
