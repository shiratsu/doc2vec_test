# -*- coding: utf-8 -*-
#
# doc2vecモデルのテスト
#
from gensim.models.doc2vec import Doc2Vec
import sys
import MeCab

def makeWakatiData(mecab,sentence):

    node = mecab.parse(sentence)

    sentences = []
    for chunk in node.splitlines()[:-1]:
        nodeParts = []
        (surface, feature) = chunk.split('\t')
        sentences.append(surface)
    return sentences

args = sys.argv

# mecab
mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
mecab.parse('')

sentences = makeWakatiData(mecab,args[2])

model = Doc2Vec.load(args[1]+'.model')
infered_vecor = model.infer_vector(sentences)

# 文章を指定してそれに近い文章を抽出する

#result = model.docvecs.most_similar([sentences])
result = model.docvecs.most_similar([infered_vecor])

print(result)
