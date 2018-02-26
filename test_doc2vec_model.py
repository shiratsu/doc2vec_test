# -*- coding: utf-8 -*-
#
# doc2vecモデルのテスト
#
from gensim.models.doc2vec import Doc2Vec
import sys

args = sys.argv

model = Doc2Vec.load(args[1]+'.model')

# 文章を指定してそれに近い文章を抽出する

result = model.docvecs.most_similar(args[2])

print(result)
