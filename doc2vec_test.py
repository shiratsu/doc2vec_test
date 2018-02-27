# -*- coding: utf-8 -*-
#
# doc2vecの実験
#
import sys

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import MeCab


def makeWakatiData(mecab,sentence):

    node = mecab.parse(sentence)

    sentences = []
    for chunk in node.splitlines()[:-1]:
        nodeParts = []
        (surface, feature) = chunk.split('\t')
        sentences.append(surface)
    return sentences

if __name__ == '__main__':

    args = sys.argv
    
    # 空のリストを作成（学習データとなる各文書を格納）
    training_docs = []
    arySentence = []

    # mecab
    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    mecab.parse('')

    # 形態素解析して、それをファイルに書き込み
    with open(args[1]+'.txt','r') as f:


        lines = f.readlines()

        for line in lines:
            arySentence.append(makeWakatiData(mecab,line))

    # doc2vecに噛ましていく
    # 各文書を表すTaggedDocumentクラスのインスタンスを作成
    # words：文書に含まれる単語のリスト（単語の重複あり）
    # tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
    for i,sentence in enumerate(arySentence):
        print("----------------")
        print(sentence)
        print('d'+str(i))
        print("----------------")
        sent = TaggedDocument(words=sentence, tags=['d'+str(i)])
        # 各TaggedDocumentをリストに格納
        training_docs.append(sent)


    model = Doc2Vec(documents=training_docs, min_count=1, dm=0)

    print('\n訓練開始')
    for epoch in range(20):
        print('Epoch: {}'.format(epoch + 1))
        model.train(training_docs)
        model.alpha -= (0.025 - 0.0001) / 19
        model.min_alpha = model.alpha

    # 学習したモデルを保存
    model.save(args[1]+'.model')

