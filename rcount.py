import sys
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import sentiwordnet as swn
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import time
import re


def raw_count(document):
    fdist = nltk.FreqDist(document)
    lst = [fdist[word] for word in documents_words]
    return lst


def pos_tagger(document):
    # docu = set(w.lower() for w in document)
    length = len(document)
    tagged = nltk.pos_tag(document)
    return tagged
    adjcounter = 0
    advcounter = 0
    nouncounter = 0
    verbcounter = 0
    numcounter = 0
    for word, tag in tagged:
        if tag == 'ADJ':
            adjcounter += 1
        if tag == 'ADV':
            advcounter += 1
        if tag == 'NOUN':
            nouncounter += 1
        if tag == 'VERB':
            verbcounter += 1
        if tag == "NUM":
            numcounter += 1
    return [adjcounter//length, advcounter//length, nouncounter//length,
            verbcounter//length, numcounter//length]
    # lst = []
    # for word in documents_words:
    #     if word in document:
    #         for word,tag in tagged:
    #
    #         # index = document.index(word)
    #         # lst.append(tagged[index][1])
    #     else:
    #         lst.append('NONE')
    # return lst
    # return dict(nltk.FreqDist(lst))


def onehot_encoder(featuresets):
    """input featuresets and output binary features"""
    ohe_encoder = preprocessing.OneHotEncoder()
    ohe_encoder.fit(featuresets)
    binary_features = ohe_encoder.transform(featuresets).toarray()
    return binary_features


def multinomia_train(featuresets, train_labels, test_labels):
    """train model, save trained classifiers, elapsed time, accuracy"""
    model_multinomia = MultinomialNB()
    for i in range(1, len(featuresets)):
        t0 = time.perf_counter()
        if i == 2:
            binary_features = onehot_encoder(featuresets[i])
            model = model_multinomia.fit(binary_features[size:], train_labels)
        if i ==1:
            binary_features = featuresets[i]
            model = model_multinomia.fit(binary_features[size:], train_labels)
        elapsed_time = time.perf_counter() - t0
        accuracy = model.score(binary_features[:size], test_labels)
        # joblib.dump(model, Multinomia_classifiers[i])
        # print('Creating Bayes classifier in', Multinomia_classifiers[i])
        print(' '*3, 'Elapsed time:', str(int(elapsed_time))+'s')
        print(' '*3, 'Accuracy:', accuracy)



train_file = open('liar_dataset/train.tsv').readlines()
test_file = open('liar_dataset/test.tsv').readlines()
valid_file = open('liar_dataset/valid.tsv').readlines()
rawdocuments = [line.split('\t') for file in [train_file, test_file, valid_file]
                for line in file]   # 12836
# print(rawdocuments)
file_words = []
for lst in rawdocuments:
    file_words.extend(nltk.word_tokenize(lst[2]))
documents_words = sorted(set(w.lower() for w in file_words))  # 15030


documents = [nltk.word_tokenize(lst[2]) for lst in rawdocuments]  # 12836
# print(documents[0])
print(len(documents))
rcfeaturesets = [raw_count(d) for d in documents]
posfeaturesets = [pos_tagger(d) for d in documents]
# print(posfeaturesets)
featuresets = [0, rcfeaturesets, posfeaturesets]


labels = [lst[1] for lst in rawdocuments]
size = int(len(documents) * 0.1)
train_labels = labels[size:]
test_labels = labels[:size]


# print(documents_words)
# print(len(documents_words))
# print(documents)
document = ['Fake', 'Happy', 'go', 'education', 'nice','123', 'quickly','dance']
# print(raw_count(document))
# print(pos_tagger(document))

# tagged = nltk.pos_tag(documents[0])
# print(tagged)
# print(multinomia_train(featuresets, train_labels, test_labels))