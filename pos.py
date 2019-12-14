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
import numpy as np
import pprint


def raw_count(document):
    """using documents_words (all words in dataset) as standard vector length,
    and calculate the occurrence of every word in the document"""
    fdist = nltk.FreqDist(document)
    lst = [fdist[word] for word in documents_words]
    return lst


# def pos_tagger(document):
#     """calculate the ratio of occurrences of adj, adv, noun, verb and num in the document"""
#     length = len(document)
#     tagged = nltk.pos_tag(document)
#     adjcounter = 0
#     advcounter = 0
#     nouncounter = 0
#     verbcounter = 0
#     numcounter = 0
#     for word, tag in tagged:
#         if tag == 'ADJ':
#             adjcounter += 1
#         elif tag == 'ADV':
#             advcounter += 1
#         elif tag == 'NOUN':
#             nouncounter += 1
#         elif tag == 'VERB':
#             verbcounter += 1
#         elif tag == "NUM":
#             numcounter += 1
#     return [adjcounter//length, advcounter//length, nouncounter//length,
#             verbcounter//length, numcounter//length]


# def read_file(filename):
#     with open(filename, 'r', encoding="utf-8") as f:
#         reader = csv.reader(f)
#         # read the header
#         header_row = next(reader)
#         data_set = []
#         # store the data in a list
#         for row in reader:
#             data_set.append(row)
#         # tokenize the text
#         for data in data_set:
#             data[0] = clean_text(data[0])
#         return data_set


# def extract_words(data_set):
#     '''construct a sorted list of all the words (no duplicates) in the data set'''
#     words = []
#     for data in data_set:
#         words += data[0]
#     return sorted(set(words))


def clean_text(text):
    '''tokenize the given text, remove numbers and punctuations'''
    # return re.findall(r"\w+|[^\w\s]", text.lower())
    return re.findall(r'[a-z]+', text.lower())


def senti_features(document):
    '''define a feature extractor that marks the words
    which have positive or negative scores over 0.5 in SentiWordNet.'''
    document_words = set(document)
    feature = []
    # For every word in the word list,
    # if it is present in a given document, check for its score in SentiWordNet.
    for word in documents_words:
        if word in document_words:
            try:
                score = list(swn.senti_synsets(word, 'a'))[0]
                if (score.pos_score() > score.neg_score()):
                    # If the word's positive score is higher, record the score as "1 + (the positive score)"
                    feature.append(1 + score.pos_score())
                elif (score.neg_score() > score.pos_score()):
                    # If the word's negative score is higher, record the score as "1 - (the negative score)"
                    feature.append(1 - score.neg_score())
                else:
                    # If the word’s positive and negative score are same, record the score as "1"
                    feature.append(1)
            except IndexError:
                # If the word is not in SentiWordNet, mark it with "0"
                feature.append(0)
        else:
            # If the word is not in the document, mark it with "0"
            feature.append(0)
    return feature



def senti_features2(document):
    '''define a feature extractor that marks the words
    which have positive or negative scores over 0.5 in SentiWordNet.'''
    total_score = 0
    # For every word in the word list,
    # if it is present in a given document, check for its score in SentiWordNet.
    for word in documents_words:
        if word in document:
            try:
                score = list(swn.senti_synsets(word, 'a'))[0]
                if (score.pos_score() > score.neg_score()):
                    # If the word's positive score is higher, record the score as "1 + (the positive score)"
                    total_score += (1 + score.pos_score())
                elif (score.neg_score() > score.pos_score()):
                    # If the word's negative score is higher, record the score as "1 - (the negative score)"
                    total_score += (1 - score.neg_score())
                else:
                    # If the word’s positive and negative score are same, record the score as "1"
                    total_score += 1
            except IndexError:
                # If the word is not in SentiWordNet, mark it with "0"
                total_score += 0
        else:
            # If the word is not in the document, mark it with "0"
            total_score += 0
    return [total_score]


def word_length_features(document):
    # construct an empty list
    counts = [0]*20
    # count the number of words with a certain length and store the numbers in the list
    for word in document:
        try:
            counts[len(word)-1] = counts[len(word)-1]+1
        except:
            pass
    # calculate the percentage by dividing the total number of words in a document
    percentage = np.around(np.array(counts)/len(document)*100)
    return percentage


if __name__ == '__main__':
    """read dataset, convert it to list of tokens"""
    file = open('data/dataset.csv').readlines()
    rawdocuments = [line.split('|') for line in file] # 12836
    file_words = []
    for lst in rawdocuments:
        file_words.extend(clean_text(lst[1]))

    documents_words = sorted(set(w.lower() for w in file_words))  # 22132
    documents = [clean_text(lst[1]) for lst in rawdocuments]  # 1080

    rcfeaturesets = [raw_count(d) for d in documents]
    posfeaturesets = [pos_tagger(d) for d in documents]
    sentifeaturesets = [senti_features(d) for d in documents]
    sentifeaturesets2 = [senti_features2(d) for d in documents]
    wlfeaturesets = [word_length_features(d) for d in documents]
    featuresets = [0, rcfeaturesets, posfeaturesets, sentifeaturesets, sentifeaturesets2,
                   wlfeaturesets]

    labels = [lst[0] for lst in rawdocuments]
    size = int(len(documents) * 0.1)
    train_labels = labels[size:]
    test_labels = labels[:size]
