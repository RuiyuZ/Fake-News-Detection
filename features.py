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
from tokenizer import *


def raw_count(document, documents_words):
    """using documents_words (all words in dataset) as standard vector length,
    and calculate the occurrence of every word in the document"""
    fdist = nltk.FreqDist(document)
    lst = [fdist[word] for word in documents_words]
    return lst

def tagger(document):
    tagged = nltk.pos_tag(document)
    return [(token, upenn_to_universal(pos)) for token, pos in tagged]


def pos_tagger(document):
    """calculate the ratio of occurrences of adj, adv, noun, verb and num in the document"""
    length = len(document)
    # tagged = nltk.pos_tag(document)
    tagged = tagger(document)
    adjcounter = 0
    advcounter = 0
    nouncounter = 0
    verbcounter = 0
    numcounter = 0
    for word, tag in tagged:
        if tag == 'ADJ':
            adjcounter += 1
        elif tag == 'ADV':
            advcounter += 1
        elif tag == 'NOUN':
            nouncounter += 1
        elif tag == 'VERB':
            verbcounter += 1
        elif tag == "NUM":
            numcounter += 1
    
    return [round(adjcounter/length, 2), round(advcounter/length, 2), round(nouncounter/length, 2),
            round(verbcounter/length,2), round(numcounter/length,2)]


def senti_features(document, documents_words):
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



def senti_features2(document, documents_words):
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
                total_score += 1
        else:
            # If the word is not in the document, mark it with "0"
            total_score += 1
    return [total_score/len(document)]


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

dict_pos_convert = {'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
                    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
                    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
                    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB',
                    'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
                    'CD': 'NUM'}

def upenn_to_universal(pos):
    if pos in dict_pos_convert:
        return dict_pos_convert[pos]
    else:
        return 'X'
