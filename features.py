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
            round(verbcounter/length, 2), round(numcounter/length, 2)]


SENTI_DEFAULT_VALUE = 1


def senti_features(document, documents_words):
    '''define a feature extractor that return a list of sentiment scores.'''
    document_words = set(document)
    feature = []
    # For every word in the word list,
    # if it is present in a given document, check for its score in SentiWordNet.
    for word in documents_words:
        if word in document_words:
            try:
                score = list(swn.senti_synsets(word, 'a'))[0]
                if (score.pos_score() > score.neg_score()):
                    # If the word's positive score is higher
                    # calculate the score by the default value + the positive score
                    feature.append(SENTI_DEFAULT_VALUE + score.pos_score())
                elif (score.neg_score() > score.pos_score()):
                    # If the word's negative score is higher
                    # calculate the score by the default value - the negative score
                    feature.append(SENTI_DEFAULT_VALUE - score.neg_score())
                else:
                    # If the word’s positive and negative score are same
                    # record the score as the default value
                    feature.append(SENTI_DEFAULT_VALUE)
            except IndexError:
                # If the word is not in SentiWordNet, mark it with the default value
                feature.append(SENTI_DEFAULT_VALUE)
        else:
            # If the word is not in the document, mark it with the default value
            feature.append(SENTI_DEFAULT_VALUE)
    # return a list of scores as a feature
    return feature


def senti_features2(document, documents_words):
    '''define a feature extractor that return an average sentiment score.'''
    total_score = 0
    # For every word in the word list,
    # if it is present in a given document, check for its score in SentiWordNet.
    for word in documents_words:
        if word in document:
            try:
                score = list(swn.senti_synsets(word, 'a'))[0]
                if (score.pos_score() > score.neg_score()):
                    # If the word's positive score is higher
                    # calculate the score by the default value + the positive score
                    # and add it to the total score
                    total_score += (SENTI_DEFAULT_VALUE + score.pos_score())
                elif (score.neg_score() > score.pos_score()):
                    # If the word's negative score is higher
                    # calculate the score by the default value - the negative score
                    # and add it to the total score
                    total_score += (SENTI_DEFAULT_VALUE - score.neg_score())
                else:
                    # If the word’s positive and negative score are same
                    # record the score as the default value, and add it to the total score
                    total_score += SENTI_DEFAULT_VALUE
            except IndexError:
                # If the word is not in SentiWordNet
                # record the score as the default value, and add it to the total score
                total_score += SENTI_DEFAULT_VALUE
        else:
            # If the word is not in the document
            # record the score as the default value, and add it to the total score
            total_score += SENTI_DEFAULT_VALUE
    # return a average score as a feature
    return [total_score/len(document)]


# assume the maximum word length is 20
MAX_WORD_LENGTH = 20


def word_length_features(document):
    # construct an empty list
    counts = [0]*MAX_WORD_LENGTH
    # count the number of words with a certain length and store the numbers in the list
    for word in document:
        try:
            counts[len(word)-1] += 1
        except Exception:
            # ignore the words longer than 20 characters
            pass
    # calculate the percentages by dividing the total number of words in a document
    percentage = np.around(np.array(counts)/len(document)*100)
    return percentage


dict_pos_convert = {'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
                    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
                    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
                    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB',
                    'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
                    'CD': 'NUM'}


# convert nltk POS tags to universal POS tags
def upenn_to_universal(pos):
    if pos in dict_pos_convert:
        return dict_pos_convert[pos]
    else:
        return 'X'
