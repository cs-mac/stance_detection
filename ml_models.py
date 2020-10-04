#!/usr/bin/env python3

import itertools
from collections import defaultdict

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn import preprocessing, svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVR, LinearSVC, NuSVC
from textblob import TextBlob

le = preprocessing.LabelEncoder()
use_glove = False
glove = {}

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = {}
        if use_glove:
            create_glove(tweets, train=False)
        features['text'] = [x for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['text_high'] = [x_high for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['sentence'] = [sentence for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['sentiment'] = [sentiment for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['sentence_length'] = [len(sentence) for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['opinion'] = [[opinion] for (x, x_high, sentence, sentiment, opinion, target) in tweets]
        features['target'] = [[i] for i in le.fit_transform([target for (x, x_high, sentence, sentiment, opinion, target) in tweets])]

        return features


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def identity(x):
    return x


def create_glove(tweets, train=False):
    global glove
    if train:
        with open("data/glove.twitter.27B.200d.txt", "rb") as lines:
            wvec = {line.split()[0].decode("utf-8"): np.array(line.split()[1:],dtype=np.float32)
                    for line in lines}
        X = [x for (x, *_) in tweets]
        model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)   
        glove = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}                 
    else:
        tweets = [x for (x, *_) in tweets]
        all_words = set(itertools.chain.from_iterable(tweets))
        with open("data/glove.twitter.27B.200d.txt", "rb") as infile:
            for line in infile:
                parts = line.split()
                word = parts[0].decode("utf-8")
                if (word in all_words):
                    nums=np.array(parts[1:], dtype=np.float32)
                    glove[word] = nums


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


class SentimentContinuous(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        sentiment = []
        for tweet in tweets:
            blob = TextBlob(tweet)
            sentiment.append([blob.sentiment.polarity])
        return sentiment


def model_words():
    '''
    The model + pipeline for features extracted from the text
    '''
    clfs = [LinearSVC(), 
            svm.SVC(kernel='linear', C=1.0), 
            PassiveAggressiveClassifier(C=1, max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
            PassiveAggressiveClassifier(C=0.1, max_iter=1500, tol=0.01, n_jobs=-1, class_weight="balanced", fit_intercept=False, loss="squared_hinge"),
            AdaBoostClassifier(n_estimators=200),
            MultinomialNB(),
            ]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list = [

                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_high')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                ])),

                ('word_n_grams', Pipeline([
                    ('selector', ItemSelector(key='sentence')),
                    ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,5)))
                ])),

                ('char_n_grams', Pipeline([
                    ('selector', ItemSelector(key='sentence')),
                    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2,5)))
                ])),

                ('sentiment', Pipeline([
                    ('selector', ItemSelector(key='sentiment')),
                    ('tfidf', TfidfVectorizer(analyzer='char'))
                ])),

                ('opinion_towards', Pipeline([
                    ('selector', ItemSelector(key='opinion')),
                ])),

                ('target', Pipeline([
                    ('selector', ItemSelector(key='target')),
                ])),

                #### FEATURES THAT DO NOT WORK ####

                # ('sentiment_cont', Pipeline([
                #     ('selector', ItemSelector(key='sentence')),
                #     ('feature', SentimentContinuous())
                # ])),

                # ('glove', Pipeline([
                #     ('selector', ItemSelector(key='sentence')),
                #     ('tfidf', TfidfEmbeddingVectorizer(glove))
                # ])),

                # ('sentence_length', Pipeline([
                #     ('selector', ItemSelector(key='sentence_length')),
                #     ('scaler', MinMaxScaler())
                # ])),

            ],

            # weight components in FeatureUnion
            transformer_weights = { 
                'text_high': 1,
                'word_n_grams': .8,
                'char_n_grams': .8,
                'sentiment': .8,
                'opinion_towards': 1,      
                'target': 1,               
            },
        )),
        # Use a classifier on the combined features
        ('clf', clfs[2]),
    ])
    return classifier
