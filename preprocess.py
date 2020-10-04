#!/usr/bin/env python3

import collections
import glob
import re
import warnings
from typing import List

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.metrics import BigramAssocMeasures
from nltk.probability import ConditionalFreqDist, FreqDist
from tqdm import tqdm

warnings.filterwarnings("ignore")

tqdm.pandas()

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 
        'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"allcaps", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=False,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def datareader(dir, names: List[str]=None, sep=",", filetype="csv", engine="python", is_dir=True):
    '''
    Read in data from file into a pandas DataFrame
    '''
    if not is_dir:
        df = pd.read_csv(dir, sep=sep, names=names, engine=engine)
        df["og_filename"] = dir
        df = df[df.Target != "Donald Trump"]
        return df

    frames = []
    
    if not dir.endswith("/"):
        dir = dir+"/"

    for file in glob.glob(dir+"*."+filetype):
        filename = file.split(dir)[1] 
        sep_f = pd.read_csv(file, sep=sep, names=names, engine=engine)
        sep_f["og_filename"] = filename
        frames.append(sep_f)
    return pd.concat(frames)


def lower(sentence):
    '''
    Lower the sentence
    '''
    return sentence.lower()


def token_replacement(sentence):
    '''
    Some simple pre-processing using token replacement, for example changing all numbers to "number"
    '''
    sentence = re.sub(r'^([\s\d]+)$','number', sentence)
    sentence = re.sub(r'<[^<>]+>','', sentence)
    sentence = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\-_=#])*','httpaddr', sentence)
    sentence = re.sub(r'[\w\.-]+@[\w\.-]+','emailaddr', sentence)
    sentence = re.sub(r'[$|¢|£|¤|¥|֏|؋|৲|৳|৻|૱|௹|฿|៛|₠|-|₽|꠸|﷼|﹩|＄|￠|￡|￥|￦]\d+([., ]?\d*)*', 'money', sentence)
    return sentence


def get_high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    '''
    Gets the high information words using chi square measure
    '''
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)

    return high_info_words


def high_information_words(X, y, title, verbose=False, min_score=5):
    '''
    Get and display info on high info words
    '''
    print(f"\n:: OBTAINING HIGH INFO WORDS [{title}]...")

    labelled_words = []
    amount_words = 0
    distinct_words = set()
    for words, genre in zip(X, y):
        labelled_words.append((genre, words))
        amount_words += len(words)
        for word in words:
            distinct_words.add(word)

    high_info_words = set(get_high_information_words(labelled_words, BigramAssocMeasures.chi_sq, min_score=min_score)) # 2

    if verbose:
        print("\tNumber of words in the data: %i" % amount_words)
        print("\tNumber of distinct words in the data: %i" % len(distinct_words))
        print("\tNumber of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words


def return_high_info(X, y, title="data", min_score=5):
    '''
    Return list of high information words per document
    '''
    try:
        high_info_words = high_information_words(X, y, title, min_score=min_score)

        X_high_info = []
        for bag in X:
            new_bag = []
            for words in bag:
                if words in high_info_words:
                    new_bag.append(words)
            X_high_info.append(new_bag)
    except ZeroDivisionError:
        print("Not enough information too get high information words, please try again with more files.", file=sys.stderr)
        X_high_info = X
    return X_high_info


def normalise(sentence):
    return " ".join(text_processor.pre_process_doc(sentence))


def label_opinion_towards(sentence):
    return int(sentence[0])


def label_target(row):
    target_dict = {
        'Hillary Clinton': 'Hillary', 
        'Legalization of Abortion': 'Abortion', 
        'Atheism': 'Atheism',
        'Climate Change is a Real Concern': 'Climate',
        'Feminist Movement': 'Feminism',
    }
    return target_dict[row]


def preprocessing(df, title):
    '''
    Apply preprocessing steps
    '''
    df['text_processed'] = df.Tweet.apply(token_replacement).progress_apply(normalise)
    df["Opinion Towards"] = df["Opinion Towards"].apply(label_opinion_towards)
    df["Target_num"] = df["Target"].apply(label_target)
    return df        


def main():
    stance_data = datareader("./data")
    stance_data['text_processed'] = stance_data.Tweet.apply(lower).apply(token_replacement)
    print(stance_data)

if __name__ == '__main__':
    main()
