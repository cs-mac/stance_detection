#!/usr/bin/env python3

# Commands to run the model
# python classifier.py --train data/train.csv --test data/test.csv

import argparse
import os
import pickle
import subprocess
import sys
from itertools import product

import joblib
import numpy as np
import pandas as pd
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from get_normalisation import *
from ml_models import *
from preprocess import *
from utilities import *


def do_grid_search(X, y, pipeline, parameters, title="", cv=10, start=False):
    '''
    Do K-fold cross-validated gridsearch over certain parameters and
    print the best parameters found according to accuracy
    '''
    if not start:
        print("\n#### SKIPPING GRIDSEARCH ...")
    else:
        print(f"\n#### GRIDSEARCH [{title}] ...")
        grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1_weighted', return_train_score=True, verbose=10, n_jobs=6) 
        grid_search.fit(X, y)

        df = pd.DataFrame(grid_search.cv_results_)[['params','mean_train_score','mean_test_score']]
        print(f"\n{df}\n")

        # store results for further evaluation
        with open('grid_' + title + '_pd.pickle', 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)        
   
        print("Best score: {0}".format(grid_search.best_score_))  
        print("Best parameters set:")  
        best_parameters = grid_search.best_estimator_.get_params()  
        for param_name in sorted(list(parameters.keys())):  
            print("\t{0}: {1}".format(param_name, best_parameters[param_name])) 


def train(pipeline, X, y, categories, verbose=False, show_plots=False, show_cm=False, show_report=False, folds=10, title="title"):
    '''
    Train the classifier and evaluate the results
    '''
    print(f"\n#### TRAINING... [{title}]")
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"Classifier used: {pipeline.named_steps['clf']}")

    accuracy = 0
    confusion_m = np.zeros(shape=(len(categories),len(categories)))
    kf = StratifiedKFold(n_splits=folds, random_state = 42).split(X, y)
    pred_overall = np.array([])
    y_test_overall = np.array([])
    for train_index, test_index in kf: 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trained  = pipeline.fit(X_train, y_train) 
        pred = pipeline.predict(X_test)
        accuracy += accuracy_score(y_test, pred)
        confusion_m = np.add(confusion_m, confusion_matrix(y_test, pred, labels=categories))
        pred_overall = np.concatenate([pred_overall, pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    print("\nAverage accuracy: %.5f"%(accuracy/folds) + "\n")

    if show_report:
        print('Classification report\n')
        print(classification_report(y_test_overall, pred_overall, digits=2))
    if show_cm:        
        print('\nConfusion matrix\n')
        print(confusion_m)

    create_confusion_matrix(confusion_m, categories, y_lim_value=3.0, title=title, show_plots= show_plots, method="TRAINING")
   

def test(classifier, Xtest, Ytest, verbose=False, show_cm=False, show_plots=False, show_report=False, title="title"):
    '''
    Test the classifier and evaluate the results
    '''    
    print(f"\n#### TESTING... [{title}]")

    reverse_list = ["FAVOR", "AGAINST", "NONE"]
    Yguess = classifier.predict(Xtest)
    Ytest = [reverse_list[sent] for sent in Ytest]
    Yguess = [reverse_list[sent] for sent in Yguess]

    if verbose:
        print(f"Classifier used: {classifier.named_steps['clf']}")
        targets = [i[-1] for i in Xtest]
        df = pd.DataFrame({"x": [" ".join(x) for (x, *_) in Xtest], "y": Ytest, "y_guess": Yguess, "target": targets})
        for target in set(targets):
            sub_df = df.loc[df['target'] == target]
            print(f"Target = {target}")
            # print(classification_report(sub_df.y, sub_df.y_guess, digits=4))
            with open("results/gold_target.txt", "w") as f:
                f.write("ID\tTarget\tTweet\tStance\n")
                for idx, row in sub_df.iterrows():
                    f.write(str(idx)+"\t"+target+"\t"+row.x+"\t"+row.y+"\n")
            with open("results/guess_target.txt", "w") as f:
                f.write("ID\tTarget\tTweet\tStance\n")
                for idx, row in sub_df.iterrows():
                    f.write(str(idx)+"\t"+target+"\t"+row.x+"\t"+row.y_guess+"\n")
            perl_script = subprocess.Popen(["./eval.pl", "results/gold_target.txt",  "results/guess_target.txt"], stdout=sys.stdout)
            perl_script.communicate()

    confusion_m = np.zeros(shape=(len(reverse_list), len(reverse_list)))

    print(f"\naccuracy = {round(accuracy_score(Ytest, Yguess), 5)}")

    if show_report:
        print('\nClassification report\n')
        print(classification_report(Ytest, Yguess, digits=4))

    confusion_m = np.add(confusion_m, confusion_matrix(Ytest, Yguess, labels = reverse_list))
    if show_cm:
        print('\nConfusion matrix')
        print(confusion_m)

    create_confusion_matrix(confusion_m, reverse_list, y_lim_value=3.0, title=title, show_plots=show_plots, save_plots=True, method="TESTING")
    return Yguess


def main(argv):
    parser = argparse.ArgumentParser(description='Control everything')
    parser.add_argument('--train', help="Provide the name of the training file")
    parser.add_argument('--test', help="Provide the name of the testing file")
    parser.add_argument('--model', help="Please provide a .pkl model")
    parser.add_argument('--save', help="Use: --save [filename] ; Saves the model, with the given filename")
    args = parser.parse_args()

    print(f"#### READING DATA & PREPROCESSING...\n")
    df_train = datareader(args.train, is_dir=False)
    df_test = datareader(args.test, is_dir=False)

    df_train = preprocessing(df_train, title="normalised_df_train.pkl")
    df_test = preprocessing(df_test, title="normalised_df_test.pkl")

    X_train, y_train = [x.split() for x in df_train["text_processed"]], df_train["Stance"]
    X_test, y_test = [x.split() for x in df_test["text_processed"]], df_test["Stance"]
    X_train_high_info = return_high_info(X_train, y_train, "train", min_score=2)
    X_test_high_info = return_high_info(X_test, y_test, "test", min_score=2)
    X_train = [(x, x_high, sentence, sentiment, opinion, target) for x, x_high, sentence, sentiment, opinion, target in zip(X_train, 
                                                                                X_train_high_info, 
                                                                                df_train["text_processed"], 
                                                                                df_train["Sentiment"],
                                                                                df_train["Opinion Towards"],
                                                                                df_train["Target_num"])]
    X_test = [(x, x_high, sentence, sentiment, opinion, target) for x, x_high, sentence, sentiment, opinion, target in zip(X_test, 
                                                                                X_test_high_info, 
                                                                                df_test["text_processed"],
                                                                                df_test["Sentiment"],
                                                                                df_test["Opinion Towards"],
                                                                                df_train["Target_num"])]

    extra_none, extra_favor = [], []
    for (x, x_high, sentence, sentiment, opinion, target), label in zip(X_train, y_train):
        if label == "NONE":
            extra_none.append((x, x_high, sentence, sentiment, opinion, target))
        if label == "FAVOR":
            extra_favor.append((x, x_high, sentence, sentiment, opinion, target))
    extra_amount = 150
    X_train = X_train + extra_none[:extra_amount] + extra_favor[:extra_amount]
    y_train = y_train.to_list() + (["NONE"]*len(extra_none))[:extra_amount] + (["FAVOR"]*len(extra_favor))[:extra_amount]

    translation_dict = {"FAVOR": 0, "AGAINST": 1, "NONE": 2}
    y_train = np.array([translation_dict[tweet] for tweet in y_train])
    y_test = np.array([translation_dict[tweet] for tweet in y_test])

    # print_distribution(y_train)
    # print_distribution(y_test)

    # GridSearch Parameters
    feature_weighting_cartesian_prod = { 
        'text_high': [0.0, 0.5, 1.0],
        'word_n_grams': [0.0, 0.5, 1.0],
        'char_n_grams': [0.0, 0.5, 1.0],
        'sentiment': [0.0, 0.5, 1.0],
        'opinion_towards': [1],      
        'target': [1],               
    }

    feature_weightings_to_check = [dict(zip(feature_weighting_cartesian_prod, v)) for v in product(*feature_weighting_cartesian_prod.values())]

    parameters = {
        'linear': {  
            'clf__C': np.logspace(-3, 2, 6),
        },
        'rbf': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__gamma': np.logspace(-3, 2, 6),
            'clf__kernel': ['rbf']
        },
        'poly': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__gamma': np.logspace(-3, 2, 6),
            'clf_degree': np.array([0,1,2,3,4,5,6]),
            'clf__kernel': ['linear']
        },
        'PA': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__fit_intercept': [False], # True, False
            'clf__max_iter': np.array([1000, 3000, 7500, 1500]),
            'clf__loss': ['squared_hinge'], # hinge, squared_hinge
            'clf__class_weight': ['balanced'], # balanced, None
            'clf__tol': np.logspace(-3, -1, 3),
        },
        'features': {
            'union__transformer_weights': 
                feature_weightings_to_check
        }
    }
    algorithms = ['linear', 'rbf', 'poly', 'PA', 'features']

    classifier_words = model_words()

    start_grid = False
    do_grid_search(X_train, y_train, classifier_words, parameters[algorithms[4]], title=algorithms[4], start=start_grid)
    if start_grid:
        return 0

    train(classifier_words, X_train, y_train, categories=[0, 1, 2], show_report=False, title="[Train]", folds=10)
    predictions = test(classifier_words, X_test, y_test, show_report=True, verbose=True, title="[Test]")

    print(f"\n#### CREATING OUTPUT FILES...")
    if os.path.isfile('results/gold_test.txt'):
        print("\n:: Gold evaluation file already exists, skipping creation...")
    else:
        with open("results/gold_test.txt", "w") as f:
            f.write("ID\tTarget\tTweet\tStance\n")
            for index, row in df_test.iterrows():
                f.write(str(index)+"\t"+row["Target"]+"\t"+row["Tweet"]+"\t"+row["Stance"]+"\n")
        print("\n:: Created gold evaluation file!")

    with open("results/guess_test.txt", "w") as f:
        f.write("ID\tTarget\tTweet\tStance\n")
        for prediction, (index, row) in zip(predictions, df_test.iterrows()):
            f.write(str(index)+"\t"+row["Target"]+"\t"+row["Tweet"]+"\t"+prediction+"\n")
    print("\n:: Created prediction evaluation file!")

    if args.model:
        the_classifier = joblib.load(args.model)
        test(the_classifier, X_test, y_test, title=f'saved_model{args.model}')
    else:
        if args.save:
            joblib.dump(classifier_words, args.save+".pkl") 

    perl_script = subprocess.Popen(["./eval.pl", "results/gold_test.txt",  "results/guess_test.txt"], stdout=sys.stdout)
    perl_script.communicate()

    perl_script = subprocess.Popen(["./eval_indirect.pl", "results/gold_test.txt",  
            "results/guess_test.txt", "data/taskA_annotation_file_for_direct_indirect_targets.csv", "1"], stdout=sys.stdout)
    perl_script.communicate()

    perl_script = subprocess.Popen(["./eval_indirect.pl", "results/gold_test.txt",  
            "results/guess_test.txt", "data/taskA_annotation_file_for_direct_indirect_targets.csv", "2"], stdout=sys.stdout)
    perl_script.communicate()

if __name__ == '__main__':
    main(sys.argv)
