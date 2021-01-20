from parse_file import parseCSV, analyseFile
from utils import getColumnIndices, getStratifiedSamples, getExperiment2Tuples, \
    getInitialModel, getExperiment2OptimizedParams, getSplits, trainAndSaveFinalModels, loadAndTestModel,\
    testMajorityRuleMethod, testRandomMethod
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
import numpy as np
import sys
import time

numIterations = 1000
models = [
        SGDClassifier(max_iter = numIterations),LogisticRegression(),
        MLPClassifier(),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(), ExtraTreesClassifier(),
        RandomForestClassifier(), ComplementNB(),
        BernoulliNB(),KNeighborsClassifier(), NearestCentroid(),
        RidgeClassifier(),
        SVC(max_iter=numIterations)
]

models_one = [
    LogisticRegression(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

model_names = ["SGDClass","LogReg", "MLP","DecTree","ExtraTree","ExtraTrees","RandForest","ComplementNB","BernoulliNB",
               "KNeighbour","NearestCentroid","Ridge","SVC"]

def experimentVariousAttributes(X,Y):
    column_names = ["LR_train", "LR_test", "LR_train_bal", "LR_test_bal", "MLP_train", "MLP_test", "MLP_train_bal",
                    "MLP_test_bal",
                    "DT_train", "DT_test", "DT_train_bal", "DT_test_bal", "RF_train", "RF_test", "RF_train_bal",
                    "RF_test_bal"]
    results = {}
    best_train = 0.0
    best_test = 0.0
    balanced_best_train = 0.0
    balanced_best_test = 0.0
    doneIters = 0
    X_train, X_test, y_train, y_test = getStratifiedSamples(X, Y, 8, 0.25)

    for comb_index in range(1, X.shape[1]):
        print(comb_index)
        column_indices, binary_string = getColumnIndices(comb_index, X.shape[1])

        if (len(column_indices) > 2):
            X_train_col_subset = X_train[:, column_indices]
            X_test_col_subset = X_test[:, column_indices]
            results[binary_string] = {}

            for ind, model in enumerate(models_one):
                model.fit(X_train_col_subset, y_train)
                y_predict_train = model.predict(X_train_col_subset)
                y_predict_test = model.predict(X_test_col_subset)

                acc_train = accuracy_score(y_train, y_predict_train)
                acc_test = accuracy_score(y_test, y_predict_test)
                balanced_acc_train = balanced_accuracy_score(y_train, y_predict_train)
                balanced_acc_test = balanced_accuracy_score(y_test, y_predict_test)

                results[binary_string][column_names[ind * 4]] = acc_train
                results[binary_string][column_names[ind * 4 + 1]] = acc_test
                results[binary_string][column_names[ind * 4 + 2]] = balanced_acc_train
                results[binary_string][column_names[ind * 4 + 3]] = balanced_acc_test

                if acc_train > best_train:
                    best_train = acc_train
                    print("New best_train for ", column_names[ind * 4], " ", acc_train)

                if acc_test > best_test:
                    best_test = acc_test
                    print("New best_test for ", column_names[ind * 4 + 1], " ", acc_test)

                if balanced_acc_train > balanced_best_train:
                    balanced_best_train = balanced_acc_train
                    print("New balanced_best_train for ", column_names[ind * 4 + 2], " ", balanced_acc_train)

                if balanced_acc_test > balanced_best_test:
                    balanced_best_test = balanced_acc_test
                    print("New balanced_best_test for ", column_names[ind * 4 + 3], " ", balanced_acc_test)

                doneIters += 1
                if (doneIters % 20 == 0):
                    df = pd.DataFrame(data=results, index=column_names).T
                    df.to_csv('multi_value_vector_part2_all.csv')

    df = pd.DataFrame(data=results, index=column_names).T
    df.to_csv('multi_value_vector_part2_all.csv', mode='a', header=False)


def crossValidation(X,Y,output_dir, experiment2_mode, start_index = 0,n_splits = 10, step = 10):
    skf = StratifiedKFold(n_splits=n_splits)

    base_column_names = ["avg_acc","std_acc","max_acc","avg_bal_acc","std_bal_acc","max_bal_acc"]
    modes = ["_train", "_test"]

    tuples = getExperiment2Tuples(experiment2_mode)
    tuple_columns = getExperiment2OptimizedParams(experiment2_mode)

    column_names = tuple_columns + base_column_names

    train_indices, test_indices = getSplits(skf,X,Y)
    print("Number of combinations: ",len(tuples))
    #for (model, model_name) in zip(models,model_names):
    for index in range(start_index,len(tuples),step):
        first_local_index = index
        last_local_index = min(index + step, len(tuples))
        print(first_local_index, last_local_index)

        results = {}
        results_helper = {}
        for index2 in range(first_local_index,last_local_index):
            print(tuples[index2])
            for mode in modes:
                results[str(index2) + mode] = {}
                results_helper[str(index2)+ mode] = {
                    "acc": [],
                    "bal_acc": []
                }

        start_time = time.time()

        iteration = 0
        for (train_index, test_index) in zip(train_indices,test_indices):
            print(iteration)
            for index2 in range(first_local_index, last_local_index):
                tup = tuples[index2]

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                model = getInitialModel(experiment2_mode, tup)

                model.fit(X_train, y_train)
                y_predict_train = model.predict(X_train)
                y_predict_test = model.predict(X_test)

                acc_train = accuracy_score(y_train, y_predict_train)
                acc_test = accuracy_score(y_test, y_predict_test)
                balanced_acc_train = balanced_accuracy_score(y_train, y_predict_train)
                balanced_acc_test = balanced_accuracy_score(y_test, y_predict_test)

                results_helper[(str(index2)+"_train")]["acc"].append(acc_train)
                results_helper[(str(index2)+ "_train")]["bal_acc"].append(balanced_acc_train)
                results_helper[(str(index2) + "_test")]["acc"].append(acc_test)
                results_helper[(str(index2) + "_test")]["bal_acc"].append(balanced_acc_test)

            iteration += 1

        end_time = time.time()
        print((last_local_index - first_local_index)," models trained and evaluated in ",end_time-start_time," seconds")

        for index2 in range(first_local_index, last_local_index):
            tup = tuples[index2]
            shouldDelete = False
            for mode in modes:
                key = str(index2) + mode
                for (elem, name) in zip(tup, tuple_columns):
                    results[key][name] = elem
                if(len(results_helper[key]["acc"])> 0 ):
                    results[key]["avg_acc"] = np.mean(results_helper[key]["acc"])
                    results[key]["std_acc"] = np.std(results_helper[key]["acc"])
                    results[key]["max_acc"] = np.max(results_helper[key]["acc"])
                else:
                    print("error: ",key,results_helper[key])
                    shouldDelete = True

                if (len(results_helper[key]["bal_acc"]) > 0):
                    results[key]["avg_bal_acc"] = np.mean(results_helper[key]["bal_acc"])
                    results[key]["std_bal_acc"] = np.std(results_helper[key]["bal_acc"])
                    results[key]["max_bal_acc"] = np.max(results_helper[key]["bal_acc"])
                else:
                    print("error: ",key,results_helper[key])
                    shouldDelete = True

                if shouldDelete == True:
                    print("deleted: ",key)
                    del results[key]

        df = pd.DataFrame(data=results, index=column_names).T
        if (index != 0):
            df.to_csv(output_dir, mode='a', header=False)
        else:
            df.to_csv(output_dir, mode='w', header=True)

def main():
    if(len(sys.argv) > 2):
        mode = int(sys.argv[1])
        input_dataset_path = sys.argv[2]
    else:
        exit(0)
    #analyseFile(input_dataset_path)
    X, Y = parseCSV(input_dataset_path, binary_value=False)

    if mode == 1:
        experimentVariousAttributes(X, Y)

    elif mode == 2:
        input_mode = sys.argv[3]
        start_index = int(sys.argv[4])
        step = int(sys.argv[5])
        n_splits = int(sys.argv[6])
        crossValidation(X,Y,output_dir = 'sprawko_data/experiment_part2_'+input_mode+'.csv',experiment2_mode=input_mode,
                         start_index=start_index, step = step , n_splits=n_splits)
    elif mode == 3:
        output_dir = sys.argv[2]
        trainAndSaveFinalModels(X, Y, output_dir)

    elif mode == 4:
        input_model_path = sys.argv[3]
        loadAndTestModel(X,Y,input_model_path)

    elif mode == 5:
        testMajorityRuleMethod(Y)

    elif mode == 6:
        n_repeats = int(sys.argv[3])
        testRandomMethod(Y,n_repeats)

    return 0

if __name__ == "__main__":
    main()
