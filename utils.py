from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pickle
import os
import random
import numpy as np
'''
Experiment part 2
'''
NUM_ITERATIONS = 800
FINAL_NUM_ITERATIONS = 1500
MLP_hidden_layer_sizes = [(100,),(42,28,14,),(28,42,14,),(42,28,)]
MLP_activations = ["identity","logistic","tanh","relu"]
MLP_solvers = ["sgd","adam"]
MLP_learning_rate_inits = [0.001 , 0.002 ]

RF_criterions = ["gini","entropy"]
RF_min_samples_splits = [2,3,4,10]
RF_min_samples_leafs = [1,2,3,4]
RF_max_features = [None, "sqrt","log2"]
RF_min_impurity_decreases = [0.0 , 0.005, 0.01 , 0.1]

def testMajorityRuleMethod(Y):
    majority_score = getMajorityScore(Y)
    print("Majority rule: score = ",majority_score)
    y_predicted = np.full_like(Y,majority_score)
    acc = accuracy_score(Y, y_predicted)
    bal_acc = balanced_accuracy_score(Y, y_predicted)
    print("Majority rule: acc = ", acc, ' bal_acc = ', bal_acc)

def getMajorityScore(Y):
    score_count = {
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0,
        5 : 0
    }
    for y in Y:
        score_count[int(y)]+= 1


    listofScores = sorted(score_count.items(), reverse=True, key=lambda x: x[1])
    print(listofScores)
    return listofScores[0][0]

def testRandomMethod(Y,n_repeats):
    accs = []
    bal_accs = []
    for _ in range(n_repeats):
        y_predicted = np.asarray([random.randint(1,5) for _ in range(Y.shape[0])], dtype=np.int)
        acc = accuracy_score(Y,y_predicted)
        bal_acc = balanced_accuracy_score(Y,y_predicted)
        accs.append(acc)
        bal_accs.append(bal_acc)
    print("Number of repeats: ",n_repeats)
    print("Random: avg_acc = ", np.mean(accs), ' avg_bal_acc = ', np.mean(bal_accs))
    print("Random: max_acc = ", np.max(accs), ' max_bal_acc = ', np.max(bal_accs))

def loadAndTestModel(X,Y,model_path):
    print("Loading model...")
    model = loadModel(model_path)
    print("Predicting...")
    y_predicted = model.predict(X)
    acc = accuracy_score(Y,y_predicted)
    bal_acc = balanced_accuracy_score(Y,y_predicted)
    print("Model: acc = ", acc, ' bal_acc = ', bal_acc)

def trainAndSaveFinalModels(X,Y, output_folder):

    try:
        os.mkdir(output_folder)
    except OSError as error:
        print(error)

    mlp_model = MLPClassifier(hidden_layer_sizes=(42,28,14,),activation="logistic",
                              solver="adam",learning_rate_init=0.001,
                              max_iter=FINAL_NUM_ITERATIONS)
    rf_model = RandomForestClassifier(criterion="entropy",min_samples_split=10,min_samples_leaf=4,
                                      max_features="log2",min_impurity_decrease=0)
    print("Training MLP...")
    mlp_model.fit(X,Y)
    print("Training RF...")
    rf_model.fit(X,Y)

    print("Evaluating...")
    y_predicted_mlp = mlp_model.predict(X)
    y_predicted_rf = rf_model.predict(X)

    print("MLP: acc = ",accuracy_score(Y,y_predicted_mlp),' bal_acc = ',balanced_accuracy_score(Y,y_predicted_mlp))
    print("RF : acc = ", accuracy_score(Y, y_predicted_rf), ' bal_acc = ', balanced_accuracy_score(Y, y_predicted_rf))

    print("Saving models...")
    saveModel(os.path.join(output_folder,"mlp"),mlp_model)
    saveModel(os.path.join(output_folder, "rf"), rf_model)

def saveModel(output_dir, model):
    pickle.dump(model, open(output_dir, 'wb'))

def loadModel(input_dir):
    model = pickle.load(open(input_dir,'rb'))
    return model

def getSplits(skf,X,Y):
    train_indices = []
    test_indices = []
    for train_index, test_index in skf.split(X,Y):
        train_indices.append(train_index)
        test_indices.append(test_index)

    return train_indices,test_indices

def getInitialModel(mode, tup):
    if mode == "RF":
        return RandomForestClassifier(criterion=tup[0],min_samples_split=tup[1],min_samples_leaf=tup[2],
                                      max_features=tup[3],min_impurity_decrease=tup[4])
    elif mode == "MLP":
        return MLPClassifier(max_iter=NUM_ITERATIONS,hidden_layer_sizes=tup[0],activation=tup[1],solver=tup[2],
                             learning_rate_init=tup[3])#,tol=tup[4],n_iter_no_change=tup[5],early_stopping=tup[6])

    return None

def getExperiment2OptimizedParams(mode):
    if mode == "RF":
        return ["criterion","min_samples_split","min_samples_leaf","max_features","min_impurity_decrease"]
    elif mode == "MLP":
        return ["hid_layer_size","activation","solver","learn_rate","tol","n_iter_no_change","early_stop"]
    return None

def getExperiment2Tuples(mode):
    tuples = []
    if mode =="RF":
        for rf_crit in RF_criterions:
            for rf_min_split in RF_min_samples_splits:
                for rf_min_leaf in RF_min_samples_leafs:
                    for rf_max_feat in RF_max_features:
                        for rf_min_imp_dec in RF_min_impurity_decreases:
                            key = (rf_crit, rf_min_split, rf_min_leaf, rf_max_feat, rf_min_imp_dec)
                            tuples.append(key)
    elif mode == "MLP":
        for mlp_hidden_layer in MLP_hidden_layer_sizes:
            for mlp_activation in MLP_activations:
                for mlp_solver in MLP_solvers:
                    for mlp_learning_rate_init in MLP_learning_rate_inits:
                            '''
                        for mlp_tol in MLP_tols:
                            for mlp_n_iter in MLP_n_iter_no_changes:
                                for mlp_early_stop in MLP_early_stoppings:
                            '''
                            key = (mlp_hidden_layer,mlp_activation,mlp_solver,
                                   mlp_learning_rate_init)#,mlp_tol,mlp_n_iter,mlp_early_stop)
                            tuples.append(key)

    return tuples

def getBinaryNumber(n, final_string_length):
    binary = ""
    while(n != 0):
        binary = str(n % 2) + binary
        n = n // 2
    string_size = len(binary)
    for _ in range(final_string_length - string_size):
        binary = "0" + binary

    assert len(binary) == final_string_length

    return binary

def getColumnIndices(n, final_string_length):
    binary = getBinaryNumber(n, final_string_length)
    indices = []
    for i in range(len(binary)):
        if binary[i] == "1":
            indices.append(i)

    return indices, binary

def getStratifiedSamples(X,Y, n_splits, test_size = 0.25):
    X_train = X
    y_train = Y

    for _ in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        stratify=y_train,
                                                        test_size=test_size)

    return X_train, X_test, y_train, y_test