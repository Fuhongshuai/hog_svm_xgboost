import time
import argparse
import os
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb


current_dir = os.getcwd()
data_dict = dict()
validation_dict = dict()
test_dict = dict()


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    print(predicted_Y)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


def train(train_model=True,test = False):

    if train_model:
        data_dict['X'] = np.load(current_dir + '/landmarks' + '.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        data_dict['X'] = np.concatenate((data_dict['X'], np.load(current_dir + '/hog' + '.npy')), axis=1)
        data_dict['Y'] = np.load(current_dir + '/labels' + '.npy')
        data = data_dict
        validation_dict['X'] = np.load(current_dir + '/landmarks' + '.npy')
        validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
        validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(current_dir + '/hog' + '.npy')), axis=1)
        validation_dict['Y'] = np.load(current_dir + '/labels' + '.npy')
        validation = validation_dict
        # Training phase
        print("building model...")
        gbm = xgb.XGBClassifier(booster='gbtree',objective='multi:softmax',num_class=11,gamma=0.1,max_depth=6
                                ,colsample_bytree=0.7,min_child_weight=1,silent=0,eta=0.01,max_delta_step=5,
                                num_boost_round=1000)
        start_time = time.time()
        gbm.fit(data['X'], data['Y'])

        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))
        print("saving model...")
        joblib.dump(gbm, "xg_model.bin")
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.4f}".format(validation_accuracy))
    if test:
        model = joblib.load("xg_model.bin")
        test_dict['X'] = np.load(current_dir + '/landmarks_test' + '.npy')
        test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])

        test_dict['X'] = np.concatenate((test_dict['X'], np.load(current_dir + '/hog_test' + '.npy')), axis=1)
        test_dict['Y'] = np.load(current_dir + '/labels_test' + '.npy')
        test = test_dict
        test_accuracy = evaluate(model, test['X'], test['Y'])
        print("  - test accuracy = {0:.4f}".format(test_accuracy))


if __name__ == '__main__':
    train(train_model=True,test = False)