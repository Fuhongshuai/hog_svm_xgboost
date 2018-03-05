import time
import argparse
import os
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.externals import joblib


current_dir = os.getcwd()
data_dict = dict()
validation_dict = dict()
test_dict = dict()


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    print(predicted_Y)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


def train(epochs=100000, random_state=0,kernel='rbf', decision_function='ovr', train_model=True,test = True):

    if train_model:
        data_dict['X'] = np.load(current_dir + 'landmarks' + '.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        data_dict['X'] = np.concatenate((data_dict['X'], np.load(current_dir + 'hog' + '.npy')), axis=1)
        data_dict['Y'] = np.load(current_dir + 'labels' + '.npy')
        data = data_dict
        validation_dict['X'] = np.load(current_dir + 'landmarks' + '.npy')
        validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
        validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(current_dir + 'hog' + '.npy')), axis=1)
        validation_dict['Y'] = np.load(current_dir + 'labels' + '.npy')
        validation = validation_dict
        # Training phase
        print("building model...")
        model = SVC( random_state=random_state, max_iter=epochs, kernel=kernel,
                    decision_function_shape=decision_function)
        start_time = time.time()
        model.fit(data['X'], data['Y'])

        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))
        print("saving model...")
        joblib.dump(model, "saved_model.bin")
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.4f}".format(validation_accuracy))
    if test:
        model = joblib.load("saved_model.bin")
        test_dict['X'] = np.load(current_dir + '/landmarks_test' + '.npy')
        test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])

        test_dict['X'] = np.concatenate((test_dict['X'], np.load(current_dir + '/hog_test' + '.npy')), axis=1)
        test_dict['Y'] = np.load(current_dir + '/labels_test' + '.npy')
        test = test_dict
        test_accuracy = evaluate(model, test['X'], test['Y'])
        print("  - test accuracy = {0:.4f}".format(test_accuracy))


if __name__ == '__main__':
    train(train_model=True,test = True)