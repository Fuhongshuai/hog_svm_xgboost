import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2
from skimage.feature import hog


image_height = 48
image_width = 48
current_dir = os.getcwd()
img_dir = current_dir + '/test/'
predict_path = current_dir + '/shape_predictor_68_face_landmarks.dat'

images = []
labels_list = []
landmarks = []
hog_features = []
hog_images = []
nb_images_per_label = list(np.zeros(11))
predictor = dlib.shape_predictor(predict_path)

def get_landmarks(image, rects):
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


for file in os.listdir(img_dir):
    file_path = img_dir+file
    name = file.split('_')
    label = name[0]

    image = cv2.imread(file_path)
    if image is None:
        continue
    image = cv2.resize(image,(48,48))
    image = image[:,:,0]
    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualise=True)
    if features is None:
        continue
    images.append(image)
    hog_features.append(features)
    hog_images.append(hog_image)
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    face_landmarks = get_landmarks(image, face_rects)
    landmarks.append(face_landmarks)
    labels_list.append(int(label))
    nb_images_per_label[int(label)] += 1
print(labels_list)
np.save(current_dir + '/labels_test' + '.npy',labels_list)
np.save(current_dir + '/hog_test' + '.npy',hog_features)
np.save(current_dir + '/landmarks_test' + '.npy',landmarks)

