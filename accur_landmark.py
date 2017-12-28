import os
import cv2
import glob
import random
import math
import numpy as np
import dlib

from sklearn.svm import SVC

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Set the classifier as a support vector machines with polynomial kernel
classifier = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True)


# Define function to get file list, randomly shuffle it and split 80/20
def get_files(emotion):
  files = glob.glob(os.path.join("dataset", emotion, "*"))
  random.shuffle(files)
  training = files[:int(len(files) * 0.8)]  # get first 80% of file list
  prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
  return training, prediction


def get_landmarks(image):
  landmarks = []
  detections = detector(image, 1)

  # For all detected face instances individually
  for k, d in enumerate(detections):
    shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
    xlist = []
    ylist = []

    for i in range(1, 68):
      # Store X and Y coordinates in two lists
      xlist.append(float(shape.part(i).x))
      ylist.append(float(shape.part(i).y))

    xmean = np.mean(xlist)
    ymean = np.mean(ylist)
    xcentral = [(x - xmean) for x in xlist]
    ycentral = [(y - ymean) for y in ylist]

    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
      landmarks.append(w)
      landmarks.append(z)
      meannp = np.asarray((ymean, xmean))
      coornp = np.asarray((z, w))
      dist = np.linalg.norm(coornp - meannp)
      landmarks.append(dist)
      landmarks.append((math.atan2(y, x) * 360) / (2 * math.pi))

  return landmarks


def make_sets():
  training_data = []
  training_labels = []
  prediction_data = []
  prediction_labels = []

  for emotion in emotions:
    print(" working on %s" % emotion)
    training, prediction = get_files(emotion)

    # Append data to training and prediction list, and generate labels 0-7
    for item in training:
      image = cv2.imread(item)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      clahe_image = clahe.apply(gray)
      landmarks = get_landmarks(clahe_image)

      if len(landmarks) > 0:
        # append image array to training data list
        training_data.append(landmarks)
        training_labels.append(emotions.index(emotion))
      else:
        print("no face detected on this one")

    for item in prediction:
      image = cv2.imread(item)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      clahe_image = clahe.apply(gray)
      landmarks = get_landmarks(clahe_image)

      if len(landmarks) > 0:
        prediction_data.append(landmarks)
        prediction_labels.append(emotions.index(emotion))
      else:
        print("no face detected on this one")

  return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []

for i in range(0, 10):
  print("Making sets %s" % i)  # Make sets by random sampling 80/20%
  training_data, training_labels, prediction_data, prediction_labels = make_sets()

  # Turn the training set into a numpy array for the classifier
  npar_train = np.array(training_data)
  npar_trainlabs = np.array(training_labels)

  print("training SVM linear %s" % i)  # train SVM
  classifier.fit(npar_train, training_labels)

  print("getting accuracies %s" % i)  # Use score() function to get accuracy
  npar_pred = np.array(prediction_data)
  pred_lin = classifier.score(npar_pred, prediction_labels)

  print "linear: ", pred_lin
  accur_lin.append(pred_lin)  # Store accuracy in a list

# FGet mean accuracy of the 10 runs
print("Mean value lin svm: %s" % np.mean(accur_lin))
