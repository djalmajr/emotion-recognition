import os
import cv2
import dlib
import glob
import math
import numpy as np

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

det_one = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
det_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
det_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
det_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

detect_options = {
  "scaleFactor": 1.1,
  "minNeighbors": 15,
  "minSize": (5, 5),
  "flags": cv2.CASCADE_SCALE_IMAGE
}


def detect_and_resize_face(gray):
  features = []

  one = det_one.detectMultiScale(gray, **detect_options)
  two = det_two.detectMultiScale(gray, **detect_options)
  three = det_three.detectMultiScale(gray, **detect_options)
  four = det_four.detectMultiScale(gray, **detect_options)

  if len(one) == 1:
    features = one
  elif len(two) == 1:
    features = two
  elif len(three) == 1:
    features = three
  elif len(four) == 1:
    features = four

  if (len(features) > 0):
    (x, y, w, h) = features[0]
    gray = gray[y:y + h, x:x + w]

  return cv2.resize(gray, (350, 350))


def get_landmarks(image):
  img = clahe.apply(image)
  landmarks = []
  detections = detector(img, 1)

  # For all detected face instances individually
  for k, d in enumerate(detections):
    # Draw Facial Landmarks with the predictor class
    shape = predictor(img, d)
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
  fish_data = []
  fish_labels = []
  land_data = []
  land_labels = []

  for emotion in emotions:
    print("working on %s" % emotion)
    files = glob.glob(os.path.join("dataset", emotion, "*"))

    for file in files:
      gray = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
      landmarks = get_landmarks(gray)

      fish_data.append(gray)
      fish_labels.append(emotions.index(emotion))

      if len(landmarks) > 0:
        land_data.append(landmarks)
        land_labels.append(emotions.index(emotion))
      else:
        print("no landmarks detected on: ", file)

  return land_data, land_labels, fish_data, fish_labels
