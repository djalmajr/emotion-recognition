import cv2
import glob
import os

faceDet_one = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

detect_options = {
  "scaleFactor": 1.1,
  "minNeighbors": 15,
  "minSize": (5, 5),
  "flags": cv2.CASCADE_SCALE_IMAGE
}

emotions = [
  "neutral", "anger", "contempt", "disgust", 
  "fear", "happy", "sadness", "surprise"
]

def make_dir(src):
  folder = os.sep.join(src.split(os.sep)[0:-1])

  if not os.path.exists(folder):
    os.makedirs(folder)

def detect_faces(emotion):
  # Get list of all images with emotion
  files = glob.glob(os.path.join("sorted", emotion, "*"))

  filenumber = 0

  for f in files:
    frame = cv2.imread(f)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect face using 4 different classifiers
    face_one = faceDet_one.detectMultiScale(gray, **detect_options)
    face_two = faceDet_two.detectMultiScale(gray, **detect_options)
    face_three = faceDet_three.detectMultiScale(gray, **detect_options)
    face_four = faceDet_four.detectMultiScale(gray, **detect_options)

    # Go over detected faces, stop at first detected face
    if len(face_one) == 1:
      facefeatures = face_one
    elif len(face_two) == 1:
      facefeatures = face_two
    elif len(face_three) == 1:
      facefeatures = face_three
    elif len(face_four) == 1:
      facefeatures = face_four
    else:
      facefeatures = []

    # Get coordinates and size of rectangle containing face
    for (x, y, w, h) in facefeatures:  
      print "face found in file: %s" % f

      # Cut the frame to size
      gray = gray[y:y + h, x:x + w]  

      try:
        # Resize face so all images have same size
        out = cv2.resize(gray, (350, 350))
        filename = os.path.join("dataset", emotion, "%s.jpg" % filenumber)
        make_dir(filename)
        cv2.imwrite(filename, out)
      except Exception:
        print "error"
        pass

    filenumber += 1

for emotion in emotions:
  detect_faces(emotion)
