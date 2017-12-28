import cv2
import imutils
import helpers
import numpy as np

from imutils.video import VideoStream
from imutils import face_utils
from sklearn.svm import SVC

fishface = cv2.face.createFisherFaceRecognizer()
# fishface = cv2.face.FisherFaceRecognizer_create()
classifier = SVC(kernel='linear', probability=True, tol=1e-3)

print("making sets...")
land_data, land_labels, fish_data, fish_labels = helpers.make_sets()

print("training SVM linear classifier...")
classifier.fit(np.array(land_data), np.array(land_labels))

print("training fisher face classifier...")
fishface.train(fish_data, np.asarray(fish_labels))

vs = VideoStream().start()

while True:
  frame = imutils.resize(vs.read(), width=400)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  rects = helpers.detector(gray, 0)
  landmarks = helpers.get_landmarks(gray)
  face = helpers.detect_and_resize_face(gray)
  thumb = cv2.cvtColor(cv2.resize(face, (100, 100)), cv2.COLOR_GRAY2RGB)

  frame[-100:, -100:] = thumb

  try:
    idx, conf = fishface.predict(face)
    cv2.putText(frame, helpers.emotions[idx], (15, 30),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  except Exception:
    pass

  try:
    idx = classifier.predict([landmarks])[0]
    cv2.putText(frame, helpers.emotions[idx], (15, 80),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  except Exception:
    pass

  # loop over the face detections
  for rect in rects:
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = helpers.predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
      cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

  # show the frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF

  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
