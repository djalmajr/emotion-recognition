import glob
import os
from shutil import copyfile

def make_dir(src):
  folder = os.sep.join(src.split(os.sep)[0:-1])

  if not os.path.exists(folder):
    os.makedirs(folder)

def copy_img(src, dst):
  make_dir(dst)
  copyfile(src, dst)

# Define emotion order
emotions = [
  "neutral", "anger", "contempt", "disgust", 
  "fear", "happy", "sadness", "surprise"
]

# Returns a list of all folders with participant numbers
participants = glob.glob(os.path.join("emotions", "*"))

for x in participants:
  # Store current participant number
  part = x.split(os.sep)[1]

  # Store list of sessions for current participant
  for sessions in glob.glob(os.path.join(x, "*")):
    for files in glob.glob(os.path.join(sessions, "*")):
      session = files.split(os.sep)[2]
      file = open(files, 'r')

      # Emotions are encoded as a float, readline as float,
      # then convert to integer.
      emotion = int(float(file.readline()))

      # Get path for first and last image in sequence
      src_neutral = glob.glob(os.path.join("images", part, session, "*"))[0]
      src_emotion = glob.glob(os.path.join("images", part, session, "*"))[-1]

      # Generate path to put neutral and emotion images
      dst_neutral = os.path.join("sorted", "neutral", src_neutral.split(os.sep)[-1])
      dst_emotion = os.path.join("sorted", emotions[emotion], src_emotion.split(os.sep)[-1])

      copy_img(src_neutral, dst_neutral)
      copy_img(src_emotion, dst_emotion)
