## Prerequisites

- [ck+ dataset](http://www.consortium.ri.cmu.edu/ckagree/) (a face database containing emotions)
- Dlib and [dlib shape predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Anaconda/Miniconda 3
- Scikit Learn
- Imutils (convenience functions to make basic image processing operations)

```bash
conda install -c conda-forge numpy opencv dlib scikit-learn
pip install imutils
```

## Usage

- Extract extended-cohn-kanade-images.zip and rename extracted folder to images
- Extract Emotion_labels.zip and rename extracted folder to emotions
- Extract shape_predictor_68_face_landmarks.dat.bz2

```bash
python sort_emotions.py
python extract_face.py
python main.py
```