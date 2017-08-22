# cv_facial_identification
Personal facial recognition (webcam; computer vision: OpenCV; DLib; OpenFace; ...)

My experiments (Oct 2016) in computer vision: facial recognition / personal identification in webcam-streamed video.
I trained a classifier to recognize my girlfriend and I.

I couldn't find any code on the web for doing the bounding boxes, so I wrote the Python code that creates the bounding boxes around the faces, and overlays the person's name and classification probability.

Dependencies:

* Python 2.7 (venv)
* OpenCV
* Dlib
* Openface

Provided "as-is."  I don't have a GPU, but the code ran remarkably well!

The uploaded video (screen capture) shows the final "product."
