# cv_facial_identification
Personal facial recognition (webcam; computer vision: OpenCV; DLib; OpenFace; ...)

My experiments (~Oct-Nov 2016) in computer vision: facial recognition / personal identification in webcam-streamed video.
I trained a classifier to recognize my girlfriend and I.

I couldn't find any code on the web for doing the bounding boxes, so I just went ahead and figured it out myself: I wrote the Python code that creates the bounding boxes around the faces, and overlays the person's name and classification probability.

Dependencies:

* Python 2.7 (venv)
* Torch7
* OpenCV
* Dlib
* Openface

Provided "as-is."  See the code for details, documentation.  I'll also upload my personal installation notes re: those libraries (Arch Linux x86_64 system; ...).  You'll need to train your own classifier, of course, on your own images, ...

I don't have a GPU, but the code ran remarkably well!  The uploaded video (screen capture) shows the final "product."
