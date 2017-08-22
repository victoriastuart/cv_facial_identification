#!/usr/bin/env python2
# coding: utf-8

# ----------------------------------------------------------------------------
#   script: /mnt/Vancouver/apps/openface/demos/webcam_face_recognition.py
#  project: https://cmusatyalab.github.io/openface/demo-4-sphere/
#   GitHub: https://github.com/cmusatyalab/openface/blob/master/demos/sphere.py
# 3D model: http://openface-models.storage.cmusatyalab.org/nn4.small2.3d.v1.t7

# ----------------------------------------------------------------------------
# SCRIPT VERSIONS:
# ----------------
# v.0: /mnt/Vancouver/apps/openface/demos/sphere.unmodified.source.py   ## but edited to work, my computer
# v.1: /mnt/Vancouver/apps/openface/demos/sphere.py                     ## trivial mods (essent. unmod.)
# v.2: /mnt/Vancouver/apps/openface/demos/sphere.inactivated.globe.py   ## inactivated that stupid globe
# v.3: /mnt/Vancouver/apps/openface/demos/sphere_modified.py            ## extensive modif.; commented;
                                                                        ## fully working but messy (commenting; print statements ...)
# v.4: /mnt/Vancouver/apps/openface/demos/webcam_face_recognition.py    ## this file (edited, 'clean' version)

# ----------------------------------------------------------------------------
# USAGE:
# ------
# env: p2 venv [Python 2.7 venv]
# pwd: /mnt/Vancouver/apps/openface
#
# python ./demos/webcam_face_recognition.py ./generated-embeddings/classifier.pkl
# python 2>/dev/null ./demos/webcam_face_recognition.py ./generated-embeddings/classifier.pkl
#
# see also: http://cmusatyalab.github.io/openface/usage/

# ============================================================================
# REQUIRE SCRIPT TO RUN IN PYTHON 2.7 VENV
# (HAS INSTALLED, REQUIRED OPENCV, DLIB, OPENFACE, ... PACKAGES)
import os

# FIRST, SEE IF WE ARE IN A CONDA VENV { py27: PYTHON 2.7 | py35: PYTHON 3.5 | tf: TENSORFLOW | thee : THEANO }
try:
    os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
    print("\n\tPlease set the py27 { p2 | Python 2.7 } environment!\n")
    exit()

# IF WE ARE IN A CONDA VENV, REQUIRE THE p2 VENV:
if os.environ['CONDA_DEFAULT_ENV'] != "py27":
    print("\n\tPlease set the py27 { p2 | Python 2.7 } environment!\n")
    exit()
# [ ... SNIP! ... ]
# more here: /home/victoria/GeanyProjects/Victoria/reference/Python%20Notes.html#Environments
# ============================================================================

import time
start = time.time()

import argparse
import cv2
import os
import dlib
# ----------------------------------------------------------------------------
import cPickle as pickle
#import pickle                       ## for the face classifier (classifier.pkl), passed on the command-line
from sklearn.mixture import GMM     ## also added, from classifier_webcam.py, for facial recognition
                                    ## GMM: Gaussian Mixture Model Ellipsoids
                                    ## http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
                                    ## http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
# ----------------------------------------------------------------------------

import numpy as np
np.set_printoptions(precision=2)    ## how floating point numbers, arrays,  other NumPy objects are displayed
                                    ## https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
                                    ## http://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array

import openface
from matplotlib import cm           ## cm: colormap; needed for bounding boxes, ...

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')


# ----------------------------------------------------------------------------
# FIND FACES IN EACH FRAME [getRep() METHOD]:
"""Via the verbose output this subsection appears to be the 'bottleneck' in this
script -- specifically the "Face detection" step:

    bb = align.getAllFaceBoundingBoxes(rgbImg   ## Face detection took 0.176... seconds

I don't think I can optimize this:

    HOW LONG DOES PROCESSING A FACE TAKE?
    https://news.encinc.com/2015/10/14/Show-HN%3A-OpenFace--Face-recognition-with-Google's-FaceNet-deep-neural-network-44971

    The processing time depends on the size of your image for face detection
    and alignment. These only run on the CPU and take from 100-200ms to over
    a second. The neural network uses a fixed-size input and has a more
    consistent runtime. Averaging over 500 forward passes of random input, the
    latency is 77.47 ms ± 50.69 ms on our 3.70 GHz CPU and 21.13 ms ± 6.15 ms
    on our Tesla K40 GPU, obtained with util/profile-network.lua

A quick Google search { optimize opencv openface dlib face predictor |
                        fastest face classifier |
                        fastest dlib face classifier }

reveals several useful articles, but these mainly involve C++ modifications /
recompiled code. This gives a pretty good overview:

    Speeding up Dlib’s Facial Landmark Detector | Learn OpenCV
    http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
"""

def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get all bounding boxes:
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    ## "align" is defined in __main__, below:
    ##      align = openface.AlignDlib(args.dlibFacePredictor)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(align.align(args.imgDim, rgbImg, box,
                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    #print('\nreps:\n', reps)
    return reps

# ============================================================================
# FACIAL PERSONAL IDENTIFICATION -- COPIED OVER FROM classifier_webcam.py
# /mnt/Vancouver/apps/openface/demos/classifier_webcam.py

"""Victoria: this method uses the facial identification classifier,

    /mnt/Vancouver/apps/openface/generated-embeddings/classifier.pkl

that I trained on my, Carmine's and Dani's images in my earlier facial
personal identification in images work.  See (e.g.)

    /home/victoria/projects/nb-cv.html#Face%20Detection%20[OpenFace]

in my CV notebook (view that URL in Firefox). Basically (here), each webcam
frame is an 'image,' so that method (below) works!  :-)
"""

# ----------------------------------------------------------------------------
# FACIAL RECOGNITION - PERSONAL IDENTIFICATION SUBSECTION [METHOD: infer()]
"""Via the verbose output and print statements (below), this subsection processes
images almost instantly.
"""

def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)          ## le : label; clf : classifier
    # ----------------------------------------------------------------------------
    # http://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram
    # can replace two lines above with the following, but no change (increase) in speed:
    #import gc
    #f = open(args.classifierModel, "r")
    ## disable garbage collector
    #gc.disable()
    #(le, clf) = pickle.load(f)
    ## enable garbage collector again
    #gc.enable()
    #f.close()
    # ----------------------------------------------------------------------------
    reps = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print("No Face detected")
            return (None, None)

        start = time.time()

        predictions = clf.predict_proba(rep).ravel()
        #print("predictions:", predictions)
        #print("predictions operation took {} seconds.".format(time.time() - start))
        ##      predictions operation took 5.48362731934e-05 seconds.
        #start = time.time()

        maxI = np.argmax(predictions)
        #print('maxI', maxI)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        #print("argmax(predictions) operation took {:0.6f} seconds.".format(time.time() - start))
        #print("argmax(predictions) operation took {} seconds.".format(time.time() - start))
        ##      argmax(predictions) operation took 3.09944152832e-06 seconds.
        #start = time.time()

        persons.append(le.inverse_transform(maxI))
        # print(str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
        # (prints the second prediction)
        #print("persons argmax label append operation took {} seconds.".format(time.time() - start))
        ##      persons argmax label append operation took 4.00543212891e-05 seconds.
        #start = time.time()

        #confidences.append(predictions[maxI])
        confidences.append(round(predictions[maxI], 4))    ### VICTORIA: ROUNDS THE PRECISION OF THE OUTPUT IN TERMINAL, VIDEO ###
        #print("confidences (rounded to 4 decimals) label append operation took {} seconds.".format(time.time() - start))
        ##      confidences (rounded to 4 decimals) label append operation took 3.09944152832e-06 seconds.
        ##start = time.time()

        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    #persons.reverse()              ## IF needed, can reverse these to get the P, C output
    #confidences.reverse()          ## order to better match the persons in webcam video stream
    return (persons, confidences)
# ============================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ADD DLIB MODEL (FACE PREDICTOR):
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

    # ADD OPENFACE MODEL (TORCH7 CLASSIFIER):
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        #default='nn4.small2.3d.v1.t7')
        #default='/mnt/Vancouver/apps/openface/models/openface/nn4.small2.3d.v1.t7')    ## << works
        default=os.path.join(
            modelDir,
            #"openface/nn4.small2.3d.v1.t7"))       ## [default] FAST, 3-dimensional facial embeddings
            "openface/nn4.small2.v1.t7"))           ## see my docstring, below ...
    # downloaded 3D model from: http://openface-models.storage.cmusatyalab.org/nn4.small2.3d.v1.t7
    # saved at: /mnt/Vancouver/apps/openface/models/openface/nn4.small2.3d.v1.t7
    """nn4.small2.3d.v1.t7:

    As explained here (https://cmusatyalab.github.io/openface/demo-4-sphere/):

        "For a brief intro to OpenFace, we provide face recognition with a deep neural network that embed
        faces on a sphere. (See our tech report for a more detailed intro to how OpenFace works.) Faces
        are often embedded onto a 128-dimensional sphere. For this demo, we re-trained a neural network
        to embed faces onto a 3-dimensional sphere that we show in real-time on top of a camera feed.
        The 3-dimensional embedding doesn't have the same accuracy as the 128-dimensional embedding, but
        it's sufficient to illustrate how the embedding space distinguishes between different people."

    So, the "3d" in "nn4.small2.3d.v1.t7" are 3-dimensional embeddings (this is why this unmodified script
    was so FAST!!).

    I will need to use this model (that I trained on): nn4.small2.v1.t7
    """

    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)

    parser.add_argument('--captureDevice', type=int, default=0, help='Capture device (0:laptop webcam; 1: usb webcam)')
    # Victoria: even though I a using my webcam connected via USB, it is recognized as the default ("0") device

    default_width = 768
    default_height = 432
    default_scale = 0.25

    #parser.add_argument('--width', type=int, default=1280)             ## default
    ##parser.add_argument('--height', type=int, default=800)            ## default
    #parser.add_argument('--height', type=int, default=720)             ## gives divisible-by-16 1280 x 720 aspect ratio

    #parser.add_argument('--scale', type=int, default=0.25)             ## default = 0.25; used to scale bbs (bounding boxes);
                                                                        ## needed to increase to 0.40 for smaller [854 x 400] aspect ratio
    parser.add_argument('--scale', type=int, default=default_scale)

    parser.add_argument('--width', type=int, default=default_width)
    parser.add_argument('--height', type=int, default=default_height)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    # ----------------------------------------------------------------------------
    # ADD FACIAL RECOGNITION CLASSIFIER (from classifier_webcam.py):

    parser.add_argument('classifierModel', type=str,
    help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    # Will need now this script) to pass "./generated-embeddings/classifier.pkl"
    # as a command-line argument; see "Usage" at top of script.
    # ----------------------------------------------------------------------------

    # FINALLY, ADD PROCESS (ADD) THOSE ARGUMENTS:
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # SET IP WEBCAM CAPTURE:
    # this is from the OpenFace source code (with the "globe" crap parsed out):
    # /mnt/Vancouver/apps/openface/demos/sphere.unmodified.source.py

    align = openface.AlignDlib(args.dlibFacePredictor)

    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be USB cam.   ## my USB webcam = 0
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    ## TO TIME THIS SCRIPT (e.g. to count warnings, 1/frame, to calc. FPS):
    #import time        ## already imported at start of script, but shown to indicate dependency
    #max_time = 30
    #start_time = time.time()
    #while (time.time() - start_time) < max_time:
    while True:
        ret, frame = video_capture.read()

        # ----------------------------------------------------------------------------
        # IDENTIFY PERSONS (INDIVIDUALS) IN THE BOUNDED (RECOGNIZED) FACES!
        # added from classifier_webcam.py :
        persons, confidences = infer(frame, args)       ## calls infer() method, above
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= args.threshold:         ## 0.5 : face detection threshold
                persons[i] = "_unknown"
        # ----------------------------------------------------------------------------

        #frame = cv2.flip(frame, 1)     ## was in source code but flips horizontally ['idiotic', here]
                                        ## 0: flips vertically; 1: flips horizontally

        frameSmall = cv2.resize(frame, (int(args.width * args.scale),
                                        int(args.height * args.scale)))

        bbs = align.getAllFaceBoundingBoxes(frameSmall)

        pts, clrs = [], []

        # Victoria -- added list of accepted frame/labels colors:
        import itertools
        colors = itertools.cycle([(0, 255, 0), (191, 0, 255), (255, 255, 0), (0, 64, 255), (0, 128, 255)])
        ##   (0, 255, 0) : green
        ## (191, 0, 255) : magenta
        ## (255, 255, 0) : cyan
        ##  (0, 64, 255) : orange-red
        ## (0, 128, 255) : orange
        ## color picker (rgb): http://www.w3schools.com/colors/colors_picker.asp
        ## note: OpenFace, OpenCV etc. use the reverse order: bgr !

        """VICTORIA: the following FOR LOOP iterates over frames, finding faces and identifying persons (within the
        limits of the trained model!  ;-) in the bounding boxes (bb).

        IDENTIFYING THE PEOPLE IN THE IDENTIFIED FACES.

        To accomplish this, I added code from the OpenFace "classifier_webcam" script to the FOR LOOP, below. The
        unmodified code, which found faces (only), was blazingly fast on my CPU. With a [768 x 432] webcam aspect
        ratio setting, this script runs at ~278 FPS (833 frames/30 sec) (no GPU). Adding the facial identification
        code dramatically slowed the FPS (~2.1 FPS), as it needs now to also run the Torch7 lua script (classifier).

        LABELING THE bb (FACES) WITH THEIR IDENTITIES.

        From my print statements analyses (see an earlier commit, if absent here), the bounding boxes for all faces
        identified in the frame are returned "en masse", per frame iteration, as tuples of coordinates; e.g. "tl"
        (top left) coordinates for a 2-person frame might be (452, 340).  Thus, in my modification where I want to
        SEPARATELY color and ALSO label (persons identities), I need to iterate over found all the faces found in
        each frame.  I accomplished this by first appending those tuples to lists -- see this article on why needed:
        http://stackoverflow.com/questions/10867882/tuple-unpacking-in-for-loops -- then iterating over that those lists
        (that renew/refreshed each frame).

        I did that by adding a small "for item in tl_list:" FOR LOOP, that (cleverly, I thought) used the parent for
        loop's "i" counter to label the individual person's identity using that counter, i, to retrieve the current
        person (id)/confidence via indexing: persons[i], confidences[i].

        COLORING THE BB, LABELS.

        This was actually the hardest part. The unmodified script automatically colored the bb by finding the center
        of the bb, mapping that value to matplotlib's "Set1" colormap [cm.Set1(); links below or in earlier
        commit of this file], multiplying the first 3 [0:3] elements by 255 and saving those as a color triplet in
        list form.  In my tests (pic of Carm & I), there are 2 faces in each frame; these are thus separately colored.

        In my first attempt, I managed to color the bb from a randomly-selected color triple/tuple in a short list,
        which 'worked' fine, but was very distracting (esp. on my non-GPU, slow FPS system).  The colors, of course,
        changed on each frame, and the consistency of the frame:label color matching was poor.

        Solution!!  In my second attempt , I solved this by using Python's itertools package to iterate
        over my colors list.  That approach allowed the script to cycle through the list, as/if needed! ;-)  I
        initially placed that iterator in the FOR LOOP, below, but I couldn't get differently-colored frames/labels.
        The solution was to place it (above) OUTSIDE the loop, and calling the iterations [colors.next()] in the for
        loop, below.  That way, the first color is selected outside the loop, then inside the loop the additional
        color(s) is called, as needed!  :-D

        As an aside, at times a mismatch or missing "i" between the parent and embedded FOR LOOPs (e.g.,
        if i > # of faces), the script would crash!  The solution was to add a "try/except" conditional statement;
        if there was an error (i > faces; ...) then then that embedded FOR LOOP just "passed" (graceful exit).  :-)
        """
        for i, bb in enumerate(bbs):

            alignedFace = align.align(96, frameSmall, bb,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

            rep = net.forward(alignedFace)

            center = bb.center()

            color_cv = colors.next()        ## gets next frame/labels color (first obtained
                                            ## outside/above this loop), via itertools

            # ----------------------------------------
            # GET SOME BB RECTANGLE APEX COORDINATES:

            bl = (int(bb.left() / args.scale), int(bb.bottom() / args.scale))   ## bottom left;  tuple: e.g. (352, 325)
            tr = (int(bb.right() / args.scale), int(bb.top() / args.scale))     ## top right; tuple, e.g.: (452, 340))
            #print('bl:', bl)

            # 'print' the frame (rectangle i.e. bb):
            cv2.rectangle(frame, bl, tr, color=color_cv, thickness=3)

            # ----------------------------------------
            # VICTORIA -- ADDED, FOR LABEL PLACEMENT:

            tl = ( int(bb.left() / args.scale), (int(bb.top() / args.scale)) - 15 )      ## top left; tuple, e.g.: (452, 340))
            # tl is a tuple, but we need to loop over it -- hence, we convert it to a list:
            tl_list = [tl]
            #print('tl', tl)                     ## tuple, e.g.: (365, 222)
            #print('tl_list', tl_list)           ## list with a tuple, e.g.: [(365, 222)]

            # ----------------------------------------
            # ADD/PLACE THE LABELS:

            #for item in tl:                    ## IF I loop over the TUPLE, I get colored bb but no labels! 'Explanation' here:
                                                ## http://stackoverflow.com/questions/10867882/tuple-unpacking-in-for-loops
            for item in tl_list:
                #print('i, item:', i, item)     ## why is there an "i"? As noted above, this section of code is in a FOR LOOP ...
                                                ## Good! We will make use of it ("i") here, in this embedded FOR LOOP!

                font_size = 0.75        ## float
                font_thickness = 2      ## int

                try:
                    # ----------------------------------------------------------------------------
                    # NOTE: moved the following further down (so as to place it on TOP of the overlay ...)
                    #cv2.putText(frame, "{} {}".format(persons[i], confidences[i]), item, cv2.FONT_HERSHEY_SIMPLEX, font_size, color_cv, font_thickness)
                    # ----------------------------------------------------------------------------
                    # define the rectangle that will frame/provide a background color for the bb labels:
                    text = "{} {}".format(persons[i], confidences[i])
                    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html#getTextSize

                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
                    #print('text_size:', text_size)      ## prints, e.g.: 'text_size:', ((302, 28), 13)    ## ((w, h), baseline); 28 + 13 = 41
                                                         ## text_size[0][0] = 302; text_size[0][1] = 28; text_size[1] = 13

                    # the use of 'text_size' allows us to auto-adjust the label background for differences in label text size, text field widths!
                    tl_label = ((int(bb.left() / args.scale))-5, (int(bb.top() / args.scale)) - (text_size[0][1] + text_size[1] + 10))
                    br_label = ((int(bb.left() / args.scale) + text_size[0][0] + 5), (int(bb.top() / args.scale)) - 5 )
                    ## tl_label : label top left; (x, y) tuple, e.g.: (452, 340)); left-shifted by 5, elevated by 50
                    ## br_label : label bottom right; width 325, elevated by 5
                    # ----------------------------------------------------------------------------
                    """OPENCV -- TRANSPARENT OVERLAYS:

                    The procedure is to first copy the image (i.e: frame); that copy will be superimposed over
                    the source image (frame) as the background for the text box; place the rectangle in that
                    overlay; set the opacity; place that overlay/text background over the image (frame).

                    See:
                        http://bistr-o-mathik.org/2012/06/13/simple-transparency-in-opencv/
                        http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
                    """
                    overlay = frame.copy()
                    #cv2.rectangle(overlay, tl_label, br_label, color_cv, -1)            ## frame color
                    #cv2.rectangle(overlay, tl_label, br_label, (200,200,200), -1)       ## light gray
                    #cv2.rectangle(overlay, tl_label, br_label, (152,152,152), -1)       ## darker gray
                    cv2.rectangle(overlay, tl_label, br_label, (152,152,152), -1)        ## black (better: set shade via opacity)
                    opacity = 0.75
                    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
                    # OpenCV puttext: http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#puttext
                    cv2.putText(frame, "{} {}".format(persons[i], confidences[i]), item, cv2.FONT_HERSHEY_SIMPLEX, font_size, color_cv, font_thickness)
                    ###cv2.imshow(frame)        ## << although generally required, not needed,  here
                    # ----------------------------------------------------------------------------
                except:
                    pass
        ## END OF "for i, bb in enumerate(bbs):" FOR LOOP

        # ----------------------------------------------------------------------------
        # PRINT THE PERSON'S NAME AND CONFIDENCE AT THE UPPER LEFT PORTION OF THE FRAME:
        # note that the precision (terminal/frames) is rounded to 4 decimals at line ~171, above:
        #     confidences.append(round(predictions[maxI], 4))
        # re-declared these variables, in case I want to change them:
        font_size = 0.75        ## float
        font_thickness = 2      ## int
        # (255, 255, 0) : cyan
        # (10, 25) = (x, y) screen coordinates
        cv2.putText(frame, "P: {}".format(persons), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), font_thickness)
        cv2.putText(frame, "C: {}".format(confidences), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), font_thickness)
        # ----------------------------------------------------------------------------

        cv2.imshow("Facial Recognition c. Personal ID", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # CLEAN UP:
    video_capture.release()
    cv2.destroyAllWindows()

# ============================================================================
