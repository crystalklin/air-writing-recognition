# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#pts = deque(maxlen=args["buffer"])
pts = deque(maxlen=512)
# pts = []

# Blackboard to overlay vector on?
blackboard = np.zeros((480,640,3), dtype=np.uint8)

# 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

point_index = 1

# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    '''
    Leaving as is for now and will maybe update if results aren't good. 
    '''
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)
        # pts.append(center)
    elif len(cnts) == 0:
        if len(pts) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Finding contours on the blackboard
            print ("THRESH")
            print (thresh1)
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) > 0:
                cnt = sorted(blackboard_cnts, key = cv2.contourArea(), reverse = True)[0]
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    character = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                    newImage = cv2.resize(alphabet, (28, 28))
                    newImage = np.array(newImage)
                    newImage = newImage.astype('float32')/255
                    
            # make predictions here
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), 2)
        cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 8)

    #for i in range(point_index, len(pts)):
    #    # if either of the tracked points are None, ignore
    #    # them
    #    if pts[i - 1] is None or pts[i] is None:
    #        continue
    #            
    #    # print ("frame: ", pts[i])
    #    # otherwise, compute the thickness of the line and
    #    # draw the connecting lines
    #    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    # print ("printing frame")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

#print (pts)

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
