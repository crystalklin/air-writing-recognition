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
import os

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

# Whiteboard to overlay vector for Display
# whiteboard = np.zeros((480,640,3), dtype=np.uint8)

# Blackboard to overlay vector for Classification Input
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

def update_depth(obj_width, focal_len, width):
    obj_depth = 0
    if width == 0:
        return obj_depth, 0
    elif obj_width == 0 or focal_len == 0:
        return obj_depth, 0
    else:
        obj_depth = obj_width * focal_len / width
        #print("Width: ", width)
        #print("Calculating depth...")
        #print("Depth: ", obj_depth, "  delta: ", obj_depth-focal_len)
    return obj_depth, obj_depth-focal_len

obj_width = 0
focal_len = 0
obj_depth = 0
count = 0
start = False

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
    width = 0

    do_save = False
    
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
            
            # recalculate
            width = 2*radius
            
            # need to update parameters from zero value
            obj_depth, delta = update_depth(obj_width, focal_len, width)

            print ("delta : ", delta)
            
            if delta > 1 or delta < -3:
                circle_status = (0, 10, 255) # too far/close, red
            elif delta > 0.5 or delta < -2:
                circle_status = (0, 255, 255) # too close, yellow
            else:
                circle_status = (10, 255, 10) # on plane, green
                do_save = True

            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), circle_status, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


        # update the points queue only if it is the correct depth
        
        print (do_save)
        # if false then empty deque so that it doesn't try to connect the last line? 
        if start and do_save:
            pts.appendleft(center)
        else:
            pts = deque(maxlen=512)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), 2)
        # EMNIST data is white characters on black
        cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 8)


    # show the frame to our screen
    mirrored_frame = frame.copy()
    mirrored_frame = cv2.flip(frame, 1)
    cv2.imshow("Frame", mirrored_frame)

    mirrored_board = blackboard.copy()
    mirrored_board = cv2.flip(blackboard, 1)
    cv2.imshow("Blackboard", mirrored_board)
    
    cv2.imwrite("blackboardimage.png", mirrored_board)
    blackboard_gray = cv2.cvtColor(mirrored_board, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("blackboard_gray_image.png", blackboard_gray)
    blur1 = cv2.medianBlur(blackboard_gray, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    cv2.imwrite("blurredimage.png", blur1)
    ret, thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Finding contours on the blackboard
    blackboard_img, blackboard_cnts, blackboard_hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite("contourblackboard.png", blackboard_img)
    if len(blackboard_cnts) > 0:
        cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
            newImage = cv2.resize(alphabet, (28, 28))
            
            path = 'input_images/'
            cv2.imwrite(os.path.join(path, "img%d.png"%count), newImage)
            
            # how to pass image as input into classification model
            #newImage = np.array(newImage)
            #newImage = newImage.astype('float32')/255

            
    
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # if the 's' key is pressed, consider this the "start size"
    if key == ord("s"):
        obj_width = width
        focal_len = 10
        start = True
        print("Tracker width: ", obj_width)
        print("Whiteboard depth: ", focal_len)

    if key == ord("d"):
        # save the image, erase blackboard, startover
        # blackboard = np.zeros((480, 640, 3), dtype=np.uint8)    
        cv2.imwrite("image.png", mirrored_blackboard)
        pass

    if key == ord("c"):
        if width == 0:
            print("Can not find tracker.")
        elif obj_width == 0 or focal_len == 0:
            print("Whiteboard not yet set.")
        else:
            obj_depth = update_depth(obj_depth, focal_len, width)


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
