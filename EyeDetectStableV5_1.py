
# Capstone
# TODO:
# TASK										- STATUS: ADDED/TESTED
# Add second alarm (non-consequative alarm) - ADDED TESTED
# Add Speaker -> GPIO						- ADDED TESTED
# Add Calibration 
# -> beep, wait 5 open,
# -> beep, wait 5 closed,
# -> use average ratios						- todo
# Add Head Tilt
# -> Find dots for head
# -> Find horinzontal angle
# -> Same flag variable
# -> Set different debug alarm				- added function for tilt	
# Modify variables and import order			- import order modified
# Add time sensitive frame measurement
# -> Get system time, compare each frame
# -> Count by time instead of frames		- use datatime.now
## https://docs.python.org/3/library/datetime.html
## https://stackoverflow.com/questions/1905403/python-timemilli-seconds-calculation/1905423#1905423

from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
from imutils.video import VideoStream
#from gpiozero import Buzzer

import cv2
import time
import dlib
import numpy as np
import RPi.GPIO as GPIO
import argparse
import imutils
import sys

# Alarm sound
def alarmOn():
	GPIO.output(buzzer,GPIO.HIGH)
	print("+ALARM+")

# Alarm sound
def alarmOff():
	GPIO.output(buzzer,GPIO.LOW)
	print("-ALARM-")

# EAR computation
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 
# Face angle calculator, using the sides of the face
def tilt_angle(leftContour, rightContour):
	# Find center coordinate for eye
	leftContourCenter = leftContour.mean(axis=0).astype("int")
	rightContourCenter = rightContour.mean(axis=0).astype("int")
	# Find angle between eyes
	dY = rightContourCenter[1] - leftContourCenter[1]
	dX = rightContourCenter[0] - leftContourCenter[0]
	print ("LC: ", leftContourCenter, " RC: ", rightContourCenter)
	print ("dY: ", dY, " dX: ", dX)
	angle = np.degrees(np.arctan2(dY, dX)) - 180
	print ("an: ", angle, " tan: ", np.arctan2(dY, dX))
	return angle

# Threshold
# ~0.22 for Fima
# ~0.29 for Moe
# Average: 0.339 for open and 0.141 for closed -> 0.24 median
# source: https://www.hrpub.org/download/20191230/UJEEEB9-14990984.pdf
EYE_AR_THRESH = 0.29
CONSECUTIVE_ALARM_SENS = 4 # Frames for consecutive alarm
DROWSINESS_ALARM_SENS = 12 # Frames for general alarm, will decrement by 1 when open

# Initialize camera
vs = VideoStream(src=0).start()	

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
buzzer = 26
GPIO.setup(buzzer,GPIO.OUT)
# Note which pin is being used, refer to
# https://projects.raspberrypi.org/en/projects/physical-computing/1
# https://learn.sparkfun.com/tutorials/raspberry-gpio/all

# Initialize the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Pull eyes from face detector
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize alarm variables
flag = 0 		# For prolonged eye closure
long_flag = 0 	# For eye closure over time (non-consecutive)

# Initial Calibration
#####

### Main
while True:
	# Read in the video strean
	frame = vs.read()
	#frame = vs.read()

	# Compress the video stream
	frame = imutils.resize(frame, width=400)
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(grayframe, 0)

	# Iterate over each face detected, though we only expect one
	for face in faces:
		# Scan the eyes and calculate EAR
		shape = predictor(grayframe, face)
		shape = face_utils.shape_to_np(shape) 
		
		# Extract contours for tilt
		leftContour = shape[0:3] #0,1,2 points on the face
		rightContour = shape[14:17] #14,15,16 points, reversed
		tilt = tilt_angle(leftContour, rightContour)
		
		# Extract Eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		# Output visual representation
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		print ("EAR: ", ear)

		# If eye is closed
		if ear < EYE_AR_THRESH:
			flag += 1
			long_flag += 1
			print ("FLAG: ", flag, " LONGFLAG: ", long_flag)
		# If eye is safe
		else:
			# Reset flag
			flag = 0
			long_flag -= 1
			if (long_flag < 0):
				long_flag = 0
			# Rapidly decrease flag, the threshold is ~50% of time eye opened

		# Debug for EAR
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (280, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "TILT: {:.2f}".format(tilt), (0, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# Alarm triggered for asleep OR for drowsy
		if (flag > CONSECUTIVE_ALARM_SENS or long_flag > DROWSINESS_ALARM_SENS):
			alarmOn() 
			# Assume alarm is effective, reset long_flag
			long_flag = 0
			if (flag <= CONSECUTIVE_ALARM_SENS):
				flag = CONSECUTIVE_ALARM_SENS + 1
			# Debug for alarm
			cv2.putText(frame, "ALARM: {:d}{:d}".format(flag, long_flag), (0, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# Else turn off the alarm
		else:
			# This can also be handled automatically with threads
			alarmOff ()
			
	cv2.imshow("Frame", frame)
 
 	# Allow exiting
	if cv2.waitKey(1) & 0xFF == ord("s"):
		break

cv2.destroyAllWindows()
vs.stop()
GPIO.cleanup() # cleanup all GPIO 
