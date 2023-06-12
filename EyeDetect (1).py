
# Capstone
# TODO:
# TASK										- STATUS: ADDED/TESTED
# Add second alarm (non-consequative alarm) - ADDED TESTED
# Add Speaker -> GPIO						- ADDED TESTED
# Add Calibration 
# -> beep, wait 5 open,
# -> beep, wait 5 closed,
# -> use average ratios						- ADDED TESTED
# Add Head Tilt
# -> Find dots for head
# -> Find horinzontal angle
# -> Same flag variables					- set alarm for roll tilt, get vertical tilt ready for expo with calibration
# -> Set different debug alarm				- tilt works, add alarm for 3 frames of tilt, test with sunglasses
# Modify variables and import order			- LATER
# Add time sensitive frame measurement
# -> Get system time, compare each frame
# -> Count by time instead of frames		- use datatime.now
# Start script on startup					- DO ON PI
# Save sample recording						- 
# Modify debug statements					- 
## Extra
# Calibration restarts if bad values 
# -> check if eyes_open > eyes_closed
#    by >0.08 distance apart or smth		-
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

# Debugging
debug = False # Must be True or False

# Beep for a short time, not system
def beep():
	print("-BEEP-")
	GPIO.output(buzzer,GPIO.HIGH)
	time.sleep(0.05)
	GPIO.output(buzzer,GPIO.LOW)
	time.sleep(0.05)
	
# Alarm sound on
def alarmOn():
	GPIO.output(buzzer,GPIO.HIGH)
	print("+ALARM+")

# Alarm sound off
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
	angle = np.degrees(np.arctan2(dY, dX))
	print ("an: ", angle, " tan: ", np.arctan2(dY, dX))
	# Lower bound is~ -125, and~ -205, += 25 at most
	# set on 0, +-25 at most
	return angle

# Threshold
# ~0.22 for Fima
# ~0.29 for Moe
# Average: 0.339 for open and 0.141 for closed -> 0.24 median
# source: https://www.hrpub.org/download/20191230/UJEEEB9-14990984.pdf
EYE_AR_THRESH = 0.29
CONSECUTIVE_ALARM_SENS = 4 	# Frames for consecutive alarm
DROWSINESS_ALARM_SENS = 12 	# Frames for general alarm, will decrement by 1 when open
TILT_ALARM_SENSE = 4 		# Frames for tilt alarm
dynamic_ear_thresh = 0

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
frames_run = 0	# For calibration
ear_open = 0
ear_closed = 0

### Main
while True:
	# Read in the video strean
	frame = vs.read()
	#frame = vs.read()

	# Compress the video stream
	frame = imutils.resize(frame, width=400)
	frame = cv2.flip(frame, -1) #1 for vertical, -1 for horizontal and vertical
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
		# Average EAR
		ear = (leftEAR + rightEAR) / 2.0

		# Output visual representation
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		# Initial Calibration
		# Wait first 6 frames to ramp up to speed and cache program
		# Calibrate open eyes for 12 frames, ignoring the first 2
		# Calibrate open eyes for 12 frames, ignoring the first 2
		# Frame 5 for beep
		# Frame [7, 16] = 10 frames for open
		# Frame 17 for beep
		# Frame [19, 29] = 10 frames for open
		if (frames_run < 30):
			flag = 0
			long_flag = 0
			if (frames_run == 5):
				beep();
			elif (frames_run > 6 and frames_run < 17):
				cv2.putText(frame, "OPEN eyes", (270, 290),
					cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
				ear_open += ear
			elif (frames_run == 17):
				beep();
			elif (frames_run > 18 and frames_run < 29):
				cv2.putText(frame, "CLOSE eyes", (270, 290),
					cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
				ear_closed += ear
			# EAR threshold is the calibrated EAR for the driver
			elif (frames_run == 29):
				beep();
				beep();
				dynamic_ear_thresh = (ear_open + ear_closed) /20
				print ("--------")
				print (dynamic_ear_thresh)
				print ("--------")
				
		frames_run += 1
		
		
		print ("EAR: ", ear)

		if (debug):
			# Debug for EAR
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (280, 30),
				cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "TILT: {:.2f}".format(tilt), (0, 290),
				cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
			
		# If eye is closed, compare against calibrated EAR
		if ear < dynamic_ear_thresh:
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
			
		# Alarm triggered for asleep OR for drowsy
		if (flag > CONSECUTIVE_ALARM_SENS or long_flag > DROWSINESS_ALARM_SENS):
			alarmOn() 
			# Assume alarm is effective, reset long_flag
			long_flag = 0
			if (flag <= CONSECUTIVE_ALARM_SENS):
				flag = CONSECUTIVE_ALARM_SENS + 1
			# Debug for alarm
			if (debug):
				cv2.putText(frame, "ALARM: {:d} and {:d}".format(flag, long_flag), (0, 30),
					cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
			else:
				cv2.putText(frame, "KEEP EYES OPEN", (0, 30),
					cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
		# Else turn off the alarm
		else:
			# This can also be handled automatically with threads
			alarmOff ()
			
	cv2.imshow("Eye of Gatsby", frame)
 
 	# Allow exiting
	if cv2.waitKey(1) & 0xFF == ord("s"):
		break

cv2.destroyAllWindows()
vs.stop()
GPIO.cleanup() # cleanup all GPIO 
print ("Shutting down, thank you for driving safely.")
