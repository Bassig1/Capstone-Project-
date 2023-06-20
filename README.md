Capstone Project 

Device detects if driver is drowsy using Eye Aspect Ratio and dlib and sounds beeper to alert driver if they are detected to be drowsy.
Below are images showing the device in use which features a calibration stage along with the actual program running. 
The main loop, responsible for keeping the driver awake if they appear drowsy, follows
the calibration phase. During this phase, the device reads in the EAR, and if the EAR is
less than the average between the EAR with eyes opened and the EAR with eyes closed
found during calibration, then the device adds to two counters for the amount spent with
eyes closed. The first counter is reset if the user opens their eyes, while the second
counter is only decremented. The first counter is when the user has their eyes closed
consecutively, and the second counter is if the user is having trouble keeping their eyes
open, though never keeping them closed long enough consecutively. If the time spent
with the eyes closed passes either timing threshold set at startup, then the alarm will go
off to alert the user awake.
![image](https://github.com/Bassig1/Capstone-Project-/assets/28423891/40638238-5e40-44bf-8097-05d891b2a83e)

![image](https://github.com/Bassig1/Capstone-Project-/assets/28423891/86e20636-317d-45b9-b7e4-d42b22940854)
