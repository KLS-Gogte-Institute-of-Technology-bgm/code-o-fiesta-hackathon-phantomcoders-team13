# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from api.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import utility

from utilities.sample_predict import sample_predict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image



# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


model = tf.keras.models.load_model('.')

qrcode =  ""

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture('./video.mp4')
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html",qr=qrcode)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0
	start = round(time.time())
	done = False

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		ret, frame = cap.read()
		if frame is not None:
			frame = imutils.resize(frame, width=400)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			# grab the current timestamp and draw it on the frame
			timestamp = datetime.datetime.now()
			cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

			# if the total number of frames has reached a sufficient
			# number to construct a reasonable background model, then
			# continue to process the frame
			if total > frameCount:
				# detect motion in the image
				motion = md.detect(gray)

				# cehck to see if motion was found in the frame
				if motion is not None:
					# unpack the tuple and draw the box surrounding the
					# "motion area" on the output frame

					if round(time.time()) - start == 5 and not done:
						done = True
						val, img = cv2.imencode('.jpg', frame)
						img1 = sample_predict(img)
						ret = np.argmax(model.predict(img1))
						print(ret)
						if ret==0:
							utility.qrgen(100)
						elif ret==1:
							utility.qrgen(200)
						else:
							utility.qrgen(300)
						
						
					(thresh, (minX, minY, maxX, maxY)) = motion
					cv2.rectangle(frame, (minX, minY), (maxX, maxY),
						(0, 0, 255), 2)
			
			# update the background model and increment the total number
			# of frames read thus far
			md.update(gray)
			total += 1

			# acquire the lock, set the output frame, and release the
			# lock
			with lock:
				outputFrame = frame.copy()
		
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		32,))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host='127.0.0.1', port=8000, debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
cap.release()


# @app.post("/predict/image")
# async def predict_api():
# 	# extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
# 	# if not extension:
# 	#     return "Image must be jpg or png format!"
# 	# image = read_imagefile(await file.read())
# 	# prediction = predict(image)

	
# 	prediction = model.predict(sample)
	
# 	return prediction