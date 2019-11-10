import cv2


# cameras listed as variable in VideoCapture "0, 1, 2" depending on the number of cameras
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture(0)

# static frame is the first frame taken by the camera.
# this should be a still image of the background you wish to monitor.
static_frame = None

while True:
	check, frame = video.read()
	# print(check)
	# print(frame)
	# # resize the frame to see entire picture
	# resized = cv2.resize(frame, (int(frame.shape[1] / 1), int(frame.shape[0] / 1)))
	# grayscale conversion of the captured frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# this blurs the image slightly allowing the program to better match differences.
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# creating the static frame.
	if static_frame is None:
		static_frame = gray
		continue

	# this is the comparison being done between the two
	delta_frame = cv2.absdiff(static_frame, gray)
	# creates the threshold parameter to reduce noise in detection.
	thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	# smooth threshold frame
	thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
	# # create a box around the target in frame
	# for x, y, w, h, in faces:
	# 	img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
	# display current frame capture. so we can see it
	cv2.imshow("Stream", gray)
	# display our static frame for comparison. so we can see it.
	cv2.imshow("Difference", delta_frame)
	# display threshold image
	cv2.imshow("Noise", thresh_frame)
	# holds captured frame open until you close it
	key = cv2.waitKey(100)
	if key == ord('q'):
		break


# stops reading video or turns off the camera as you use it
video.release()
cv2.destroyAllWindows()