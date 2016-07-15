import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
# lower = np.array([0, 48, 80], dtype = "uint8")
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

while( cap.isOpened() ):
    # Capture frame by frame
	ret, frame = cap.read()


	# Convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 1)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)



	gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
	blured = cv2.GaussianBlur(gray, (5, 5), 0)
	ret, binary = cv2.threshold(blured, 110, 255, cv2.THRESH_BINARY)
	

  	binary_clone = np.copy(binary)

  	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  	drawing = np.zeros(frame.shape, np.uint8)

  	max_area=0
  	ci = -1
  	# Find largest conture
  	for i in range(len(contours)):
  		cnt=contours[i]
  		area = cv2.contourArea(cnt)
  		if(area>max_area):
  			max_area=area
  			ci=i

  	if ci is not -1:
	  	cnt = contours[ci]
	  	hull = cv2.convexHull(cnt)
	  	moments = cv2.moments(cnt)
	  	if moments['m00'] != 0:
	  		cx = int(moments['m10'] / moments['m00']) # cx = M10/M00
	  		cy = int(moments['m01'] / moments['m00']) # cy = M01/M00

	  	center = (cx,cy)  	
	  	cv2.circle(frame, center, 5, [0,0,255], 2)
	  	
	  	cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
	  	cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

	  	cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
	  	hull = cv2.convexHull(cnt, returnPoints = False)

		
		text = "Close"	  	
		defects = cv2.convexityDefects(cnt, hull)
		if defects is not None:  		
	  		for i in range(defects.shape[0]):
	  			s,e,f,d = defects[i,0]
	  			start = tuple(cnt[s][0])
	  			end = tuple(cnt[e][0])
	  			far = tuple(cnt[f][0])
	  			dist = cv2.pointPolygonTest(cnt, center, True)
	  			cv2.line(frame, start, end, [0,255,0], 2)
	  			cv2.circle(frame, far, 5, [0, 0, 255], -1)
	  	
	  		if len(defects) >= 4:
	  			text = "Open"
		
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, text, (100, 100), font, 4, (255, 255, 255), 2)

  	cv2.imshow('output', drawing)
  	cv2.imshow('input', frame)

  	if cv2.waitKey(1) & 0xFF == ord('q'):
		break