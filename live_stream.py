import numpy as np
import cv2

cap = cv2.VideoCapture(0)
imageType = 1

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
# lower = np.array([0, 48, 80], dtype = "uint8")
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

while cap.isOpened():
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

	# show the skin in the image along with the mask
	cv2.imshow("images", np.hstack([frame, skin]))
 
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	continue


	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# blured = cv2.GaussianBlur(gray, (5, 5), 0)	

	# ret, binary = cv2.threshold(blured, 110, 220, cv2.THRESH_BINARY_INV)
	
	# binary_clone = np.copy(binary)

	# # contours, _ = cv2.findContours(binary_clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# contours, hierarchy = cv2.findContours(binary_clone, 2, 1)
	
	# # contours, _ = cv2.findContours(binary_clone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# # extract the largest contour
	# max_area = 0
	# ci = -1
	# contour = None
	# defects = None
	# hull = None
	
	# for i in range(len(contours)):
	# 	cnt = contours[i]
	# 	area = cv2.contourArea(cnt)
	# 	if(area > max_area):
	# 		max_area = area
	# 		ci = i
  	
 #  	if ci > -1:
 #  		contour = contours[ci]
	# 	# hull = cv2.convexHull(contour)
	# 	hull = cv2.convexHull(cnt, returnPoints = False)

	# 	if len(hull)>3 and len(contour)>3:
	# 		print "convexityDefects"			
	# 		defects = cv2.convexityDefects(contour, hull)

	# # clone frame
	# if imageType == 1:
	# 	clone = np.copy(frame)
	# elif imageType == 2:
	# 	clone = np.copy(gray)
	# else:
	# 	clone = np.copy(binary)

	# # cv2.drawContours(clone, contours, -1, (255, 0, 0), thickness = 2)
	# if contour is not None and hull is not None:
	# 	cv2.drawContours(clone, [contour], 0, (0, 255, 0), thickness = 2)
	# 	# cv2.drawContours(clone, [hull], 0, (0, 0, 255), thickness = 2)

	# 	if defects is not None:
	# 		for i in range(defects.shape[0]):
	# 		    s,e,f,d = defects[i,0]
	# 		    start = tuple(contour[s][0])
	# 		    end = tuple(contour[e][0])
	# 		    far = tuple(contour[f][0])
	# 		    cv2.line(clone,start,end,[0,255,0],2)
	# 		    cv2.circle(clone,far,5,[0,0,255],-1)

	# # Display the resulting frame
	# cv2.imshow('frame', clone)

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break

	# if cv2.waitKey(1) & 0xFF == ord('1'):
	# 	imageType = 1

	# if cv2.waitKey(1) & 0xFF == ord('2'):
	# 	imageType = 2

	# if cv2.waitKey(1) & 0xFF == ord('3'):
	# 	imageType = 3

cap.release()
cv2.destroyAllWindows()