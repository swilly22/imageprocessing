import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Resize
# cap.set(3, 320)
# cap.set(4, 240)
while cap.isOpened():
	# Capture frame by frame
	ret, frame = cap.read()

	blur = cv2.GaussianBlur(frame, (5, 5), 0)

	# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
	sobelx64f = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=5)
	abs_sobel64f = np.absolute(sobelx64f)
	sobel_8u = np.uint8(abs_sobel64f)

	# resize original frame
	original_resized = cv2.resize(frame, None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
	
	# placemeant position
	x_offset = y_offset = 0

	# draw resized original frame ontop of sobel
	sobel_8u[y_offset:y_offset + original_resized.shape[0], x_offset:x_offset + original_resized.shape[1]] = original_resized
	
	# Display the resulting frame
	cv2.imshow('frame', sobel_8u)

	# press q to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()