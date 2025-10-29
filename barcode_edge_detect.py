import cv2
import numpy as np
import sys

# Usage: python barcode_edge_detect.py path/to/image.jpg
if len(sys.argv) < 2:
    print("Usage: python barcode_edge_detect.py path/to/image.jpg")
    sys.exit()

# Load image
image_path = sys.argv[1]
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Check the file name and path.")
    sys.exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradients using the Sobel operator
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# Blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
_, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# Morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Perform erosion and dilation
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# Find contours
contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    print("Barcode region detected.")
else:
    print("No barcode detected.")

# Show result
cv2.imshow("Barcode Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
