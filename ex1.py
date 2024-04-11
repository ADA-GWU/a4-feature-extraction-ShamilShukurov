import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image

file_path = 'feature_extraction/original_images/ex1.jpg'
# print(os.path.exists(file_path))
image = cv2.imread(file_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Gaussian blur to smooth the image to help edge detection
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)


# Edge detection using Canny
edges = cv2.Canny(blurred_image, 40, 50)


# Corner detection 
corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.1, minDistance=150)
corners = np.intp(corners)

# Create a copy of the original image to draw corners on
image_with_corners = np.copy(image)
for i in corners:
    x, y = i.ravel()
    cv2.circle(image_with_corners, (x, y), 3, (0, 255, 0), -1)

    
# Line detection using HoughLines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=20)

# Create a copy of the original image to draw lines on
image_with_lines = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
# Circle detection using HoughCircles
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=200, param2=30, minRadius=0, maxRadius=0)


# Create a copy of the original image to draw circles on
image_with_circles = np.copy(image)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image_with_circles, (i[0], i[1]), 2, (0, 0, 255), 3)

# Show the images
plt.figure(figsize=(10, 10))

plt.subplot(221)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')

plt.subplot(222)
plt.title('Corner Detection')
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))

plt.subplot(223)
plt.title('Line Detection')
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))

plt.subplot(224)
plt.title('Circle Detection')
plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()