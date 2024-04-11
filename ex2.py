import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian

# Initialize list to store points of the contour
points = []

# Mouse callback function to detect clicks and capture points
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Load the image
image = cv2.imread('feature_extraction/original_images/ex1.jpg')
gray_image = rgb2gray(image)
smoothed_image = gaussian(gray_image, 3)

# Set up window and set mouse callback function
cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Image', select_points)

# Display the image and prompt the user to click to define the initial contour
print("Click to select points for the initial contour and press Enter.")
cv2.imshow('Image', image)

# Wait until enter is pressed
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 13:  # Enter key
        break

# Close the image window
cv2.destroyAllWindows()

# If the user did not select enough points, exit
if len(points) < 5:
    print("Not enough points selected.")
    exit(0)
print(points)

# Convert the points to a numpy array
init = np.array(points)

# Perform Active Contour (snakes)
snake = active_contour(smoothed_image, init, alpha=0.015, beta=10, gamma=0.001)

# Display the result
image_with_snake = image.copy()
cv2.polylines(image_with_snake, [np.int32(snake)], isClosed=True, color=(0, 255, 0), thickness=5)
cv2.imshow('Snakes', image_with_snake)
cv2.waitKey(0)
cv2.destroyAllWindows()



