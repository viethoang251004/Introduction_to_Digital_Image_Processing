import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 1
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("geeksforgeeks.png", cv2.IMREAD_COLOR)

# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)

# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()


# Example 2
img = cv2.imread("geeks.png")
# Displaying image using plt.imshow() method
plt.imshow(img)

# hold the window
plt.waitforbuttonpress()
plt.close('all')


# Example 3
img = cv2.imread("geeks.png")

# Converting BGR color to RGB color format
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Displaying image using plt.imshow() method
plt.imshow(RGB_img)

# hold the window
plt.waitforbuttonpress()
plt.close('all')


# Example 4
# path
path = r'geeksforgeeks.png'

# Using cv2.imread() method
# Using 0 to read image in grayscale mode
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Displaying the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
