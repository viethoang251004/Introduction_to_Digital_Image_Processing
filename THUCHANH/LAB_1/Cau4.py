# Python program to explain cv2.line() method
import numpy as np
import cv2


# Example 1
# path
path = r'D:\Hoang\NHAPMONXULYANHSO\THUCHANH\geeks2.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# Start coordinate, here (0, 0)
# represents the top left corner of image
start_point = (0, 0)

# End coordinate, here (250, 250)
# represents the bottom right corner of image
end_point = (250, 250)

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 9 px
thickness = 9

# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px
image = cv2.line(image, start_point, end_point, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)


# Example 2
# Reading an image in grayscale mode
image = cv2.imread(path, 0)

# Window name in which image is displayed
window_name = 'Image'

# Start coordinate, here (225, 0)
# represents the top right corner of image
start_point = (225, 0)

# End coordinate, here (0, 225)
# represents the bottom left corner of image
end_point = (0, 225)

# Black color in BGR
color = (0, 0, 0)

# Line thickness of 5 px
thickness = 5

# Using cv2.line() method
# Draw a diagonal black line with thickness of 5 px
image = cv2.line(image, start_point, end_point, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)


# Example 3
# Creating a black screen image using numpy.zeros function
Img = np.zeros((512, 512, 3), dtype='uint8')
# Start coordinate, here (100, 100). It represents the top left corner of image
start_point = (100, 100)
# End coordinate, here (450, 450). It represents the bottom right corner of the image according to resolution
end_point = (450, 450)
# White color in BGR
color = (255, 250, 255)
# Line thickness of 9 px
thickness = 9
# Using cv2.line() method to draw a diagonal green line with thickness of 9 px
image = cv2.line(Img, start_point, end_point, color, thickness)
# Display the image
cv2.imshow('Drawing_Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
