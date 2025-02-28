# Python program to explain cv2.ellipse() method
import cv2


# Example 1
# path
path = r'D:\Hoang\NHAPMONXULYANHSO\THUCHANH\geeks2.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

center_coordinates = (120, 100)

axesLength = (100, 50)

angle = 0

startAngle = 0

endAngle = 360

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 5 px
thickness = 5

# Using cv2.ellipse() method
# Draw a ellipse with red line borders of thickness of 5 px
image = cv2.ellipse(image, center_coordinates, axesLength,
                    angle, startAngle, endAngle, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)


# Example 2
center_coordinates = (120, 100)

axesLength = (100, 50)

angle = 30

startAngle = 0

endAngle = 360

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of -1 px
thickness = -1

# Using cv2.ellipse() method
# Draw a ellipse with blue line borders of thickness of -1 px
image = cv2.ellipse(image, center_coordinates, axesLength, angle,
                    startAngle, endAngle, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)
