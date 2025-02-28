# Python program to explain cv2.arrowedLine() method
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

# End coordinate
end_point = (200, 200)

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 9 px
thickness = 9

# Using cv2.arrowedLine() method
# Draw a diagonal green arrow line
# with thickness of 9 px
image = cv2.arrowedLine(image, start_point, end_point,
                        color, thickness)

# Displaying the image
cv2.imshow(window_name, image)


# Example 2
# Start coordinate, here (225, 0)
# represents the top right corner of image
start_point = (225, 0)

# End coordinate
end_point = (0, 90)

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 9 px
thickness = 9

# Using cv2.arrowedLine() method
# Draw a red arrow line
# with thickness of 9 px and tipLength = 0.5
image = cv2.arrowedLine(image, start_point, end_point,
                        color, thickness, tipLength=0.5)

# Displaying the image
cv2.imshow(window_name, image)
