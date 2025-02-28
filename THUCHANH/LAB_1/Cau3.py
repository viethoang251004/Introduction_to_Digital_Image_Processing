# Python program to explain cv2.putText() method

# importing cv2
import cv2

# Example 1
# path
path = r'D:\Hoang\NHAPMONXULYANHSO\THUCHANH\geeks2.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
image = cv2.putText(image, 'OpenCV', org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

# Displaying the image
cv2.imshow(window_name, image)


# Example 2
# text
text = 'GeeksforGeeks'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (00, 185)

# fontScale
fontScale = 1

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
image = cv2.putText(image, text, org, font, fontScale,
                    color, thickness, cv2.LINE_AA, False)

# Using cv2.putText() method
image = cv2.putText(image, text, org, font, fontScale,
                    color, thickness, cv2.LINE_AA, True)

# Displaying the image
cv2.imshow(window_name, image)
