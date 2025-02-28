# Python program to explain cv2.circle() method
import cv2
import numpy as np


# Example 1
# path
path = r'D:\Hoang\NHAPMONXULYANHSO\THUCHANH\geeks2.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# Center coordinates
center_coordinates = (120, 50)

# Radius of circle
radius = 20

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)


# Example 2
# Center coordinates 
center_coordinates = (120, 100) 
  
# Radius of circle 
radius = 30
   
# Red color in BGR 
color = (0, 0, 255) 
   
# Line thickness of -1 px 
thickness = -1
   
# Using cv2.circle() method 
# Draw a circle of red color of thickness -1 px 
image = cv2.circle(image, center_coordinates, radius, color, thickness) 
   
# Displaying the image  
cv2.imshow(window_name, image)


# Example 3
# Reading an image in default mode 
Img = np.zeros((512, 512, 3), np.uint8) 
     
# Window name in which image is displayed 
window_name = 'Image'
    
# Center coordinates 
center_coordinates = (220, 150) 
   
# Radius of circle 
radius = 100
    
# Red color in BGR 
color = (255, 133, 233) 
    
# Line thickness of -1 px 
thickness = -1
    
# Using cv2.circle() method 
# Draw a circle of red color of thickness -1 px 
image = cv2.circle(Img, center_coordinates, radius, color, thickness) 
    
# Displaying the image 
cv2.imshow(window_name, image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()