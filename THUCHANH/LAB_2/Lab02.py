import cv2
import numpy as np

# Ex2.1
# Cau 1
# Load the image
img = cv2.imread('lab02_ex.png')
# Split each color channel
b, g, r = cv2.split(img)


# Cau 2
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw bounding rectangles for each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('522H0120_Image with Bounding Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cau 3
# Define the color names
colors = ['Yellow', 'Blue', 'Red', 'Green', 'Orange']
# Draw bounding rectangles and color names for each contour
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, colors[i % len(colors)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow('522H0120_Image with Bounding Boxes and Color Names', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cau 4
# Load the image
image = cv2.imread('lab02_ex.png')
# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define the lower and upper bounds for yellow color in HSV
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)
# Create a mask for yellow pixels
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# Apply the mask to the original image to extract the yellow balloon
yellow_balloon = cv2.bitwise_and(image, image, mask=yellow_mask)

# Display the yellow balloon image
cv2.imshow('522H0120_Yellow Balloon', yellow_balloon)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cau 5
# Load the image
img = cv2.imread('lab02_ex.png')
# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Define the range of yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# Threshold the HSV image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# Bitwise-AND mask and original image
yellow_balloon = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('522H0120_Yellow Balloon', yellow_balloon)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cau 6
image = cv2.imread('lab02_ex.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
image[np.where(yellow_mask == 255)] = (0, 255, 0)

cv2.imshow('522H0120_Green Balloon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cau 7
# Get the center of the balloon
center = (x + w//2, y + h//2)
# Compute the rotation matrix
M = cv2.getRotationMatrix2D(center, 20, 1.0)
# Perform the rotation
rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Display the rotated image
cv2.imshow('522H0120_Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ex2.2
# Cau 8
from PIL import Image, ImageEnhance

# Open an image file
with Image.open('face.jpg') as img:
    # Create a Brightness enhancer
    enhancer = ImageEnhance.Brightness(img)
    
    # Increase the brightness by a factor of 2
    brighter_img = enhancer.enhance(2)
    
    # Save the brightened image
    brighter_img.save('brightened_image.jpg')

# Cau 9
img = cv2.imread('face.jpg', 0)
img_equ = cv2.equalizeHist(img)
cv2.imwrite('enhanced_image.jpg', img_equ)

# Cau 10
image = cv2.imread('face.jpg', 0)

# Create a CLAHE object (Arguments are optional)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)

# Save the image
cv2.imwrite('clahe_image.jpg', cl1)

# page 25
from PIL import Image
import numpy as np

image = Image.open('face.jpg').convert('L')
data = np.array(image)
# Apply thresholding
binary = np.where(data < 128, 0, 255)
# Create a new image from the binary array
new_image = Image.fromarray(binary.astype(np.uint8))
# Save the new image
new_image.save('thresholded_image.jpg')

# page 32
image = Image.open('RGB_image.png')
data = np.array(image)
red_pixels = np.all(data == [255, 0, 0], axis=-1)
data[red_pixels] = [255, 255, 255]
data[~red_pixels] = [0, 0, 0]
# Create a new image from the transformed array
new_image = Image.fromarray(data)
new_image.save('color_transformed_image.jpg')

# page 38
image = cv2.imread('face.png')
data = np.array(image)
lightness = (np.max(data, axis=2) + np.min(data, axis=2)) / 2
average = np.mean(data, axis=2)
luminosity = 0.21 * data[:,:,0] + 0.72 * data[:,:,1] + 0.07 * data[:,:,2]
cv2.imwrite('lightness.jpg', lightness)
cv2.imwrite('average.jpg', average)
cv2.imwrite('luminosity.jpg', luminosity)

# page 46
image = cv2.imread('RGB_image.jpg')
data = np.array(image)
cmy = 1 - (data / 255.0)
rgb = (1 - cmy) * 255
# Save the new images
cv2.imwrite('cmy_image.jpg', cmy)
cv2.imwrite('rgb_image.jpg', rgb)

# page 48
from PIL import Image
image = Image.open('RGB_image2.jpg')
image = image.convert('RGB')
image_cmyk = image.convert('CMYK')
image_cmyk.save('cmyk_image.jpg')
