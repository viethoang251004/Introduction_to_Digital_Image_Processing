import cv2 as cv2
import numpy as np

img = cv2.imread('input1.jpg')
kernel = np.ones((5,5),np.uint8)
def extract_color(img, upper, lower):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower, upper) # Create a mask with the specified color range
    result = cv2.bitwise_and(img, img, mask=mask) # Apply the mask to the original image
    closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel) # Reduce noise in the image

    return closing
lower_yel = np.array([0, 230, 238])
upper_yel = np.array([190, 255, 255])

lower_orange = np.array([0, 65, 70])
upper_orange = np.array([50, 150, 255])

lower_pink = np.array([70, 0, 150])
upper_pink = np.array([150, 100 , 250])

lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 150, 10])

lower_green = np.array([0, 145, 0])
upper_green = np.array([150, 255, 130])

lower_purple = np.array([100, 30, 100])
upper_purple = np.array([150, 100, 160])

# The color of yellow and orange are similar to backgrond color, so we need to get it sperately and apply the thresh
mask_yel = cv2.inRange(img, lower_yel, upper_yel)
mask_orange = cv2.inRange(img, lower_orange, upper_orange)
mask = cv2.add(mask_yel, mask_orange)
mask = cv2.bitwise_not(mask)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN , (5,5))

gray = cv2.imread('input1.jpg', 0)
gray = cv2.medianBlur(gray, 21)
thresh  = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

result = cv2.bitwise_and(thresh, opening) 

cv2.imwrite('Yellow_star.jpg', extract_color(img, upper_yel, lower_yel))
cv2.imwrite('Orange_star.jpg', extract_color(img, upper_orange, lower_orange))
cv2.imwrite('Pink_star.jpg', extract_color(img, upper_pink, lower_pink))
cv2.imwrite('Blue_star.jpg', extract_color(img, upper_blue, lower_blue))
cv2.imwrite('Green_star.jpg', extract_color(img, upper_green, lower_green))
cv2.imwrite('Purple_star.jpg', extract_color(img, upper_purple, lower_purple))
cv2.imwrite('Dark_stars.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################################################################################
image = cv2.imread('input2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 15), 0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
kernel = np.ones((3, 3), np.uint8)

closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Set filter for counter if it too small, ignore it (for address the noise)
    if cv2.contourArea(contour) < 250:
        continue
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    # Set filter for counter if it too long, ignore it (for address the line underneath digits)
    aspect_ratio = w / float(h)
    if aspect_ratio > 2:  
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite('Boundingbox_digits.png', image)
