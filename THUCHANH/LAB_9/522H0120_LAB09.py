import numpy as np
import cv2 as cv
import math
import sys
import cv2


# Ex1
# Example 1
# Read the original image
img = cv2.imread('sudoku.png')
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1,
                   dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0,
                   dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
# Combined X and Y Sobel Edge Detection
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100,
                  threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save Sobel X image
cv2.imwrite('sobel_x.jpg', sobelx)
# Save Sobel Y image
cv2.imwrite('sobel_y.jpg', sobely)
# Save Sobel XY image
cv2.imwrite('sobel_xy.jpg', sobelxy)
# Save Canny edge detection image
cv2.imwrite('canny_edge.jpg', edges)

"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""

# Example 2


def main(argv):

    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
    cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.imwrite('source.jpg', src)
    cv.imwrite('hough_lines_standard.jpg', cdst)
    cv.imwrite('hough_lines_probabilistic.jpg', cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])


# def main2(argv):
#     default_file = 'sudoku.png'
#     filename = argv[0] if len(argv) > 0 else default_file
#     # Loads an image
#     src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
#     # Check if image is loaded fine
#     if src is None:
#         print('Error opening image!')
#         print(
#             'Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
#         return -1

#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

#     gray = cv.medianBlur(gray, 5)

#     rows = gray.shape[0]
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
#                               param1=100, param2=30,
#                               minRadius=1, maxRadius=30)

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         # circle center
#         cv.circle(src, center, 1, (0, 100, 100), 3)
#         # circle outline
#         radius = i[2]
#         cv.circle(src, center, radius, (255, 0, 255), 3)

#     cv.imshow("detected circles", src)
#     cv.waitKey(0)

#     return 0


# if __name__ == "__main2__":
#     main2(sys.argv[1:])




# Ex2
import cv2
import numpy as np

# Read the input image
image = cv2.imread('sudoku.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise
blurred = cv2.medianBlur(gray, 5)

# Perform edge detection using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Apply Hough line transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw the detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3)

# Display the image with detected lines
cv2.imshow("Detected Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Ex3
import cv2
bd = cv2.barcode.BarcodeDetector()
img = cv2.imread('barcode.png')
