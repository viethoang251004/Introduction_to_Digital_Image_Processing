import numpy as np
import cv2

# Load the binary image
B = cv2.imread('Example.png', cv2.IMREAD_GRAYSCALE)
# Ensure the image is binary
B = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)[1]
# Get the size of the image
M, N = B.shape
# Initialize the average intensity
avgI = 0
# Calculate the sum of the pixel intensities
for r in range(M):
    for c in range(N):
        avgI += B[r, c]

# Calculate the average intensity
avgI = avgI / (M * N)
print(f'Average intensity: {avgI}')
