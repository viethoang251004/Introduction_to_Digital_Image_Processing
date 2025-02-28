import cv2
import numpy as np
from matplotlib import pyplot as plt
# from skimage import data, filters

# Ex5.1. Image histogram equalization
# Load the image in grayscale mode
img = cv2.imread('g1onwm0o.png', 0)
# Calculate the histogram of the image
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# Plot the grayscale image histogram
plt.plot(hist)
plt.title('Grayscale Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# Perform histogram equalization
equalized_img = cv2.equalizeHist(img)
# Calculate the histogram of the equalized image
equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
# Plot the histogram of the equalized image
plt.plot(equalized_hist)
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# Display the original and equalized images
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ex5.2. Image blurring
image = cv2.imread('g1onwm0o.png')
# Add noise to the image
noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)
# Apply different blurring techniques
blurred_images = []
blur_titles = []
# Gaussian blur
gaussian_blur = cv2.GaussianBlur(noisy_image, (5, 5), 0)
blurred_images.append(gaussian_blur)
blur_titles.append('Gaussian Blur')
# Median blur
median_blur = cv2.medianBlur(noisy_image, 5)
blurred_images.append(median_blur)
blur_titles.append('Median Blur')
# Box blur
box_blur = cv2.boxFilter(noisy_image, -1, (5, 5))
blurred_images.append(box_blur)
blur_titles.append('Box Blur')
# Motion blur
motion_kernel = np.zeros((9, 9))
motion_kernel[4, :] = 1 / 9
motion_blur = cv2.filter2D(noisy_image, -1, motion_kernel)
blurred_images.append(motion_blur)
blur_titles.append('Motion Blur')
# Depth of field blur
depth_of_field_blur = cv2.GaussianBlur(noisy_image, (0, 0), 10)
blurred_images.append(depth_of_field_blur)
blur_titles.append('Depth of Field Blur')
# Display the original and blurred images
titles = ['Original Image'] + blur_titles
images = [image] + blurred_images

for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Ex5.3. Image sharpening
image = cv2.imread('g1onwm0o.png')
# Define the blur kernel
blur_kernel = np.ones((5, 5), np.float32) / 25
# Sharpen the image using addWeighted
sharpened_add = cv2.addWeighted(image, 1.5, image, -0.5, 0)
# Sharpen the image using filter2D
sharpened_filter = cv2.filter2D(image, -1, blur_kernel)
# Display the original and sharpened images
titles = ['Original Image', 'Sharpened (addWeighted)', 'Sharpened (filter2D)']
images = [image, sharpened_add, sharpened_filter]

for i in range(len(images)):
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Ex5.4. Simple Motion Estimation in Videos using OpenCV
# import numpy as np
# import cv2
# from skimage import data, filters

# Open Video
cap = cv2.VideoCapture('video.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):

  # Read frame
  ret, frame = cap.read()
  # Convert current frame to grayscale
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Calculate absolute difference of current frame and 
  # the median frame
  dframe = cv2.absdiff(frame, grayMedianFrame)
  # Treshold to binarize
  th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
  # Display image
  cv2.imshow('frame', dframe)
  cv2.waitKey(20)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()