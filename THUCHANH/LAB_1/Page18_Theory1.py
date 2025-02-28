import numpy as np
import matplotlib.pyplot as plt

# Define the fill_border_with_black function
def fill_border_with_black(image):
    height, width = image.shape
    # Create a copy of the original image
    filled_image = np.ones((height, width), dtype=np.uint8) * 255  # Fill with white color
    # Fill the border with black
    filled_image[0:2, :] = 0
    filled_image[-2:, :] = 0
    filled_image[:, 0:2] = 0
    filled_image[:, -2:] = 0
    return filled_image

# Create a 7x6 image with all white pixels (value=1)
image = np.ones((6, 7), dtype=np.uint8)
# Apply the fill_border_with_black function to the image
filled_image = fill_border_with_black(image)
# Plot the original and filled images using matplotlib's 'gray' colormap for better visualization of black and white images.
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(filled_image, cmap='gray')
plt.title('Filled Border Image')
plt.show()

# import numpy as np

# def fill_border_with_black(image):
#     height, width = image.shape

#     # Create a copy of the original image
#     filled_image = image.copy()

#     # Fill the top and bottom borders with black
#     filled_image[:2, :] = 0
#     filled_image[-2:, :] = 0

#     # Fill the left and right borders with black
#     filled_image[:, :2] = 0
#     filled_image[:, -2:] = 0

#     return filled_image

# # Create a sample binary image
# image = np.zeros((10, 10), dtype=np.uint8)
# image[2:8, 2:8] = 1

# # Print the original image
# print("Original Image:")
# print(image)

# # Fill the border
# filled_image = fill_border_with_black(image)

# # Print the filled image
# print("\nFilled Image:")
# print(filled_image)