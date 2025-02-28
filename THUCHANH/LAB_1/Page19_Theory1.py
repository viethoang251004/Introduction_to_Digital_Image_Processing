# import numpy as np

# def fill_middle_rows_columns(image):
#     height, width = image.shape

#     # Create a copy of the original image
#     filled_image = image.copy()

#     # Calculate the middle rows and columns
#     middle_row = height // 2
#     middle_column = width // 2

#     # Fill the middle rows with white color
#     filled_image[middle_row-1:middle_row+2, :] = 1

#     # Fill the middle columns with white color
#     filled_image[:, middle_column-1:middle_column+2] = 1

#     return filled_image

# # Create a sample binary image
# image = np.zeros((9, 9), dtype=np.uint8)
# image[3:6, 2:7] = 1

# # Print the original image
# print("Original Image:")
# print(image)

# # Fill the middle rows and columns
# filled_image = fill_middle_rows_columns(image)

# # Print the filled image
# print("\nFilled Image:")
# print(filled_image)

import numpy as np
import matplotlib.pyplot as plt

def fill_middle_rows_columns(image):
    height, width = image.shape
    # Create a copy of the original image
    filled_image = image.copy()
    # Calculate the middle rows and columns
    middle_row = height // 2
    middle_column = width // 2
    # Fill the middle rows with white color
    filled_image[middle_row-1:middle_row+2, :] = 1
    # Fill the middle columns with white color
    filled_image[:, middle_column-1:middle_column+2] = 1
    return filled_image

# Create a sample binary image
image = np.zeros((9, 9), dtype=np.uint8)
image[3:6, 2:7] = 1
# Fill the middle rows and columns
filled_image = fill_middle_rows_columns(image)
# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
# Display the filled image
plt.subplot(1, 2, 2)
plt.imshow(filled_image, cmap='gray')
plt.title('Filled Image')
# Show the plot
plt.show()
