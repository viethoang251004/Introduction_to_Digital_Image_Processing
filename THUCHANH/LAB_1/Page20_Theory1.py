import numpy as np
import matplotlib.pyplot as plt

def invert_binary_image(image):
    inverted_image = 1 - image
    return inverted_image

# Create a sample binary image
image = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1]])

# Invert the colors of the binary image
inverted_image = invert_binary_image(image)
# Display the original and inverted images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(inverted_image, cmap='gray')
plt.title('Inverted Image')
plt.show()