from pyzbar.pyzbar import decode
from PIL import Image
# Load the image
image = Image.open('Example.png')
# Decode the barcode
decoded_objects = decode(image)
# Print the barcode information
for obj in decoded_objects:
    print('Type:', obj.type)
    print('Data:', obj.data.decode('utf-8'))
