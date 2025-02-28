import cv2
import numpy as np

# Exercise 1
# Load the image
img = cv2.imread('lab3_img1.png')
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Detect faces in the image with adjusted scaleFactor and minNeighbors
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
# Sort the detected faces by the x-coordinate of the bounding box
faces = sorted(faces, key=lambda f: f[0])
# For each detected face
for i, (x, y, w, h) in enumerate(faces):
    # Create a circular mask
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), min(w//2,h//2), (255), thickness=-1)
    # Apply the mask to the face 
    face = np.zeros_like(img[y:y+h,x:x+w])
    face[mask == 255] = img[y:y+h,x:x+w][mask == 255]
    # Save each extracted circular masked-face with a unique name based on counter value 
    filename = f'face{i+1}.png'
    cv2.imwrite(filename, face)

# Exercise 2
# Load the two images
img1 = cv2.imread('lab3_img1.png')
img2 = cv2.imread('lab3_img2.png')
# Resize the images to have the same dimensions (if needed)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
# Blend the images using alpha blending
alpha = 0.5  # Adjust the blending ratio as desired
blended_img = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
# Save the blended image
cv2.imwrite('blended_image.png', blended_img)

# Exercise 3
# Đọc hình ảnh đầu vào
src = cv2.imread("threshold.png", cv2.IMREAD_GRAYSCALE)
# Chuyển đổi hình ảnh thành hình ảnh nhị phân
# Hình ảnh nhị phân thứ nhất: số lớn hơn hoặc bằng 180 được chuyển thành màu đen
_, dst180black = cv2.threshold(src, 179, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Output 1", dst180black)
# Hình ảnh nhị phân thứ hai: số nhỏ hơn 180 được chuyển thành màu trắng
_, dst = cv2.threshold(src, 179, 255, cv2.THRESH_TOZERO_INV)
_, dst = cv2.threshold(dst, 4, 255, cv2.THRESH_BINARY)
cv2.imshow("Output 2", dst)
# Trích xuất từng số trong hình ảnh đầu vào thành các hình ảnh riêng biệt
# _, dst = cv2.threshold(src, 128, 255, cv2.THRESH_TOZERO_INV)
# _, dst = cv2.threshold(dst, 127, 128, cv2.THRESH_BINARY)
# cv2.imshow("Extracted Numbers", dst)
numbers = [32, 64, 100, 128, 180, 200, 255]
for number in numbers:
    mask = cv2.inRange(src, number, number)
    output = cv2.bitwise_and(src, src, mask = mask)
    cv2.imshow(f"Extracted Number {number}", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Homework
# Load the logo image
logo = cv2.imread('lab3_img2.png')
# Resize the logo to desired size
logo = cv2.resize(logo, (50, 50))
# Start the webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Define the region of interest (ROI) for logo placement
    roi = frame[-50:, -50:]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(logo, logo, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    frame[-50:, -50:] = dst
    # Display the resulting frame
    cv2.imshow('Video with Logo', frame)
    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
