import cv2

# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# load the photograph
pixels = cv2.imread('test1.jpg')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
    # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
# save the image
cv2.imwrite('face_detection.jpg', pixels)
# load the photograph
pixels = cv2.imread('test1.jpg')
# example of face detection with opencv cascade classifier
# load the photograph
pixels = cv2.imread('test1.jpg')
# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
    print(box)
# extracta
x, y, width, height = box
x2, y2 = x + width, y + height
# draw a rectangle over the pixels
cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
# show the image
# cv2.imshow('face detection', pixels)
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()
# plot photo with detected faces using opencv cascade classifier

# load the photograph
pixels = cv2.imread('test1.jpg')
# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
    # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
# show the image
cv2.imshow('face detection', pixels)
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()
