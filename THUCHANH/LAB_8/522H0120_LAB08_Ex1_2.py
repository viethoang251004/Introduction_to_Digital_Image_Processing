import cv2
import dlib
# Load the detector
detector = dlib.get_frontal_face_detector()
# Start the capture
cap = cv2.VideoCapture(0)
while True:
    # Read frames from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = detector(gray)
    # Draw rectangles around the faces
    for rect in faces:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the frame with faces marked
    cv2.imshow('Frame', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()