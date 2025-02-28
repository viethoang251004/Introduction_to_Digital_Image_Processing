# Ex1
import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) > 0 and len(right_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    else:
        return None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture('Lab10_test2.mp4')

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        break

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ex2
import cv2
import numpy as np
import os

# Define a function to detect lanes in an image
def detect_lanes(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Define the region of interest
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (100, image.shape[0]),
        (image.shape[1]-100, image.shape[0]),
        (image.shape[1]//2, image.shape[0]//2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    # Mask the edges image
    masked_edges = cv2.bitwise_and(edges, mask)
    # Hough transform for line detection
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Create an image to draw the lines on
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # Combine the line image with the original image
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image

# Create a directory to save the output images
output_dir = 'detected_lanes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Initialize video capture
cap = cv2.VideoCapture(0)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect lanes in the frame
    lane_image = detect_lanes(frame)
    # Save the frame with detected lanes
    cv2.imwrite(f'{output_dir}/frame_{frame_count}.jpg', lane_image)
    frame_count += 1
    # Display the frame with detected lanes
    cv2.imshow('Lane Detection', lane_image)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
