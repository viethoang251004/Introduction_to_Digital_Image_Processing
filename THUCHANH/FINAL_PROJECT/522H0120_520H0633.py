import cv2
import numpy as np
import math
import os

# Function to resize image to a standard size
def resize_image(img):
    height, width, _ = img.shape
    scale_factor = 1000 / max(height, width)
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
    return img

# Function to detect the clock in the image
def clock_detection(img, blurred):
    # Detect circles (clock face) using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=100, minRadius=100, maxRadius=500)
    if circles is not None:
        max_circle = None
        for circle in circles[0, :]:
            x, y, r = circle
            if r > radius:
                max_circle = circle
        x, y, r = max_circle
        center_x = int(x)
        center_y = int(y)
        radius = int(r)
    else:
        # If no circle is detected, fallback to contour detection
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rect = None
        # Find the largest contour (assumed to be the clock face)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_rect = contour
        x, y, w, h = cv2.boundingRect(max_rect)
        center_x = x + w // 2
        center_y = y + h // 2
        radius = min(w, h) // 2
    return center_x, center_y, radius

# Function to detect lines in the image
def line_detection(img, blurred):
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=30, maxLineGap=5)
    return lines

# Function to group detected lines by angle and proximity
def group_lines_detection(lines, center_x, center_y, radius):
    groups = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
        max_length = np.max([length1, length2])
        min_length = np.min([length1, length2])
        if (max_length < radius) and (min_length < radius*50/100):
            angle = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle)
            grouped = False
            for group in groups:
                mean_angle = group['mean_angle']
                # Check if the angle is close to the mean angle of the group
                if abs(angle - mean_angle) < 12 or abs(angle - mean_angle - 180) < 12 or abs(angle - mean_angle + 180) < 12:
                    group['lines'].append(line)
                    grouped = True
                    break
            if not grouped:
                groups.append({'lines': [line], 'mean_angle': angle})
    return groups

# Function to calculate the distance between two parallel lines
def distance_between_parallel_lines(line1, line2):
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]
    vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
    vector2 = np.array([x2_2 - x1_2, y2_2 - y1_2])
    vector_between_lines = np.array([x1_2 - x1_1, y1_2 - y1_1])
    distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)
    return distance

# Function to detect hands of the clock based on line thickness and length
def hands_detection(groups, center_x, center_y):
    hands = []
    for group in groups:
        lines = group['lines']
        num_lines = len(lines)
        max_thickness = 0
        max_length = 0
        for i in range(num_lines):
            x1, y1, x2, y2 = lines[i][0]
            length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            length = np.max([length1, length2])
            if length > max_length:
                max_length = length
                if length == length1:
                    max_line = x1, y1, center_x, center_y
                else:
                    max_line = x2, y2, center_x,center_y
            for j in range(i+1, num_lines):
                thickness = distance_between_parallel_lines(lines[i], lines[j])
                if thickness > max_thickness:
                    max_thickness = thickness
        line = max_line, max_thickness, max_length
        if max_thickness > 0:
            hands.append(line)
    hands.sort(key=lambda x: x[2], reverse=True)  # Sort hands by length
    hands = hands[:3]  # Select the three longest hands
    return hands

# Function to determine which hand is which (hour, minute, second)
def get_hands(hands):
    sorted_hands_by_thickness = sorted(hands, key=lambda hands: hands[1])  # Sort by thickness
    second_hand = sorted_hands_by_thickness[0]  # The thinnest hand is the second hand
    hands.remove(second_hand)  # Remove the second hand from the list
    sorted_hands_by_length = sorted(hands, key=lambda hands: hands[2])  # Sort the remaining by length
    hour_hand = sorted_hands_by_length[0]  # The shortest of the remaining is the hour hand
    minute_hand = sorted_hands_by_length[1]  # The second shortest is the minute hand
    return hour_hand, minute_hand, second_hand

# Function to calculate the bounding rectangle coordinates around a hand
def calculate_rect_coordinates(hand):
    x1, y1, x2, y2 = hand[0]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    rect_width = abs(x2 - x1)
    rect_height = abs(y2 - y1)
    text_x = center_x
    text_y = center_y
    # Add a buffer around the hand for the rectangle
    rect_width += 20
    rect_height += 20
    return center_x - rect_width // 2, center_y - rect_height // 2, rect_width, rect_height, text_x, text_y

# Function to draw rectangles around detected hands and label them (hour, minute, second)
def draw_hands_frame(img, hour_hand, minute_hand, second_hand):
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(hour_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 3)  # Red for hour hand
    cv2.putText(img, 'Hour', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(minute_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 3)  # Green for minute hand
    cv2.putText(img, 'Minute', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(second_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 3)  # Blue for second hand
    cv2.putText(img, 'Second', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Function to calculate the vector of a line
def get_vector(hand):
    x1, y1, x2, y2 = hand[0]
    vector = [x2 - x1, y2 - y1]
    return vector

# Function to compute the dot product of two vectors
def dot_product(u, v):
    return u[0] * v[0] + u[1] * v[1]

# Function to compute the cross product of two vectors
def cross_product(u, v):
    return u[0] * v[1] - u[1] * v[0]

# Function to get the angle between a hand and the vertical (12 o'clock position)
def get_angle(hand, center_x, center_y):
    u = get_vector(hand)
    v = [center_x - center_x, center_y - (center_y-100)]
    dot_uv = dot_product(u, v)
    length_u = math.sqrt(u[0]**2 + u[1]**2)
    length_v = math.sqrt(v[0]**2 + v[1]**2)
    cos_theta = dot_uv / (length_u * length_v)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta)
    theta_degrees = math.degrees(theta)
    cross_uv = cross_product(u, v)
    if cross_uv > 0:
        return 360 - theta_degrees
    else:
        return theta_degrees


# Function to convert angles of clock hands into a time string
def get_time(hour_angle, minute_angle, second_angle):
    hour = hour_angle / 30  # Convert hour angle to hour value (12 hours on a clock)
    minute = minute_angle / 6  # Convert minute angle to minute value (60 minutes on a clock)
    second = second_angle / 6  # Convert second angle to second value (60 seconds on a clock)

    # Adjust hour, minute, and second values based on specific conditions
    if (round(hour)*30 - hour_angle <= 6) and ((355 < minute_angle and minute_angle < 360) or (minute_angle < 90)):
        hour = round(hour)
        if hour == 12:
            hour = 0
    if (hour_angle - hour*30 <= 6) and (355 < minute_angle and minute_angle < 360):
        minute = 0
    if (round(minute)*6 - minute_angle <= 6)and (second_angle < 6):
        minute = round(minute)
        if minute == 60:
            minute = 0
    if (minute_angle - minute*30 <= 6) and (354 < second_angle and second_angle < 360):
        second = 0

    # Convert values to integers and format them into a time string (HH:MM:SS)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    time = f"{hour:02d}:{minute:02d}:{second:02d}"
    return time

# Function to draw the time string onto the image
def draw_time(img, time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_position = (50, 100)
    text_color = (0, 0, 0)
    cv2.putText(img, time, text_position, font, font_scale, text_color, font_thickness)

# Function to process the image and detect the clock hands
def solve(img):
    img = resize_image(img)  # Resize the image
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  # Convert image to HSV color space
    img_hsv = cv2.bitwise_not(img_hsv)  # Invert HSV image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object for contrast enhancement
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])  # Apply CLAHE to the V channel of HSV image
    _, thresh = cv2.threshold(img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply Otsu's thresholding
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)  # Apply Gaussian blur for smoothing
    center_x, center_y, radius = clock_detection(img, blurred)  # Detect the center and radius of the clock
    lines = line_detection(img, blurred)  # Detect lines in the image
    groups = group_lines_detection(lines, center_x, center_y, radius)  # Group lines into potential clock hands
    hands = hands_detection(groups, center_x, center_y)  # Detect the clock hands
    hour_hand, minute_hand, second_hand = get_hands(hands)  # Identify the hour, minute, and second hands
    draw_hands_frame(img, hour_hand, minute_hand, second_hand)  # Draw rectangles around detected hands
    hour_angle = get_angle(hour_hand, center_x, center_y)  # Calculate the angle of the hour hand
    minute_angle = get_angle(minute_hand, center_x, center_y)  # Calculate the angle of the minute hand
    second_angle = get_angle(second_hand, center_x, center_y)  # Calculate the angle of the second hand
    time = get_time(hour_angle, minute_angle, second_angle)  # Convert angles to a time string
    draw_time(img, time)  # Draw the time string onto the image
    return img

# Main part of the code
input_folder = "input"  # Path to the input folder containing images
output_folder = "output"  # Path to the output folder for processed images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        # Read the image
        img = cv2.imread(os.path.join(input_folder, filename))
        
        # Process the image
        output_img = solve(img)
        
        # Save the output image
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, output_img)
        
        print(f"Processed {filename} and saved to {output_filename}")

print("All files processed!")
