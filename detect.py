import cv2
import numpy as np

# Load cascade files
cascade_src = 'cars.xml'
pedestrian_src = 'pedestrian.xml'
video_src = 'Road_Lane.mp4'

# Load pre-trained cascades
car_cascade = cv2.CascadeClassifier(cascade_src)
bike_cascade = cv2.CascadeClassifier(pedestrian_src)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Function: Define region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function: Draw detected lines
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# Function: Process each frame for lane detection
def process(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width // 2, int(height / 1.5)),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(
        canny_image, np.array([region_of_interest_vertices], np.int32)
    )
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

# Main video capture and processing
cap = cv2.VideoCapture(video_src)

if not cap.isOpened():
    print(f"Error: Cannot open video source '{video_src}'")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for lane detection
    processed_frame = process(frame)


    # Detect cars
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display both lane detection and car detection
    cv2.imshow('Lane Detection', processed_frame)
    cv2.imshow('Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
