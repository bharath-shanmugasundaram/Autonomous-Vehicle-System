import cv2
import numpy as np

def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    left_line = None
    right_line = None
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    
    if left_line is not None and right_line is not None:
        return np.array([left_line, right_line])
    elif left_line is not None:
        return np.array([left_line])
    elif right_line is not None:
        return np.array([right_line])
    else:
        return None

def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Use BGR if image is loaded with cv2.imread or from a video
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage

def region_of_interest(image):
    height = image.shape[0]
    width=image.shape[1]
    polygon = np.array([[
        (0, height),
        (width * 0.45, height * 0.6), 
        (width * 0.55, height * 0.6), 
        (width, height)              
    ]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Set color based on the slope (right lanes typically have positive slopes)
            color = (0, 255, 0) 
            cv2.line(line_image, (x1, y1), (x2, y2), color, 10)
    return line_image

cap = cv2.VideoCapture("Resource/test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        break
    canny = CannyEdge(frame)
    cropped_image = region_of_interest(canny)
    rho = 2
    theta = np.pi / 180
    threshold = 100
    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Image", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()