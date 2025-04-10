import cv2

# Read an image in grayscale mode
gray = cv2.imread("./test.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Laplacian edge detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Apply Canny edge detection with thresholds
canny = cv2.Canny(gray, 100, 200)

# Create and resize windows for displaying images
window_name1 = "gray"
cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name1, 800, 600)

window_name2 = "laplacian"
cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name2, 800, 600)

window_name3 = "canny"
cv2.namedWindow(window_name3, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name3, 800, 600)

# Display the images in the created windows
cv2.imshow(window_name1, gray)
cv2.imshow(window_name2, laplacian)
cv2.imshow(window_name3, canny)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()