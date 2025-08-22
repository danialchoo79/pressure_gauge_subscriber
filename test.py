import cv2
import numpy as np

image_path = "received_images/2025-08-22/cam2/cam2_2025-08-22_13-43-24.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Cannot load image: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3,3), 1.0)

edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

_, binary = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(
        image=binary,
        rho=1,
        theta = np.pi/180,
        threshold=50,
        minLineLength=30,
        maxLineGap=5
    )

img_lines = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1,y1), (x2,y2), (0,255,0),2)    


cv2.imshow("Binary for Hough", binary)
cv2.imshow("Hough Lines", img_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()



# _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# cv2.imshow("Thresholded Image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()