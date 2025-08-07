"""
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
"""

from PIL import Image
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import arrow
import sys
import os
import math

def read_image(imagename):

    """
        This reads an Image using Open CV
    """

    # Read Image from a file
    image = cv2.imread(f'{imagename}.png')

    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def str_to_bool(v):

    """
        Converts String to Bool for Passing Arguments in Terminal
    """

    v = v.lower()
    print(f"V value is: {v}")

    if v in ('true', '1', 'yes'):
        return True
    elif v in ('false', '0', 'no'):
        return False
    else:
        raise ValueError(f"Invalid Boolean Value {v}")

def save_image(line_key, pump_key, output_path):

    """
        This opens up an image from a File Path, Rotates and Crops Image

        Prepares for Image Processing Later On

        returns Image after pre processing
    """

    try:
        image_filename_ori = "{}_{}_ori.png".format(line_key, pump_key)
        image_filename_adj = "{}_{}_adj.png".format(line_key, pump_key)
        image_filename_cropped = "{}_{}_cropped.png".format(line_key, pump_key)
        image_path_ori = output_path + "/" + image_filename_ori
        image_path_adj = output_path + "/" + image_filename_adj
        image_path_cropped = output_path + "/" + image_filename_cropped
        
        # Read, Rotate, Crop
        img_obj = cv2.imread(image_path_ori)
        img_ccw_90 = cv2.rotate(img_obj, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(image_path_adj, img_ccw_90)

        img_cropped = cv2.imread(image_path_adj)
        #  X:X Y:Y
        cropImg = img_cropped[31:550, 200:750]

        #  YY
        # cropImg[340:380, 205:315] = (255, 255, 255)
        cropImg[320:380, 205:315] = (255, 255, 255)
        cv2.imwrite(image_path_cropped, cropImg)

        return image_path_cropped
    except Exception as e:
        raise e

def prepare_image_for_hough(image_path, use_sobel=False, show_steps=False, save_image = False):

    """ 
        This performs either (1) Sobel Operation OR (2) Canny Operation

        Before that it does Gray Scaling and Gaussian Blur

        This is to ensure that Hough Transform will be more Accurate

        returns edges detected by Sobel or Canny
    """
    
    # Load Image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 0

    # Use Gaussian Blur to Reduce Noise
    blurred = cv2.GaussianBlur(gray, (5,5), sigmaX=1.4)

    # Edge Detection
    if use_sobel:
        # Sobel Gradients
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        edges = sobel
        edge_used = "sobel"
    else:
        edges = cv2.Canny(blurred,8,43)
        edge_used = "canny"

        # Thresholding
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    if show_steps:
        cv2.imshow("Original", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Edges (Canny)" if not use_sobel else "Edges (Sobel)", edges)
        cv2.imshow("Thresholded",thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_image:
        image_path_edge = "201_PRESSURE_PUMP_1_{}.png".format(edge_used)

        directory = os.path.dirname(image_path)
        output_path = directory + "/" + image_path_edge

        print(output_path)

        cv2.imwrite(output_path, edges)

    return edges # Return Preprocessed Edge Map for Hough Lines

def avg_circles(circles, b):

    """ 
        This averages all circles to find the best circle to represent the Gauge 

        It finds the centroid of the best representative circle.   
    """
    
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        # optional d- average for multiple circles (can happen when a gauge is at a slight angle)
        # For each circle find the value of x, y, r then compute the average values
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x / (b))
    avg_y = int(avg_y / (b))
    avg_r = int(avg_r / (b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    
    """
        Euclidean Distance is used to calculate straight-line distance in 2D space
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def parallel_offset(x1, y1, x2, y2, x, y, img):

    """
        Offsets the Hough Line Parallel-ly to be more representative of Needle
    """

      # ----------- Calculating Offset -----------

    # Offset from center
    offset_d = 4

    # Length of Coordinates
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx,dy)
   
    # Midpoint of Line
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Vector from Gauge Center
    vx = mx - x
    vy = my - y

    # Cross Product to Determine Side
    cross = (dx * vy)- (dy * vx)

    if cross > 0:
        offset_dir = -1
    else:
        offset_dir = +1

    # Perpendicular Offset Vector
    offset_x = offset_dir * (-dy / length) * offset_d
    offset_y = offset_dir * (dx / length) * offset_d

    # Apply Offset to get Centerline
    x1_c = int(x1 + offset_x)
    y1_c = int(y1 + offset_y)
    x2_c = int(x2 + offset_x)
    y2_c = int(y2 + offset_y)

    # ----------- Generate the Lines -----------

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(img, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)

    return x1_c, x2_c, y1_c, y2_c

def check_intersect_lines(x1, y1, x2, y2, x3, y3, x4, y4, angle_thresh=2):

    """
        Checks intersect lines, should they be too close or nearly parallel.
        This means that the detection algorithm detects the same edge.
    """

    # Vector 1
    dx1 = x2 - x1
    dy1 = y2 - y1

    # Vector 2
    dx2 = x4 - x3
    dy2 = y4 - y3

    # Dot Product and Magnitude
    dot = dx1*dx2 + dy1*dy2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)

    # Avoid Division by 0
    if mag1 == 0 or mag2 == 0:
        raise ValueError("One of the lines has zero length")
    
    # Capture Angle in Radians
    cos_theta = dot / (mag1 * mag2)
    angle_rad = math.acos(cos_theta)

    # Convert in degrees
    angle_deg = math.degrees(angle_rad)

    if angle_thresh< angle_deg < (180 - angle_thresh):
        print("The lines are okay.")
        return angle_deg
    
    else:
        print("The lines are too parallel to one another")
        return angle_deg

def intersect_hough_for_needle(lines,edges,img, cx, cy):

    """
        Gets the 2 Hough Lines on 2 Sides of the Needle to Intersect
        and marks a line from the point to centre gauge coordinates

        Args:
            lines: lines generated hough line transform
            edges: thresholded image needed before finding needle 
            img: image where the cv operations are run on 
            cx: x-coordinate for center of gauge
            cy: y-coordinate for center of gauge
        
        Returns:
            -
    """

    def line_angle(x1, y1, x2, y2):

        """
            Compute the angle between two lines based on two pair of endpoints
        """

        return np.degrees(np.arctan2(y2-y1, x2-x1)) # range 0-180
    
    def intersect(p1, p2, p3, p4):

        """
            Compute determinant to determine left or right orientation from a vector
        """

        x_in = y_in = None
        x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = -1

        def det(a,b):
            return a[0]*b[1] - a[1]*b[0]
        
        xdiff = (p1[0] - p2[0], p3[0] - p4[0])
        ydiff = (p1[1] - p2[1], p3[1] - p4[1])

        div = det(xdiff, ydiff)

        if div == 0:
            # Lines are Parallel
            return None
        
        d = (det(p1,p2), det(p3,p4)) # Orientation & Position of two Endpoints
        x = det(d, xdiff) / div      # x - coordinate of intersection
        y = det(d, ydiff) / div      # y - coordinate of instersection

        return(int(x), int(y))

    # Filter and sort lines by Length
    if lines is not None:
        lines = [line[0] for line in lines]

    # Try all line pairs, look for nearly Parallel
    found = False

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]

            # Step 1: Calculate Angle
            angle1 = line_angle(x1, y1, x2, y2)
            angle2 = line_angle(x3, y3, x4, y4)

            # Step 2: Check for nearly parallel
            if abs(angle1 - angle2) < 10:
                pt = intersect((x1,y1), (x2,y2), (x3,y3), (x4,y4))

                print(f"Lines Coordinates are x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, x3: {x3}, y3: {y3}, x4: {x4}, y4: {y4}")

                try:
                    # Get the intersection coordinates
                    x_in, y_in = pt
                    print(f"x-intercept is {x_in} and y-intercept is {y_in}")
                except:
                    print("x-intercept, y-intercept cant find, restarting algorithm")
                    python = sys.executable
                    os.execv(python, [python] + sys.argv)

                if pt is not None:
                    # Step 3: Draw Edge Lines
                    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.line(img, (x3,y3), (x4,y4), (0,255,0), 2)

                    # Step 4: Draw Needle Center from intersection to gauge center
                    cv2.line(img, pt, (cx,cy), (0,0,255), 2)
                    cv2.circle(img, pt, 5, (255,0,0), -1)
                    found = True
                    break
            else:
                print("Cannot get coordinates, restarting algorithm")
                python = sys.executable
                os.execv(python, [python] + sys.argv)

        if found:
            break

          # Show Result
        cv2.imshow("Grayscale",edges)
        cv2.imshow("Needle Center", img)

    return x_in, y_in, x1, y1, x2, y2, x3, y3, x4, y4

def calibrate_gauge(filepath, output_path, line_key, pump_key):
    """

    Operations:
        (1) Find Circle (center, radius)
        (2) Draw Reference Lines to Read Angles
        (3) Ask User for Gauge Extremes
        (4) Ask for Units
        (5) Return Calibration Info 
                (min_angle, min_value)
                (max_angle, max_value)

    This function should be run using a test image in order to calibrate the range available to the dial as well as the
    units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
    (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
    as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
    position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
    the gauge is linear (as most probably are).
    It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
    and the units (as a string)...
    """

    # Reizes Image to (271, 262) using (4x4) Square
    image = cv2.imread(filepath)
    img = cv2.resize(image, dsize=(271, 262), interpolation=cv2.INTER_CUBIC)

    # --TEST CODE--
    # img.save('images\guage-4-271.jpeg')
    # img = cv2.imread(file_type)
    # cv2.imshow('image', img)
    # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # Get height, width and convert it to gray
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Circles
    # Restricting the search from 35-48% of Image Height gives the possible radii gives fairly good results across different samples.
    # These are pixel values which correspond to the possible radii search range.30,47 for
    
    circles = cv2.HoughCircles(
        gray,                   # Image
        cv2.HOUGH_GRADIENT,     # Method (Gradient Information of Edges)
        1.5,                    # Accumulator Resolution (accumulates votes based on edge points). Higher = finer resolution, vice versa.                  
        5,                      # Minimum Distance Between Centroids in Pixels
        np.array([]),           # Empty Array (Unused Parameter)
        100,                    # Canny Edge Threshold to detect Strong Edge
        50,                     # Threshold votes for Center Detection
        int(height * 0.25),     # Minimum Radius
        int(height * 0.46),     # Maximum Radius
    )
    
    # Average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape                 # (1) Get the no.of image in batch, (2) no. of circles detected, (3) x,y,radius
    x, y, r = avg_circles(circles, b)       # Get the average centroid

    # x_corr_offset = 4
    # x = x + x_corr_offset

    print(f"x-centroid: {x}, y-centroid: {y}, radius: {r}")

    # Draw Center and Circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # Draws Red Circle with thickness 3, 
     
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # Draws Green Center of Circle

    # For Testing, Output Circles on Image
    image_filename_circle = output_path + "/{}_{}_circle.png".format(line_key, pump_key)
    cv2.imwrite(image_filename_circle, img)

    # For calibration, plot lines from center going out at every 10 degrees and add marker
    # For i from 0 to 36 (every 10 deg)

    """
    Goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    """

    separation = 10.0  # In degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))  # Set empty arrays with 36 rows and 2 colums
    p2 = np.zeros((interval, 2))  
    p_text = np.zeros((interval, 2))


    # This loop builds a set of (x, y) points along a circular arc or full circle, 
    # and stores them in the p1 array.
    # Note: 0.9 is used as the area of interest is smaller than the full radius

    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0:  # x-coordinates
                p1[i][j] = x + 0.9 * r * np.cos(
                    separation * i * 3.14 / 180
                )  
            else: # y-coordinates
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)

    # Text alignments
    text_offset_x = 10
    text_offset_y = 5

    # Plot Values outside the Angles
    # Plot the Text Labels
    # np.cos() provides the Horizontal Component, 
    # np.sin() provides the Vertical Component

    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0: # x-coordinates
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = (
                    x
                    - text_offset_x
                    + 1.2 * r * np.cos((separation) * (i + 9) * 3.14 / 180)
                )  # point for text labels, i+9 rotates the labels by 90 degrees

            else: # y-coordinates
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = (
                    y
                    + text_offset_y
                    + 1.2 * r * np.sin((separation) * (i + 9) * 3.14 / 180)
                )  # point for text labels, i+9 rotates the labels by 90 degrees

    # Add the Lines and Labels to the Image
    for i in range(0, interval):

        # Plot the Dial Lines
        cv2.line(
            img,
            (int(p1[i][0]), int(p1[i][1])), # pt1: Starting Pt
            (int(p2[i][0]), int(p2[i][1])), # pt2: Ending Pt
            (0, 255, 0),                    # Green Color
            2,                              # Thickness
        )

        cv2.putText(
            img,
            "%s" % (int(i * separation)),            # string of i * (0-35)
            (int(p_text[i][0]), int(p_text[i][1])),  # position of label 
            cv2.FONT_HERSHEY_SIMPLEX,                # font
            0.3,                                     # font scale
            (0, 0, 0),                               # black
            1,                                       # thickness
            cv2.LINE_AA,                             # anti-aliasing (smooth)
        )

    # Store image
    image_filename_final = output_path + "/{}_{}_final.png".format(line_key, pump_key)
    cv2.imwrite(image_filename_final, img)

    # Min & Max Angles / Value
    min_angle = 47  # Lowest angle for gauge
    max_angle = 315 # Highest angle for gauge
    min_value = 0   # Minimum value for gauge
    max_value = 4   # Maximum value for gauge
    units = "psi"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def get_current_value(
    img,
    min_angle,
    max_angle,
    min_value,
    max_value,
    x,
    y,
    r,
    output_path,
    line_key,
    pump_key,
):
    # TEST CODE:
    # img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray scaling

    # Set threshold and maxValue
    thresh = 50
    maxValue = 250

    # Apply thresholding which helps for finding lines
    # Returns, th: threshold used & dst2: thresholded image
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY)

    cv2.imshow("dst2",dst2)

    # Store thresholded image
    image_filename_grey = output_path + "/{}_{}_grayscale.png".format(
        line_key, pump_key
    )
    cv2.imwrite(image_filename_grey, dst2)

    # Line Detection
    # Note: # rho is set to 3 to detect more lines, 
    # easier to get more then filter them out later
    minLineLength = 10

    # Note: Each line is shaped [[x1,y1,x2,y2]]
    lines = cv2.HoughLinesP(
        image=dst2,                     # Single-Channel Binary Image
        rho=3,                          # Distance Resolution of Accumulator (Pixels)
        theta=np.pi / 180,              # Angle Resolution of Accumulator (Radians). 1 now.
        threshold=15,                  # Minimum number of intersections in Hough space 
        minLineLength=minLineLength,    # Minimum length accepted
        maxLineGap=5,                   # Maximum gap between lines to be considered single line
    )  

    # TEST CODE: show all found lines
    # for i in range(0, len(lines)):
    #   for x1, y1, x2, y2 in lines[i]:
    #      cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #      cv2.imwrite(output_path + "/" +'lines.jpg', img)

    # Remove all Lines outside a Given Radius
    # Note: diff1 controls length of line within Radius
    # Note: diff2 controls how close line is radius of circle

    final_line_list = []
    # print "radius: %s" %r
    # diff1LowerBound = 0.15
    diff1LowerBound = 0.00  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    # diff1UpperBound = 0.25
    diff1UpperBound = 0.50
    # diff2LowerBound = 0.5
    diff2LowerBound = 0.5  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    # diff2UpperBound = 1.0
    diff2UpperBound = 1

    line_length_lst = []
    line_pos_lst = []
    
    # Find the difference of lengths from center coordinates (x,y)
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier

            # Swap diff1 and diff2 should diff1 > diff2
            if diff1 > diff2:
                temp = diff1
                diff1 = diff2
                diff2 = temp

            # Check if line is within an acceptable range
            if (
                (diff1 < diff1UpperBound * r)
                and (diff1 > diff1LowerBound * r)
                and (diff2 < diff2UpperBound * r)
            ) and (diff2 > diff2LowerBound * r):
                # if (((diff1<160) and (diff1>140) and (diff2<190)) and (diff2>170)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                line_length_lst.append(line_length)
                # add to final list
                # final_line_list.append([x1, y1, x2, y2])
                line_pos_lst.append([x1, y1, x2, y2])

    # Jerrold
    print(line_length_lst)
    final_indx = line_length_lst.index(max(line_length_lst)) # Finds the position of the longest line
    final_line_list.append(line_pos_lst[final_indx]) # Appends final_line_list with the longest line

    # TEST CODE
    # testing only, show all lines after filtering
    # for i in range(0,len(final_line_list)):
    #     x1 = final_line_list[i][0]
    #     y1 = final_line_list[i][1]
    #     x2 = final_line_list[i][2]
    #     y2 = final_line_list[i][3]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.imwrite(output_path + "/" +'filter_lines.jpg', img)

    # Plots the line on the image
    # Note: It assumes the first line is the Best One
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]

    x1, y1 = x, y 

    centroid_offset = 2
    x = x + centroid_offset
    y = y + centroid_offset
    x2, y2, x1corr, y1corr, x2corr, y2corr, x3corr, y3corr, x4corr, y4corr = intersect_hough_for_needle(lines,dst2,img, x, y)
    
    angle = check_intersect_lines(x1corr, y1corr, x2corr, y2corr, x3corr, y3corr, x4corr, y4corr, angle_thresh=2)
    print(angle)

    angle_threshold = 3

    if angle <= angle_threshold:
        print("Same edge, restarting algorithm")
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    # # UNCOMMENT TO USE PARALLEL OFFSET
    # x1,x2,y1,y2 = parallel_offset(x1, y1, x2, y2, x, y, img)

    ## UNCOMMENT TO USE MANUAL METHOD
    # cv2.line(img, (x1+4, y1+4), (x2+4, y2+4), (0, 255, 0), 2)

    # TEST CODE
    # for testing purposes, show the line overlayed on the original image
    # cv2.imwrite('gauge-1-test.jpg', img)

    # Draw the needle on the image
    image_filename_needle = output_path + "/{}_{}_needle.png".format(line_key, pump_key)
    cv2.imwrite(image_filename_needle, img)

    # Find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    print(x1)
    print(y1)
    print(x2)
    print(y2)
    print(dist_pt_0)

    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    print(dist_pt_1)

    # Angle Calculation
    if dist_pt_0 > dist_pt_1:
        x_angle = x1 - x    # x-offset
        print(x1)
        print(x)
        print(x_angle)
        y_angle = y - y1    # y-offset
        print(y_angle)
    else:
        x_angle = x2 - x
        y_angle = y - y2

    # Take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    print(res)
    # np.rad2deg(res) #coverts to degrees

    # print x_angle
    # print y_angle
    # print res
    # print np.rad2deg(res)

    # Compensation as arc tan does not handle quadrants properly
    # Note: these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res

    # TEST CODE
    # print final_angle

    # Linear Mapping to get the Value
    old_min = float(min_angle) # 47
    old_max = float(max_angle) # 315

    new_min = float(min_value) # 0
    new_max = float(max_value) # 4

    old_value = final_angle

    old_range = old_max - old_min # 268
    new_range = new_max - new_min # 4
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    print(new_value)
    return new_value

def main(
    filepath,
    output_path,
    line_key,
    pump_key,
):
    try:
        # name the calibration image of your gauge 'gauge-#.jpg', for example 'gauge-5.jpg'.  It's written this way so you can easily try multiple images
        min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(
            filepath, output_path, line_key, pump_key
        )
        # feed an image (or frame) to get the current value, based on
        # the calibration, by default uses same image as calibration
        # img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))

        # img = cv2.imread('images\gauge-%s.%s' % (gauge_number, file_type))
        images = np.array(Image.open(filepath))

        img = cv2.resize(images, dsize=(271, 262), interpolation=cv2.INTER_CUBIC)

        val = get_current_value(
            img,
            min_angle,
            max_angle,
            min_value,
            max_value,
            x,
            y,
            r,
            output_path,
            line_key,
            pump_key,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        format_float = "{:.2f}".format(val) # 2 significant figures
        print("------result--------")
        print(format_float)
        return format_float
    
    except Exception as e:
        raise e

if __name__ == "__main__":

    # # --SAVE_IMAGE--

    # line_key = sys.argv[1]
    # pump_key = sys.argv[2]
    # output_path = sys.argv[3]
    
    # save_image(line_key, pump_key, output_path)
    
    # --HOUGH IMAGE--

    # image_path = sys.argv[1]
    # use_sobel = str_to_bool(sys.argv[2])
    # show_steps = str_to_bool(sys.argv[3])
    # save_images = str_to_bool(sys.argv[4])
    
    # prepare_image_for_hough(image_path, use_sobel, show_steps, save_images)

    # --CALIBRATE GAUGE--

    # calibrate_gauge(r"C:\Users\D121A4527\Desktop\imgprocess2\201_PRESSURE_PUMP_1_sobel.png", r"C:\Users\D121A4527\Desktop\imgprocess2", "201", "PRESSURE_PUMP")

    # -- GET VALUE --

    # -- MAIN --

    filepath = sys.argv[1]
    output_path = sys.argv[2]
    line_key = sys.argv[3]
    pump_key = sys.argv[4]

    main(filepath, output_path, line_key, pump_key)