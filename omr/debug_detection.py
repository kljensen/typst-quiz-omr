#!/usr/bin/env python3
"""
Debug script to visualize detection steps
"""

import cv2
import numpy as np
from pathlib import Path

def debug_bracket_detection(image_path: Path):
    """Debug corner bracket detection"""
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img.shape}")
    
    # Save original
    cv2.imwrite("debug_original.png", img)
    
    # Binary threshold
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("debug_binary.png", binary)
    
    # Try to find corners using corner detection
    corners = cv2.goodFeaturesToTrack(binary, maxCorners=100, qualityLevel=0.01, minDistance=30)
    
    if corners is not None:
        print(f"Found {len(corners)} corner features")
        corner_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imwrite("debug_corners.png", corner_img)
    
    # Try HoughCircles to see if we can find bubbles
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=25
    )
    
    if circles is not None:
        print(f"Found {len(circles[0])} circles")
        circle_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            cv2.circle(circle_img, (x, y), r, (0, 0, 255), 2)
        
        cv2.imwrite("debug_circles.png", circle_img)
        
        # Find bounding box of circles
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        print(f"Bubble region bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        
        # Draw bounding box
        cv2.rectangle(circle_img, (x_min-20, y_min-20), (x_max+20, y_max+20), (255, 0, 0), 2)
        cv2.imwrite("debug_bubble_region.png", circle_img)
    
    # Look for L-shaped patterns
    # Try edge detection
    edges = cv2.Canny(img, 50, 150)
    cv2.imwrite("debug_edges.png", edges)
    
    # Look for lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        print(f"Found {len(lines)} lines")
        line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("debug_lines.png", line_img)
    
    print("\nDebug images saved:")
    print("  debug_original.png - Original grayscale")
    print("  debug_binary.png - Binary threshold")
    print("  debug_corners.png - Corner features")
    print("  debug_circles.png - Detected circles")
    print("  debug_bubble_region.png - Bubble bounding box")
    print("  debug_edges.png - Edge detection")
    print("  debug_lines.png - Detected lines")


if __name__ == "__main__":
    test_image = Path("tests/generated/test_data_page_1.png")
    if test_image.exists():
        debug_bracket_detection(test_image)
    else:
        print(f"Test image not found: {test_image}")