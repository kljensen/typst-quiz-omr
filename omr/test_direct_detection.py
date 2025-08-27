#!/usr/bin/env python3
"""
Direct test of bracket detection on generated images
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_image(image_path: Path):
    """Analyze image to understand what we're working with"""
    
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\nAnalyzing: {image_path}")
    print(f"Shape: {img.shape}")
    print(f"Pixel range: {gray.min()} - {gray.max()}")
    print(f"Mean pixel value: {gray.mean():.2f}")
    
    # Try different thresholds
    for thresh_val in [127, 150, 180, 200, 220, 240]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        black_pixels = np.sum(binary > 0)
        print(f"Threshold {thresh_val}: {black_pixels} black pixels")
    
    # Try adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    print(f"Adaptive threshold: {np.sum(adaptive > 0)} black pixels")
    
    # Save samples for inspection
    cv2.imwrite(f"sample_original_{image_path.stem}.png", gray[500:1000, 100:600])
    cv2.imwrite(f"sample_binary_{image_path.stem}.png", adaptive[500:1000, 100:600])
    
    # Look for lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        print(f"Found {len(lines)} lines")
        
        # Categorize lines
        horizontal = 0
        vertical = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 150:
                horizontal += 1
            elif 60 < angle < 120:
                vertical += 1
        
        print(f"  Horizontal: {horizontal}, Vertical: {vertical}")
    else:
        print("No lines detected")
    
    # Check for corner patterns
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = np.sum(harris > 0.01 * harris.max())
    print(f"Harris corners: {harris_corners}")
    
    return gray


# Test on different pages
for page_num in [1, 3, 6, 9, 12]:
    page_path = Path(f"tests/generated/test_data_page_{page_num}.png")
    if page_path.exists():
        analyze_image(page_path)
    else:
        print(f"Page {page_num} not found")