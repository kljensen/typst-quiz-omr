#!/usr/bin/env python3
"""
Debug simple bracket detection
"""

import cv2
import numpy as np
from pathlib import Path
from simple_bracket_detector import detect_brackets_morphology, create_l_kernel

def debug_detection(image_path: Path):
    """Debug simple bracket detection"""
    
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Image shape: {img.shape}")
    print(f"Min/Max pixel values: {gray.min()}/{gray.max()}")
    
    # Binarize
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    print(f"Binary non-zero pixels: {np.count_nonzero(binary)}")
    
    # Save binary for inspection
    cv2.imwrite("debug_binary.png", binary)
    print("Binary image saved to debug_binary.png")
    
    # Test detection
    corners = detect_brackets_morphology(img)
    print(f"\nDetected {len(corners)} corners:")
    
    corner_counts = {}
    for corner in corners[:20]:  # Show first 20
        corner_counts[corner.corner_type] = corner_counts.get(corner.corner_type, 0) + 1
        print(f"  {corner.corner_type} at {corner.position} (conf: {corner.confidence:.2f})")
    
    print(f"\nCorner type counts: {corner_counts}")
    
    # Visualize
    vis = img.copy()
    colors = {
        'top-left': (255, 0, 0),      # Blue
        'top-right': (0, 255, 0),     # Green  
        'bottom-left': (0, 0, 255),   # Red
        'bottom-right': (255, 255, 0) # Yellow
    }
    
    for corner in corners:
        color = colors.get(corner.corner_type, (255, 255, 255))
        cv2.circle(vis, corner.position, 10, color, 2)
    
    cv2.imwrite("debug_corners_simple.png", vis)
    print(f"Visualization saved to debug_corners_simple.png")
    
    # Try to detect specific regions
    # Look for concentrated areas of black pixels
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    density = cv2.filter2D(binary.astype(np.float32), -1, kernel)
    
    # Find local maxima in density
    _, density_thresh = cv2.threshold(density, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug_density.png", density_thresh.astype(np.uint8))
    print("Density map saved to debug_density.png")
    
    # Try edge detection to find brackets
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite("debug_edges_canny.png", edges)
    print("Edge detection saved to debug_edges_canny.png")

if __name__ == "__main__":
    test_image = Path("tests/generated/test_data_page_3.png")
    if test_image.exists():
        debug_detection(test_image)
    else:
        print(f"Test image not found: {test_image}")