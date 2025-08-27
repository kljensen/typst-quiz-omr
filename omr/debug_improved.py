#!/usr/bin/env python3
"""
Debug improved bracket detection
"""

import cv2
import numpy as np
from pathlib import Path
from improved_detector import ImprovedBracketDetector

def debug_detection(image_path: Path):
    """Debug improved bracket detection"""
    
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Min/Max values: {img.min()}/{img.max()}")
    
    detector = ImprovedBracketDetector()
    
    # Test each method individually
    print("\n1. Testing Harris corners:")
    harris_corners = detector.detect_harris_corners(gray)
    print(f"   Found {len(harris_corners)} Harris corners")
    for c in harris_corners[:5]:  # Show first 5
        print(f"   - {c.corner_type} at {c.position}")
    
    print("\n2. Testing template matching:")
    template_corners = detector.detect_template_corners(gray)
    print(f"   Found {len(template_corners)} template matches")
    for c in template_corners[:5]:
        print(f"   - {c.corner_type} at {c.position} (conf: {c.confidence:.2f})")
    
    print("\n3. Testing contour detection:")
    contour_corners = detector.detect_contour_corners(gray)
    print(f"   Found {len(contour_corners)} contour corners")
    for c in contour_corners[:5]:
        print(f"   - {c.corner_type} at {c.position}")
    
    print("\n4. Testing line intersections:")
    line_corners = detector.detect_line_intersection_corners(gray)
    print(f"   Found {len(line_corners)} line corners")
    for c in line_corners[:5]:
        print(f"   - {c.corner_type} at {c.position}")
    
    # Visualize detections
    vis = img.copy()
    all_corners = harris_corners + template_corners + contour_corners + line_corners
    
    colors = {
        'harris': (255, 0, 0),      # Blue
        'template': (0, 255, 0),    # Green  
        'contour': (0, 0, 255),     # Red
        'lines': (255, 255, 0)      # Yellow
    }
    
    for corner in all_corners:
        color = colors.get(corner.method, (255, 255, 255))
        cv2.circle(vis, corner.position, 10, color, 2)
        cv2.putText(vis, corner.corner_type[:2], 
                   (corner.position[0] - 20, corner.position[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite("debug_corners_detected.png", vis)
    print(f"\nVisualization saved to debug_corners_detected.png")
    
    # Check if we can find the region
    final_corners = detector.merge_corners(all_corners)
    print(f"\nAfter merging: {len(final_corners)} corners")
    
    if len(final_corners) >= 4:
        print("SUCCESS: Found enough corners for region detection!")
    else:
        print("FAILED: Not enough corners for region detection")
        
        # Try adjusting parameters
        print("\nTrying with adjusted parameters...")
        
        # Test with different scales
        for scale in [10, 15, 25, 35, 45, 60]:
            templates = detector.generate_l_templates(scale)
            matches = 0
            for template, _ in templates:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.6  # Lower threshold
                locations = np.where(result >= threshold)
                matches += len(locations[0])
            if matches > 0:
                print(f"   Scale {scale}: {matches} matches found")


if __name__ == "__main__":
    test_image = Path("tests/generated/test_data_page_1.png")
    if not test_image.exists():
        # Try from omr directory
        test_image = Path("../tests/generated/test_data_page_1.png")
    
    if test_image.exists():
        debug_detection(test_image)
    else:
        print(f"Test image not found: {test_image}")