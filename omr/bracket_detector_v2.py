#!/usr/bin/env python3
"""
Improved bracket detection using connected component analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BracketCorner:
    """Represents a detected bracket corner"""
    position: Tuple[int, int]  # (x, y)
    corner_type: str  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    confidence: float


def find_l_shaped_components(binary: np.ndarray, min_size: int = 20, max_size: int = 200) -> List[Tuple[int, int, int, int]]:
    """
    Find L-shaped connected components in binary image.
    Returns list of bounding boxes (x, y, w, h).
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    l_shaped_components = []
    
    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Check size constraints
        if min_size <= w <= max_size and min_size <= h <= max_size:
            # Extract component
            component = (labels == i).astype(np.uint8) * 255
            component_roi = component[y:y+h, x:x+w]
            
            # Check if it's L-shaped
            if is_component_l_shaped(component_roi):
                l_shaped_components.append((x, y, w, h))
    
    return l_shaped_components


def is_component_l_shaped(component: np.ndarray, threshold: float = 0.2) -> bool:
    """
    Check if a connected component is L-shaped.
    An L-shape should have pixels mainly in two perpendicular lines.
    """
    h, w = component.shape
    
    # Check if aspect ratio is roughly square (L-shapes are typically square-ish)
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    # Count pixels in different regions
    total_pixels = np.sum(component > 0)
    if total_pixels < 10:  # Too small to be a bracket
        return False
    
    # Check for L-pattern: should have pixels in corner regions
    # Divide into quadrants
    mid_h, mid_w = h // 2, w // 2
    
    # For an L-shape, we expect pixels in:
    # - Either top row/left column (top-left L)
    # - Or top row/right column (top-right L)
    # - Or bottom row/left column (bottom-left L)
    # - Or bottom row/right column (bottom-right L)
    
    # Check top-left L pattern
    top_strip = component[:max(1, h//4), :]
    left_strip = component[:, :max(1, w//4)]
    top_left_pixels = np.sum(top_strip > 0) + np.sum(left_strip > 0)
    
    # Check top-right L pattern
    right_strip = component[:, -max(1, w//4):]
    top_right_pixels = np.sum(top_strip > 0) + np.sum(right_strip > 0)
    
    # Check bottom-left L pattern
    bottom_strip = component[-max(1, h//4):, :]
    bottom_left_pixels = np.sum(bottom_strip > 0) + np.sum(left_strip > 0)
    
    # Check bottom-right L pattern
    bottom_right_pixels = np.sum(bottom_strip > 0) + np.sum(right_strip > 0)
    
    # At least one pattern should have significant pixels
    max_pattern_pixels = max(top_left_pixels, top_right_pixels, bottom_left_pixels, bottom_right_pixels)
    
    return max_pattern_pixels / total_pixels > threshold


def detect_brackets_v2(img: np.ndarray) -> List[BracketCorner]:
    """
    Detect L-shaped brackets using connected component analysis.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find L-shaped components
    l_components = find_l_shaped_components(binary)
    
    detected_corners = []
    
    for x, y, w, h in l_components:
        # Determine corner type based on position in image
        img_h, img_w = gray.shape
        cx, cy = x + w // 2, y + h // 2
        
        if cx < img_w // 2 and cy < img_h // 2:
            corner_type = 'top-left'
        elif cx >= img_w // 2 and cy < img_h // 2:
            corner_type = 'top-right'
        elif cx < img_w // 2 and cy >= img_h // 2:
            corner_type = 'bottom-left'
        else:
            corner_type = 'bottom-right'
        
        # Find the actual corner point within the component
        component = binary[y:y+h, x:x+w]
        corner_point = find_corner_point(component, corner_type)
        
        if corner_point is not None:
            actual_x = x + corner_point[0]
            actual_y = y + corner_point[1]
            
            detected_corners.append(BracketCorner(
                position=(actual_x, actual_y),
                corner_type=corner_type,
                confidence=0.9
            ))
    
    return detected_corners


def find_corner_point(component: np.ndarray, corner_type: str) -> Optional[Tuple[int, int]]:
    """Find the actual corner point within an L-shaped component"""
    h, w = component.shape
    
    # Find the corner based on type
    if corner_type == 'top-left':
        # Look for the point where horizontal and vertical lines meet
        for y in range(h // 2):
            for x in range(w // 2):
                if component[y, x] > 0:
                    return (x, y)
    elif corner_type == 'top-right':
        for y in range(h // 2):
            for x in range(w // 2, w):
                if component[y, x] > 0:
                    return (x, y)
    elif corner_type == 'bottom-left':
        for y in range(h // 2, h):
            for x in range(w // 2):
                if component[y, x] > 0:
                    return (x, y)
    elif corner_type == 'bottom-right':
        for y in range(h // 2, h):
            for x in range(w // 2, w):
                if component[y, x] > 0:
                    return (x, y)
    
    # Fallback to centroid
    points = np.where(component > 0)
    if len(points[0]) > 0:
        return (int(np.mean(points[1])), int(np.mean(points[0])))
    
    return None


def find_bubble_region_v2(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bubble region defined by L-shaped brackets.
    Returns (x, y, width, height) or None if not found.
    """
    corners = detect_brackets_v2(img)
    
    if len(corners) < 4:
        print(f"Only found {len(corners)} corners")
        return None
    
    # Group corners by type
    corner_dict = {}
    for corner in corners:
        if corner.corner_type not in corner_dict:
            corner_dict[corner.corner_type] = []
        corner_dict[corner.corner_type].append(corner)
    
    # Get one corner of each type
    tl = corner_dict.get('top-left', [None])[0]
    tr = corner_dict.get('top-right', [None])[0]
    bl = corner_dict.get('bottom-left', [None])[0]
    br = corner_dict.get('bottom-right', [None])[0]
    
    if not all([tl, tr, bl, br]):
        print(f"Missing corner types: TL={tl is not None}, TR={tr is not None}, BL={bl is not None}, BR={br is not None}")
        return None
    
    # Calculate bounding box with some margin
    margin = 10
    x_min = min(tl.position[0], bl.position[0]) + margin
    x_max = max(tr.position[0], br.position[0]) - margin
    y_min = min(tl.position[1], tr.position[1]) + margin
    y_max = max(bl.position[1], br.position[1]) - margin
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = Path("tests/generated/test_data_page_1.png")
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    img = cv2.imread(str(image_path))
    
    # Debug: show detected corners
    corners = detect_brackets_v2(img)
    print(f"Detected {len(corners)} corners:")
    for corner in corners:
        print(f"  {corner.corner_type} at {corner.position}")
    
    region = find_bubble_region_v2(img)
    
    if region:
        print(f"\nFound bubble region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Draw the detected region
        x, y, w, h = region
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Also draw the detected corners
        colors = {
            'top-left': (255, 0, 0),      # Blue
            'top-right': (0, 255, 0),     # Green  
            'bottom-left': (0, 0, 255),   # Red
            'bottom-right': (255, 255, 0) # Yellow
        }
        for corner in corners:
            color = colors.get(corner.corner_type, (255, 255, 255))
            cv2.circle(vis, corner.position, 15, color, 3)
            cv2.putText(vis, corner.corner_type[:2].upper(), 
                       (corner.position[0] - 20, corner.position[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imwrite("detected_region_v2.png", vis)
        print("Visualization saved to detected_region_v2.png")
    else:
        print("\nCould not detect bubble region")