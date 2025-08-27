#!/usr/bin/env python3
"""
Final bracket detection approach - focus on actual L-brackets
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


def detect_brackets_final(img: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], List[BracketCorner]]:
    """
    Detect the bubble region by finding the four L-shaped brackets.
    Returns (region, corners) where region is (x, y, width, height) or None.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    
    # Use adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Define search regions for each corner (smaller regions for efficiency)
    margin = 100  # Search within this margin from edges
    search_size = 400  # Size of search region
    
    corners = []
    
    # Top-left corner
    tl_region = binary[margin:margin+search_size, margin:margin+search_size]
    tl_corner = find_l_bracket_in_region(tl_region, 'top-left')
    if tl_corner:
        tl_corner.position = (tl_corner.position[0] + margin, tl_corner.position[1] + margin)
        corners.append(tl_corner)
    
    # Top-right corner
    tr_region = binary[margin:margin+search_size, w-margin-search_size:w-margin]
    tr_corner = find_l_bracket_in_region(tr_region, 'top-right')
    if tr_corner:
        tr_corner.position = (tr_corner.position[0] + w - margin - search_size, tr_corner.position[1] + margin)
        corners.append(tr_corner)
    
    # Bottom-left corner
    bl_region = binary[h-margin-search_size:h-margin, margin:margin+search_size]
    bl_corner = find_l_bracket_in_region(bl_region, 'bottom-left')
    if bl_corner:
        bl_corner.position = (bl_corner.position[0] + margin, bl_corner.position[1] + h - margin - search_size)
        corners.append(bl_corner)
    
    # Bottom-right corner
    br_region = binary[h-margin-search_size:h-margin, w-margin-search_size:w-margin]
    br_corner = find_l_bracket_in_region(br_region, 'bottom-right')
    if br_corner:
        br_corner.position = (br_corner.position[0] + w - margin - search_size, br_corner.position[1] + h - margin - search_size)
        corners.append(br_corner)
    
    # Calculate region if all 4 corners found
    region = None
    if len(corners) == 4:
        # Sort corners by type
        corner_dict = {c.corner_type: c for c in corners}
        
        tl = corner_dict.get('top-left')
        tr = corner_dict.get('top-right')
        bl = corner_dict.get('bottom-left')
        br = corner_dict.get('bottom-right')
        
        if all([tl, tr, bl, br]):
            # Calculate bounding box with inset
            inset = 20  # Move slightly inside the brackets
            x_min = max(tl.position[0], bl.position[0]) + inset
            x_max = min(tr.position[0], br.position[0]) - inset
            y_min = max(tl.position[1], tr.position[1]) + inset
            y_max = min(bl.position[1], br.position[1]) - inset
            
            if x_max > x_min and y_max > y_min:
                region = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return region, corners


def find_l_bracket_in_region(region: np.ndarray, corner_type: str) -> Optional[BracketCorner]:
    """
    Find an L-shaped bracket in a specific region.
    Uses template matching with generated L-templates.
    """
    if region.size == 0:
        return None
    
    # Generate L-shaped template based on corner type
    template = generate_l_template(60, corner_type)
    
    # Match template
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If confidence is high enough, we found a bracket
    if max_val > 0.5:  # Threshold for match confidence
        # Adjust position based on corner type
        x, y = max_loc
        
        # Find the actual corner point of the L
        if corner_type == 'top-left':
            pass  # Already at corner
        elif corner_type == 'top-right':
            x += template.shape[1]  # Move to right edge
        elif corner_type == 'bottom-left':
            y += template.shape[0]  # Move to bottom edge
        elif corner_type == 'bottom-right':
            x += template.shape[1]  # Move to right edge
            y += template.shape[0]  # Move to bottom edge
        
        return BracketCorner(
            position=(x, y),
            corner_type=corner_type,
            confidence=max_val
        )
    
    # Fallback: look for perpendicular lines
    lines_corner = find_perpendicular_lines(region, corner_type)
    if lines_corner:
        return lines_corner
    
    return None


def generate_l_template(size: int, corner_type: str) -> np.ndarray:
    """Generate an L-shaped template for matching"""
    template = np.zeros((size, size), dtype=np.uint8)
    thickness = 6  # Line thickness in pixels
    length = size * 2 // 3  # Length of each arm
    
    if corner_type == 'top-left':
        # Horizontal line from left
        template[:thickness, :length] = 255
        # Vertical line from top
        template[:length, :thickness] = 255
    elif corner_type == 'top-right':
        # Horizontal line from right
        template[:thickness, -length:] = 255
        # Vertical line from top
        template[:length, -thickness:] = 255
    elif corner_type == 'bottom-left':
        # Horizontal line from left
        template[-thickness:, :length] = 255
        # Vertical line from bottom
        template[-length:, :thickness] = 255
    elif corner_type == 'bottom-right':
        # Horizontal line from right
        template[-thickness:, -length:] = 255
        # Vertical line from bottom
        template[-length:, -thickness:] = 255
    
    return template


def find_perpendicular_lines(region: np.ndarray, corner_type: str) -> Optional[BracketCorner]:
    """
    Find perpendicular lines that form an L-bracket.
    """
    # Detect lines using Hough transform
    edges = cv2.Canny(region, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        return None
    
    # Find horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:  # Horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        elif 60 < angle < 120:  # Vertical
            vertical_lines.append((x1, y1, x2, y2))
    
    # Check if we have both types of lines
    if not horizontal_lines or not vertical_lines:
        return None
    
    # Find intersection point based on corner type
    h, w = region.shape
    
    if corner_type == 'top-left':
        # Look for lines near top-left
        for h_line in horizontal_lines:
            if h_line[1] < h // 3:  # Top third
                for v_line in vertical_lines:
                    if v_line[0] < w // 3:  # Left third
                        return BracketCorner(
                            position=(v_line[0], h_line[1]),
                            corner_type=corner_type,
                            confidence=0.7
                        )
    elif corner_type == 'top-right':
        # Look for lines near top-right
        for h_line in horizontal_lines:
            if h_line[1] < h // 3:  # Top third
                for v_line in vertical_lines:
                    if v_line[0] > 2 * w // 3:  # Right third
                        return BracketCorner(
                            position=(v_line[0], h_line[1]),
                            corner_type=corner_type,
                            confidence=0.7
                        )
    elif corner_type == 'bottom-left':
        # Look for lines near bottom-left
        for h_line in horizontal_lines:
            if h_line[1] > 2 * h // 3:  # Bottom third
                for v_line in vertical_lines:
                    if v_line[0] < w // 3:  # Left third
                        return BracketCorner(
                            position=(v_line[0], h_line[1]),
                            corner_type=corner_type,
                            confidence=0.7
                        )
    elif corner_type == 'bottom-right':
        # Look for lines near bottom-right
        for h_line in horizontal_lines:
            if h_line[1] > 2 * h // 3:  # Bottom third
                for v_line in vertical_lines:
                    if v_line[0] > 2 * w // 3:  # Right third
                        return BracketCorner(
                            position=(v_line[0], h_line[1]),
                            corner_type=corner_type,
                            confidence=0.7
                        )
    
    return None


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
    region, corners = detect_brackets_final(img)
    
    print(f"Detected {len(corners)} corners:")
    for corner in corners:
        print(f"  {corner.corner_type} at {corner.position} (confidence: {corner.confidence:.2f})")
    
    if region:
        print(f"\nFound bubble region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Visualize
        x, y, w, h = region
        vis = img.copy()
        
        # Draw region
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw corners
        colors = {
            'top-left': (255, 0, 0),      # Blue
            'top-right': (0, 255, 0),     # Green  
            'bottom-left': (0, 0, 255),   # Red
            'bottom-right': (255, 255, 0) # Yellow
        }
        
        for corner in corners:
            color = colors.get(corner.corner_type, (255, 255, 255))
            cv2.circle(vis, corner.position, 20, color, 3)
            cv2.putText(vis, corner.corner_type[:2].upper(), 
                       (corner.position[0] - 30, corner.position[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imwrite("detected_region_final.png", vis)
        print("Visualization saved to detected_region_final.png")
    else:
        print("\nCould not detect all four corners")