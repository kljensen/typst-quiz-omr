#!/usr/bin/env python3
"""
Simplified bracket detection using morphological operations
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


def detect_brackets_morphology(img: np.ndarray) -> List[BracketCorner]:
    """
    Detect L-shaped brackets using morphological operations.
    
    This approach:
    1. Binarizes the image
    2. Uses morphological operations to enhance L-shapes
    3. Detects corners using contour analysis
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Binarize the image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create L-shaped kernels for different orientations
    kernel_size = 15
    thickness = 3
    
    kernels = {
        'top-left': create_l_kernel(kernel_size, thickness, 'top-left'),
        'top-right': create_l_kernel(kernel_size, thickness, 'top-right'),
        'bottom-left': create_l_kernel(kernel_size, thickness, 'bottom-left'),
        'bottom-right': create_l_kernel(kernel_size, thickness, 'bottom-right')
    }
    
    detected_corners = []
    
    for corner_type, kernel in kernels.items():
        # Apply morphological hit-or-miss transform
        result = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
        
        # Find positions where the kernel matched
        positions = np.where(result > 0)
        
        for y, x in zip(positions[0], positions[1]):
            detected_corners.append(BracketCorner(
                position=(x, y),
                corner_type=corner_type,
                confidence=1.0
            ))
    
    # If morphological approach fails, try contour-based detection
    if len(detected_corners) < 4:
        detected_corners.extend(detect_brackets_contours(binary))
    
    return detected_corners


def create_l_kernel(size: int, thickness: int, orientation: str) -> np.ndarray:
    """Create an L-shaped kernel for morphological operations"""
    kernel = np.zeros((size, size), dtype=np.uint8)
    
    if orientation == 'top-left':
        # Horizontal line
        kernel[:thickness, :size//2+thickness] = 1
        # Vertical line
        kernel[:size//2+thickness, :thickness] = 1
    elif orientation == 'top-right':
        # Horizontal line
        kernel[:thickness, size//2-thickness:] = 1
        # Vertical line
        kernel[:size//2+thickness, -thickness:] = 1
    elif orientation == 'bottom-left':
        # Horizontal line
        kernel[-thickness:, :size//2+thickness] = 1
        # Vertical line
        kernel[size//2-thickness:, :thickness] = 1
    elif orientation == 'bottom-right':
        # Horizontal line
        kernel[-thickness:, size//2-thickness:] = 1
        # Vertical line
        kernel[size//2-thickness:, -thickness:] = 1
    
    return kernel


def detect_brackets_contours(binary: np.ndarray) -> List[BracketCorner]:
    """
    Detect brackets using contour analysis.
    Looks for L-shaped contours in the binary image.
    """
    detected_corners = []
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for L-shaped contours (3-4 vertices)
        if 3 <= len(approx) <= 5:
            # Check if it forms an L-shape
            if is_l_shaped(approx):
                # Find the corner point of the L
                corner_point = find_l_corner(approx)
                if corner_point is not None:
                    corner_type = determine_corner_type(corner_point, binary.shape)
                    detected_corners.append(BracketCorner(
                        position=tuple(corner_point),
                        corner_type=corner_type,
                        confidence=0.8
                    ))
    
    return detected_corners


def is_l_shaped(contour: np.ndarray) -> bool:
    """Check if a contour is L-shaped"""
    if len(contour) < 3:
        return False
    
    # Calculate angles between consecutive edges
    angles = []
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        p3 = contour[(i + 2) % len(contour)][0]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        angles.append(angle)
    
    # Check for right angles (approximately 90 degrees)
    has_right_angle = any(75 < angle < 105 for angle in angles)
    
    return has_right_angle


def find_l_corner(contour: np.ndarray) -> Optional[np.ndarray]:
    """Find the corner point of an L-shaped contour"""
    if len(contour) < 3:
        return None
    
    # Find the point with maximum curvature (the corner)
    max_curvature = 0
    corner_point = None
    
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        p3 = contour[(i + 2) % len(contour)][0]
        
        # Calculate curvature
        v1 = p2 - p1
        v2 = p3 - p2
        cross_product = abs(np.cross(v1, v2))
        
        if cross_product > max_curvature:
            max_curvature = cross_product
            corner_point = p2
    
    return corner_point


def determine_corner_type(point: np.ndarray, image_shape: Tuple[int, int]) -> str:
    """Determine which corner type based on position in image"""
    h, w = image_shape[:2]
    x, y = point
    
    if x < w // 2 and y < h // 2:
        return 'top-left'
    elif x >= w // 2 and y < h // 2:
        return 'top-right'
    elif x < w // 2 and y >= h // 2:
        return 'bottom-left'
    else:
        return 'bottom-right'


def find_bubble_region(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bubble region defined by L-shaped brackets.
    Returns (x, y, width, height) or None if not found.
    """
    corners = detect_brackets_morphology(img)
    
    if len(corners) < 4:
        return None
    
    # Group corners by type
    corner_dict = {}
    for corner in corners:
        if corner.corner_type not in corner_dict:
            corner_dict[corner.corner_type] = []
        corner_dict[corner.corner_type].append(corner)
    
    # Find the best set of 4 corners (one of each type)
    if len(corner_dict) < 4:
        return None
    
    # Get one corner of each type (use the most confident one)
    tl = max(corner_dict.get('top-left', []), key=lambda c: c.confidence, default=None)
    tr = max(corner_dict.get('top-right', []), key=lambda c: c.confidence, default=None)
    bl = max(corner_dict.get('bottom-left', []), key=lambda c: c.confidence, default=None)
    br = max(corner_dict.get('bottom-right', []), key=lambda c: c.confidence, default=None)
    
    if not all([tl, tr, bl, br]):
        return None
    
    # Calculate bounding box
    x_min = min(tl.position[0], bl.position[0])
    x_max = max(tr.position[0], br.position[0])
    y_min = min(tl.position[1], tr.position[1])
    y_max = max(bl.position[1], br.position[1])
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = Path("tests/generated/test_data_page_3.png")
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    img = cv2.imread(str(image_path))
    region = find_bubble_region(img)
    
    if region:
        print(f"Found bubble region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Draw the detected region
        x, y, w, h = region
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite("detected_region.png", vis)
        print("Visualization saved to detected_region.png")
    else:
        print("Could not detect bubble region")