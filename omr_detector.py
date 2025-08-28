#!/usr/bin/env python3
"""Complete OMR detection pipeline for Typst-generated quizzes."""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse

@dataclass
class MarkerInfo:
    """Information about a detected ArUco marker."""
    id: int
    corners: np.ndarray
    center: Tuple[float, float]

@dataclass
class BubbleInfo:
    """Information about a detected answer bubble."""
    question: int
    option: str
    center: Tuple[int, int]
    filled: bool
    fill_ratio: float

@dataclass 
class OMRResult:
    """Complete OMR detection result."""
    netid_region: Optional[np.ndarray]
    bubble_region: Optional[np.ndarray]
    answers: Dict[int, List[str]]  # Question number to selected options
    bubbles: List[BubbleInfo]
    markers: List[MarkerInfo]

def pdf_to_image(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Convert first page of PDF to image."""
    output_path = pdf_path.with_suffix('.png')
    
    cmd = [
        'convert',
        '-density', str(dpi),
        f'{pdf_path}[0]',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error converting PDF: {result.stderr}")
    
    img = cv2.imread(str(output_path))
    if img is None:
        raise RuntimeError(f"Error reading image: {output_path}")
    
    output_path.unlink()
    return img

def rotate_scan(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct rotation using Hough line detection.
    Technique from exam-maker for handling skewed scans.
    Returns rotated image and angle of rotation.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Use top portion for rotation detection (less affected by content)
    top_region = gray[0:h//5, 0:w]
    edges = cv2.Canny(top_region, 50, 150)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=w//3, maxLineGap=w//25)
    
    if lines is None:
        return img, 0.0
    
    # Calculate angles
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Normalize to small rotations
            if -45 <= angle <= 45:
                angles.append(angle)
    
    if not angles or abs(np.median(angles)) < 0.5:
        return img, 0.0
    
    # Rotate image
    median_angle = np.median(angles)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    
    return rotated, median_angle

def detect_aruco_markers(image: np.ndarray) -> List[MarkerInfo]:
    """Detect ArUco markers in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    markers = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            center = marker_corners.mean(axis=0)
            markers.append(MarkerInfo(
                id=marker_id,
                corners=marker_corners,
                center=tuple(center)
            ))
    
    return markers

def extract_region(image: np.ndarray, marker1: MarkerInfo, marker2: MarkerInfo, 
                   padding: int = 10, region_type: str = "default") -> np.ndarray:
    """Extract region between two markers with perspective correction."""
    
    if region_type == "bubble":
        # For bubble region, markers 2 and 3 are at diagonal corners (top-left and bottom-right)
        # Extract the rectangle defined by these corners
        x1 = int(min(marker1.center[0], marker2.center[0]) - padding)
        x2 = int(max(marker1.center[0], marker2.center[0]) + padding)
        y1 = int(min(marker1.center[1], marker2.center[1]) - padding)
        y2 = int(max(marker1.center[1], marker2.center[1]) + padding)
    else:
        # For netid region, extract between the markers
        x1 = int(min(marker1.center[0], marker2.center[0]) - padding)
        x2 = int(max(marker1.center[0], marker2.center[0]) + padding)
        y1 = int(min(marker1.center[1], marker2.center[1]) - padding)
        y2 = int(max(marker1.center[1], marker2.center[1]) + padding * 3)  # More vertical space for handwriting
    
    # Ensure we stay within image bounds
    x1 = max(0, x1)
    x2 = min(image.shape[1], x2)
    y1 = max(0, y1)
    y2 = min(image.shape[0], y2)
    
    # Extract the region
    region = image[y1:y2, x1:x2]
    
    return region

def adaptive_threshold(img: np.ndarray, blur: bool = True) -> np.ndarray:
    """Apply adaptive threshold with OTSU - technique from exam-maker."""
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 1)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_column_layout(circles_found: List[Tuple[int, int, int]]) -> Tuple[str, Optional[float]]:
    """Detect if bubbles are in one or two columns by finding gaps in X coordinates."""
    if not circles_found:
        return "single", None
    
    # Get unique X coordinates (with some tolerance for alignment)
    x_coords = sorted(set(c[0] for c in circles_found))
    
    # Find gaps between consecutive X positions
    gaps = []
    for i in range(1, len(x_coords)):
        gap = x_coords[i] - x_coords[i-1]
        if gap > 50:  # Significant gap
            gaps.append((gap, (x_coords[i-1] + x_coords[i]) / 2))
    
    # Look for a large gap that would indicate column separation
    # Typically the gap between columns is much larger than between bubbles
    large_gaps = [g for g in gaps if g[0] > 100]
    
    if large_gaps:
        # Find the largest gap - this is likely the column separator
        largest_gap = max(large_gaps, key=lambda x: x[0])
        return "double", largest_gap[1]
    
    return "single", None

def detect_circles_liberal(image: np.ndarray) -> List[Tuple[int, int, int]]:
    """Detect circles with liberal thresholds for first pass."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Use adaptive threshold
    thresh = cv2.bitwise_not(adaptive_threshold(gray))
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > 3000:  # More liberal area bounds
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.6:  # More liberal circularity
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 8 < radius < 50:  # More liberal radius range
                circles.append((int(x), int(y), int(radius)))
    
    return circles

@dataclass
class GridStructure:
    """Detected grid structure for bubbles."""
    col_positions: List[float]
    row_positions: List[float]
    col_spacing: float
    row_spacing: float
    
def detect_grid_structure(circles: List[Tuple[int, int, int]], 
                         expected_cols: int = 5, force_cols: bool = True) -> Optional[GridStructure]:
    """Detect grid structure from detected circles using clustering.
    
    Strategy: Over-cluster and take the top clusters by population.
    This naturally filters out outliers like the '0' in '10'.
    """
    if len(circles) < expected_cols * 2:  # Need at least 2 rows
        return None
    
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Extract X and Y coordinates
    x_coords = np.array([c[0] for c in circles])
    y_coords = np.array([c[1] for c in circles])
    
    # Use over-clustering approach: cluster into more groups than needed
    # then select the top clusters by population
    n_clusters_to_try = min(7, len(set(x_coords)))  # Try up to 7 clusters
    
    if n_clusters_to_try < expected_cols:
        return None
    
    # Cluster X coordinates
    kmeans_x = KMeans(n_clusters=n_clusters_to_try, random_state=42, n_init=10)
    x_labels = kmeans_x.fit_predict(x_coords.reshape(-1, 1))
    x_centers = kmeans_x.cluster_centers_.flatten()
    
    # Count population of each cluster
    cluster_populations = []
    for i in range(n_clusters_to_try):
        pop = np.sum(x_labels == i)
        cluster_populations.append((x_centers[i], pop, i))
    
    # Sort by population and take the top clusters
    cluster_populations.sort(key=lambda x: x[1], reverse=True)
    
    # Debug: show cluster populations
    # print(f"Cluster populations (x_pos, count): {[(round(c[0]), c[1]) for c in cluster_populations]}")
    
    # Take the top N clusters by population based on expected_cols
    # The interference (like "0" from "10") should have lower population
    # print(f"Cluster populations: {[(round(c[0]), c[1]) for c in cluster_populations]}")
    
    # Take exactly expected_cols clusters with highest population
    best_cols = sorted([c[0] for c in cluster_populations[:expected_cols]])
    
    if len(best_cols) == 0:
        return None
    
    # Find row clusters
    # Estimate number of rows from total circles / columns
    estimated_rows = max(3, len(circles) // len(best_cols))
    
    # Cluster Y coordinates
    kmeans_y = KMeans(n_clusters=min(estimated_rows, len(y_coords)), 
                      random_state=42, n_init=10)
    kmeans_y.fit(y_coords.reshape(-1, 1))
    row_positions = sorted(kmeans_y.cluster_centers_.flatten())
    
    # Clean up row positions - merge rows that are too close
    cleaned_rows = []
    min_row_gap = 30  # Minimum gap between rows
    for row in row_positions:
        if not cleaned_rows or (row - cleaned_rows[-1]) > min_row_gap:
            cleaned_rows.append(row)
        else:
            # Merge with previous row (average position)
            cleaned_rows[-1] = (cleaned_rows[-1] + row) / 2
    
    # Calculate median spacings
    col_spacings = np.diff(best_cols)
    row_spacings = np.diff(cleaned_rows)
    
    col_spacing = np.median(col_spacings) if len(col_spacings) > 0 else 50
    row_spacing = np.median(row_spacings) if len(row_spacings) > 0 else 50
    
    return GridStructure(
        col_positions=best_cols,
        row_positions=cleaned_rows,
        col_spacing=col_spacing,
        row_spacing=row_spacing
    )

def create_bubble_mask(image_shape: tuple, grid: GridStructure, 
                       margin: int = 20) -> np.ndarray:
    """Create mask with circles at grid intersections."""
    # Start with white image
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
    
    # Draw black circles at each grid intersection
    for x_pos in grid.col_positions:
        for y_pos in grid.row_positions:
            cv2.circle(mask, (int(x_pos), int(y_pos)), margin, 0, -1)
    
    return mask

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image, making masked areas white."""
    result = image.copy()
    # Where mask is white (255), make image white
    if len(result.shape) == 3:
        result[mask == 255] = [255, 255, 255]
    else:
        result[mask == 255] = 255
    return result

def detect_bubbles_old(bubble_region: np.ndarray, num_questions: int = None, 
                  options: str = "ABCD") -> List[BubbleInfo]:
    """Detect and analyze answer bubbles in the bubble region."""
    gray = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
    
    # Debug: save gray image
    cv2.imwrite("debug_gray.png", gray)
    
    # Use adaptive threshold with OTSU for better bubble detection
    thresh = cv2.bitwise_not(adaptive_threshold(gray))
    cv2.imwrite("debug_thresh.png", thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find circles
    circles_found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > 2000:  # Filter by area
            continue
            
        # Check if contour is circular
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.7:  # Reasonably circular
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 10 < radius < 40:  # Reasonable bubble size
                circles_found.append((int(x), int(y), int(radius)))
    
    # print(f"Found {len(circles_found)} circular contours")
    
    bubbles = []
    if circles_found:
        # Detect if we have one or two columns
        layout_type, split_x = detect_column_layout(circles_found)
        # print(f"Detected layout: {layout_type}, split_x: {split_x}")
        
        if layout_type == "double" and split_x:
            # Two-column layout
            left_circles = [c for c in circles_found if c[0] < split_x]
            right_circles = [c for c in circles_found if c[0] >= split_x]
            
            # Process left column
            left_sorted = sorted(left_circles, key=lambda c: (c[1], c[0]))
            left_rows = []
            current_row = []
            last_y = -1
            row_threshold = 20
            
            for circle in left_sorted:
                x, y, r = circle
                if last_y == -1 or abs(y - last_y) < row_threshold:
                    current_row.append(circle)
                    last_y = y
                else:
                    if current_row:
                        left_rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [circle]
                    last_y = y
            if current_row:
                left_rows.append(sorted(current_row, key=lambda c: c[0]))
            
            # Process right column
            right_sorted = sorted(right_circles, key=lambda c: (c[1], c[0]))
            right_rows = []
            current_row = []
            last_y = -1
            
            for circle in right_sorted:
                x, y, r = circle
                if last_y == -1 or abs(y - last_y) < row_threshold:
                    current_row.append(circle)
                    last_y = y
                else:
                    if current_row:
                        right_rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [circle]
                    last_y = y
            if current_row:
                right_rows.append(sorted(current_row, key=lambda c: c[0]))
            
            # Process left column (questions 1 to N/2)
            for row_idx, row in enumerate(left_rows):
                question_num = row_idx + 1
                for option_num, circle in enumerate(row):
                    if option_num < len(options):
                        x, y, r = circle
                        mask = np.zeros(gray.shape, dtype="uint8")
                        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
                        bubble_roi = gray[mask == 255]
                        mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                        fill_ratio = (255 - mean_val) / 255
                        is_filled = mean_val < 150
                        
                        bubbles.append(BubbleInfo(
                            question=question_num,
                            option=options[option_num],
                            center=(x, y),
                            filled=is_filled,
                            fill_ratio=fill_ratio
                        ))
            
            # Process right column (questions N/2+1 to N)
            left_count = len(left_rows)
            for row_idx, row in enumerate(right_rows):
                question_num = left_count + row_idx + 1
                for option_num, circle in enumerate(row):
                    if option_num < len(options):
                        x, y, r = circle
                        mask = np.zeros(gray.shape, dtype="uint8")
                        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
                        bubble_roi = gray[mask == 255]
                        mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                        fill_ratio = (255 - mean_val) / 255
                        is_filled = mean_val < 150
                        
                        bubbles.append(BubbleInfo(
                            question=question_num,
                            option=options[option_num],
                            center=(x, y),
                            filled=is_filled,
                            fill_ratio=fill_ratio
                        ))
        else:
            # Single column layout (existing logic)
            sorted_circles = sorted(circles_found, key=lambda c: (c[1], c[0]))
            
            # Group circles into rows based on y-coordinate
            rows = []
            current_row = []
            last_y = -1
            row_threshold = 20
            
            for circle in sorted_circles:
                x, y, r = circle
                if last_y == -1 or abs(y - last_y) < row_threshold:
                    current_row.append(circle)
                    last_y = y
                else:
                    if current_row:
                        rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [circle]
                    last_y = y
            
            if current_row:
                rows.append(sorted(current_row, key=lambda c: c[0]))
            
            # Process each row
            for question_num, row in enumerate(rows, 1):
                for option_num, circle in enumerate(row):
                    if option_num < len(options):
                        x, y, r = circle
                        
                        # Extract bubble region - smaller mask to focus on interior
                        mask = np.zeros(gray.shape, dtype="uint8")
                        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)  # Use 70% of radius to avoid edges
                        
                        # Calculate fill ratio by checking darkness inside bubble
                        bubble_roi = gray[mask == 255]
                        mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                        
                        # Lower mean value = darker = filled
                        # Empty bubbles should be mostly white (high values)
                        # Filled bubbles will have darker pixels (lower values)
                        fill_ratio = (255 - mean_val) / 255
                        
                        # Determine if bubble is filled (stricter threshold)
                        # Typical empty bubble: ~240-255, Filled bubble: ~0-100
                        is_filled = mean_val < 150  # Darker than this threshold = filled
                        
                        bubbles.append(BubbleInfo(
                            question=question_num,
                            option=options[option_num],
                            center=(x, y),
                            filled=is_filled,
                            fill_ratio=fill_ratio
                        ))
    
    return bubbles

def detect_bubbles(bubble_region: np.ndarray, num_questions: int = None,
                  options: str = "ABCDE", debug: bool = False) -> List[BubbleInfo]:
    """Two-pass bubble detection with grid-based filtering.
    
    Approach:
    1. Detect all circles liberally
    2. Use circles to determine layout (1 or 2 columns)
    3. Process each column separately with over-clustering
    4. Clean and re-detect for final results
    """
    # Step 1: Liberal detection to find all potential bubbles
    circles = detect_circles_liberal(bubble_region)
    
    if debug:
        debug_img = bubble_region.copy()
        for x, y, r in circles:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
        cv2.imwrite("debug_pass1_circles.png", debug_img)
        print(f"Pass 1: Found {len(circles)} circles")
    
    # Step 2: Detect layout using the circles
    x_sorted = sorted([c[0] for c in circles])
    gaps = []
    for i in range(1, len(x_sorted)):
        gap = x_sorted[i] - x_sorted[i-1]
        if gap > 200:  # Large gap indicating column separation
            gaps.append((gap, (x_sorted[i-1] + x_sorted[i]) / 2))
    
    is_two_column = len(gaps) > 0
    
    if debug:
        print(f"Layout: {'Two-column' if is_two_column else 'Single column'}")
        if is_two_column:
            print(f"Column gap at x={gaps[0][1]}")
    
    bubbles = []
    gray = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY) if len(bubble_region.shape) == 3 else bubble_region
    
    if is_two_column:
        # Step 3: Process two-column layout
        split_x = gaps[0][1]
        left_circles = [c for c in circles if c[0] < split_x]
        right_circles = [c for c in circles if c[0] >= split_x]
        
        # Process left column with simplified grid detection
        left_grid = detect_grid_structure(left_circles, expected_cols=len(options))
        if left_grid:
            if debug:
                print(f"Left grid: {len(left_grid.col_positions)} cols, {len(left_grid.row_positions)} rows")
            
            # Create mask for left column
            left_mask = create_bubble_mask(bubble_region.shape, left_grid, margin=25)
            left_cleaned = apply_mask_to_image(bubble_region, left_mask)
            left_final = detect_circles_liberal(left_cleaned)
            
            # Map left column circles to bubbles
            for row_idx, y_pos in enumerate(left_grid.row_positions):
                question_num = row_idx + 1
                for col_idx, x_pos in enumerate(left_grid.col_positions):
                    if col_idx >= len(options):
                        continue
                    
                    # Find circle closest to this grid position
                    best_circle = None
                    min_dist = float('inf')
                    
                    for cx, cy, cr in left_final:
                        dist = np.sqrt((cx - x_pos) ** 2 + (cy - y_pos) ** 2)
                        if dist < min_dist and dist < 30:
                            min_dist = dist
                            best_circle = (cx, cy, cr)
                    
                    if best_circle:
                        x, y, r = best_circle
                        mask = np.zeros(gray.shape, dtype="uint8")
                        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
                        bubble_roi = gray[mask == 255]
                        mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                        fill_ratio = (255 - mean_val) / 255
                        is_filled = mean_val < 150
                        
                        bubbles.append(BubbleInfo(
                            question=question_num,
                            option=options[col_idx],
                            center=(x, y),
                            filled=is_filled,
                            fill_ratio=fill_ratio
                        ))
        
        # Process right column
        right_grid = detect_grid_structure(right_circles, expected_cols=len(options))
        if right_grid:
            if debug:
                print(f"Right grid: {len(right_grid.col_positions)} cols, {len(right_grid.row_positions)} rows")
            
            # Create mask for right column
            right_mask = create_bubble_mask(bubble_region.shape, right_grid, margin=25)
            right_cleaned = apply_mask_to_image(bubble_region, right_mask)
            right_final = detect_circles_liberal(right_cleaned)
            
            # Map right column circles to bubbles
            left_questions = len(left_grid.row_positions) if left_grid else 0
            for row_idx, y_pos in enumerate(right_grid.row_positions):
                question_num = left_questions + row_idx + 1
                for col_idx, x_pos in enumerate(right_grid.col_positions):
                    if col_idx >= len(options):
                        continue
                    
                    # Find circle closest to this grid position
                    best_circle = None
                    min_dist = float('inf')
                    
                    for cx, cy, cr in right_final:
                        dist = np.sqrt((cx - x_pos) ** 2 + (cy - y_pos) ** 2)
                        if dist < min_dist and dist < 30:
                            min_dist = dist
                            best_circle = (cx, cy, cr)
                    
                    if best_circle:
                        x, y, r = best_circle
                        mask = np.zeros(gray.shape, dtype="uint8")
                        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
                        bubble_roi = gray[mask == 255]
                        mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                        fill_ratio = (255 - mean_val) / 255
                        is_filled = mean_val < 150
                        
                        bubbles.append(BubbleInfo(
                            question=question_num,
                            option=options[col_idx],
                            center=(x, y),
                            filled=is_filled,
                            fill_ratio=fill_ratio
                        ))
    else:
        # Single column layout - use original logic
        grid = detect_grid_structure(circles, expected_cols=len(options))
        if grid is None:
            print("Warning: Could not detect grid structure, falling back to old method")
            return detect_bubbles_old(bubble_region, num_questions, options[:4])
        
        if debug:
            print(f"Grid: {len(grid.col_positions)} cols, {len(grid.row_positions)} rows")
        
        mask = create_bubble_mask(bubble_region.shape, grid, margin=25)
        cleaned_image = apply_mask_to_image(bubble_region, mask)
        
        if debug:
            cv2.imwrite("debug_mask.png", mask)
            cv2.imwrite("debug_cleaned.png", cleaned_image)
        
        final_circles = detect_circles_liberal(cleaned_image)
        
        if debug:
            print(f"Pass 2: Found {len(final_circles)} circles after cleaning")
        
        for row_idx, y_pos in enumerate(grid.row_positions):
            question_num = row_idx + 1
            for col_idx, x_pos in enumerate(grid.col_positions):
                if col_idx >= len(options):
                    continue
                
                best_circle = None
                min_dist = float('inf')
                
                for cx, cy, cr in final_circles:
                    dist = np.sqrt((cx - x_pos) ** 2 + (cy - y_pos) ** 2)
                    if dist < min_dist and dist < 30:
                        min_dist = dist
                        best_circle = (cx, cy, cr)
                
                if best_circle:
                    x, y, r = best_circle
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
                    bubble_roi = gray[mask == 255]
                    mean_val = np.mean(bubble_roi) if len(bubble_roi) > 0 else 255
                    fill_ratio = (255 - mean_val) / 255
                    is_filled = mean_val < 150
                    
                    bubbles.append(BubbleInfo(
                        question=question_num,
                        option=options[col_idx],
                        center=(x, y),
                        filled=is_filled,
                        fill_ratio=fill_ratio
                    ))
    
    return bubbles

def detect_answers(image_path: Path, visualize: bool = False) -> OMRResult:
    """Main OMR detection pipeline."""
    # Convert PDF to image if needed
    if image_path.suffix.lower() == '.pdf':
        image = pdf_to_image(image_path)
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
    
    # Rotation correction for scanned documents
    image, rotation_angle = rotate_scan(image)
    if rotation_angle != 0:
        print(f"Corrected rotation by {rotation_angle:.2f} degrees")
    
    # Detect ArUco markers
    markers = detect_aruco_markers(image)
    markers_by_id = {m.id: m for m in markers}
    
    # Extract regions
    netid_region = None
    bubble_region = None
    
    if 0 in markers_by_id and 1 in markers_by_id:
        netid_region = extract_region(image, markers_by_id[0], markers_by_id[1], region_type="netid")
    
    if 2 in markers_by_id and 3 in markers_by_id:
        bubble_region = extract_region(image, markers_by_id[2], markers_by_id[3], region_type="bubble")
    
    # Detect bubbles and extract answers
    bubbles = []
    answers = {}
    
    if bubble_region is not None:
        bubbles = detect_bubbles(bubble_region, debug=False)
        
        # Group answers by question
        for bubble in bubbles:
            if bubble.filled:
                if bubble.question not in answers:
                    answers[bubble.question] = []
                answers[bubble.question].append(bubble.option)
    
    result = OMRResult(
        netid_region=netid_region,
        bubble_region=bubble_region,
        answers=answers,
        bubbles=bubbles,
        markers=markers
    )
    
    if visualize:
        visualize_results(image, result, image_path)
    
    return result

def visualize_results(image: np.ndarray, result: OMRResult, source_path: Path):
    """Create visualization of detection results."""
    vis_image = image.copy()
    
    # Draw markers
    for marker in result.markers:
        pts = marker.corners.astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        cx, cy = marker.center
        cv2.putText(vis_image, f"ID: {marker.id}", 
                   (int(cx - 20), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw regions
    markers_by_id = {m.id: m for m in result.markers}
    
    if 0 in markers_by_id and 1 in markers_by_id:
        pt1 = tuple(markers_by_id[0].corners[0].astype(int))
        pt2 = tuple(markers_by_id[1].corners[2].astype(int))
        cv2.rectangle(vis_image, pt1, pt2, (255, 0, 255), 2)
        cv2.putText(vis_image, "NetID Region", 
                   (pt1[0], pt1[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    if 2 in markers_by_id and 3 in markers_by_id:
        pt1 = tuple(markers_by_id[2].corners[0].astype(int))
        pt2 = tuple(markers_by_id[3].corners[2].astype(int))
        cv2.rectangle(vis_image, pt1, pt2, (255, 255, 0), 2)
        cv2.putText(vis_image, "Bubble Region", 
                   (pt1[0], pt1[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Save visualization
    output_path = source_path.with_name(f"{source_path.stem}_detected.png")
    cv2.imwrite(str(output_path), vis_image)
    print(f"Visualization saved to {output_path}")
    
    # Also save extracted regions if available
    if result.bubble_region is not None:
        bubble_vis = result.bubble_region.copy()
        
        # Draw detected bubbles on the bubble region
        for bubble in result.bubbles:
            color = (0, 255, 0) if bubble.filled else (0, 0, 255)
            cv2.circle(bubble_vis, bubble.center, 12, color, 2)
            cv2.putText(bubble_vis, f"{bubble.question}{bubble.option}", 
                       (bubble.center[0] - 10, bubble.center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        bubble_path = source_path.with_name(f"{source_path.stem}_bubbles.png")
        cv2.imwrite(str(bubble_path), bubble_vis)
        print(f"Bubble region saved to {bubble_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OMR detection for Typst quizzes")
    parser.add_argument("input", type=Path, help="Input PDF or image file")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Save visualization images")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF conversion (default: 300)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        sys.exit(1)
    
    try:
        result = detect_answers(args.input, visualize=args.visualize)
        
        print(f"\nDetected {len(result.markers)} ArUco markers")
        print(f"Found {len(result.bubbles)} bubbles")
        print(f"\nAnswers:")
        for question in sorted(result.answers.keys()):
            options = ", ".join(result.answers[question])
            print(f"  Question {question}: {options}")
        
        if not result.answers:
            print("  No filled bubbles detected")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()