#!/usr/bin/env python3
"""Complete OMR detection pipeline for Typst-generated quizzes."""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
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

def detect_bubbles(bubble_region: np.ndarray, num_questions: int = None, 
                  options: str = "ABCD") -> List[BubbleInfo]:
    """Detect and analyze answer bubbles in the bubble region."""
    gray = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
    
    # Debug: save gray image
    cv2.imwrite("debug_gray.png", gray)
    
    # Use threshold to find dark areas (bubbles are outlined)
    # Adjusted threshold for better bubble outline detection
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
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
    
    print(f"Found {len(circles_found)} circular contours")
    
    bubbles = []
    if circles_found:
        # Sort circles by y-coordinate (row), then x-coordinate (column)
        sorted_circles = sorted(circles_found, key=lambda c: (c[1], c[0]))
        
        # Group circles into rows based on y-coordinate
        rows = []
        current_row = []
        last_y = -1
        row_threshold = 20  # Pixels difference to consider same row
        
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

def detect_answers(image_path: Path, visualize: bool = False) -> OMRResult:
    """Main OMR detection pipeline."""
    # Convert PDF to image if needed
    if image_path.suffix.lower() == '.pdf':
        image = pdf_to_image(image_path)
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
    
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
        bubbles = detect_bubbles(bubble_region)
        
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