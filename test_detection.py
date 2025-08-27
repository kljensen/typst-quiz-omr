#!/usr/bin/env python3
"""Test ArUco marker detection on generated quiz PDFs."""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class MarkerInfo:
    """Information about a detected ArUco marker."""
    id: int
    corners: np.ndarray
    center: Tuple[float, float]

def pdf_to_image(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Convert first page of PDF to image."""
    output_path = pdf_path.with_suffix('.png')
    
    # Use ImageMagick's convert command
    cmd = [
        'convert',
        '-density', str(dpi),
        f'{pdf_path}[0]',  # First page only
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting PDF: {result.stderr}")
        sys.exit(1)
    
    # Read the image
    img = cv2.imread(str(output_path))
    if img is None:
        print(f"Error reading image: {output_path}")
        sys.exit(1)
    
    # Clean up
    output_path.unlink()
    
    return img

def detect_aruco_markers(image: np.ndarray) -> List[MarkerInfo]:
    """Detect ArUco markers in an image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
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

def visualize_detection(image: np.ndarray, markers: List[MarkerInfo]) -> np.ndarray:
    """Draw detected markers on the image."""
    vis_image = image.copy()
    
    # Draw each marker
    for marker in markers:
        # Draw marker outline
        pts = marker.corners.astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        
        # Draw marker ID
        cx, cy = marker.center
        cv2.putText(vis_image, f"ID: {marker.id}", 
                   (int(cx - 20), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw center point
        cv2.circle(vis_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    
    # Draw regions based on marker IDs
    if len(markers) >= 4:
        # Sort markers by ID
        markers_by_id = {m.id: m for m in markers}
        
        # Draw NetID region (markers 0 and 1)
        if 0 in markers_by_id and 1 in markers_by_id:
            pt1 = tuple(markers_by_id[0].corners[0].astype(int))
            pt2 = tuple(markers_by_id[1].corners[2].astype(int))
            cv2.rectangle(vis_image, pt1, pt2, (255, 0, 255), 2)
            cv2.putText(vis_image, "NetID Region", 
                       (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw Bubble region (markers 2 and 3)
        if 2 in markers_by_id and 3 in markers_by_id:
            pt1 = tuple(markers_by_id[2].corners[0].astype(int))
            pt2 = tuple(markers_by_id[3].corners[2].astype(int))
            cv2.rectangle(vis_image, pt1, pt2, (255, 255, 0), 2)
            cv2.putText(vis_image, "Bubble Region", 
                       (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return vis_image

def main():
    """Main testing function."""
    # Check for command line argument
    if len(sys.argv) > 1:
        quiz_pdf = Path(sys.argv[1])
    else:
        # Generate a test quiz if it doesn't exist
        quiz_pdf = Path("example_quiz_with_markers.pdf")
        if not quiz_pdf.exists():
            print("Generating test quiz...")
            subprocess.run(["typst", "compile", "example_quiz_with_markers.typ"])
    
    print(f"Converting {quiz_pdf} to image...")
    image = pdf_to_image(quiz_pdf)
    print(f"Image shape: {image.shape}")
    
    print("Detecting ArUco markers...")
    markers = detect_aruco_markers(image)
    
    print(f"\nFound {len(markers)} markers:")
    for marker in markers:
        print(f"  - ID {marker.id}: center at {marker.center}")
    
    if len(markers) == 4:
        print("\n✓ All 4 markers detected successfully!")
    else:
        print(f"\n⚠ Expected 4 markers but found {len(markers)}")
    
    # Visualize detection
    vis_image = visualize_detection(image, markers)
    
    # Save visualization
    output_path = Path("detection_result.png")
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nVisualization saved to {output_path}")
    
    # Also save a smaller version for quick viewing
    scale = 0.3
    small = cv2.resize(vis_image, None, fx=scale, fy=scale)
    cv2.imwrite("detection_result_small.png", small)
    print(f"Small version saved to detection_result_small.png")

if __name__ == "__main__":
    main()