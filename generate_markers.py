#!/usr/bin/env python3
"""
Generate ArUco markers for the OMR system.
Creates 4 markers as PNG files that can be used in the Typst template.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

def generate_aruco_marker(marker_id, size=200):
    """Generate a single ArUco marker."""
    # Use the predefined dictionary DICT_4X4_50
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size)
    return marker_image

def main():
    # Create markers directory
    markers_dir = Path("markers")
    markers_dir.mkdir(exist_ok=True)
    
    # Generate 4 markers
    marker_ids = {
        0: "Top-left of netid box",
        1: "Bottom-right of netid box", 
        2: "Top-left of bubble region",
        3: "Bottom-right of bubble region"
    }
    
    for marker_id, description in marker_ids.items():
        print(f"Generating marker {marker_id}: {description}")
        
        # Generate marker
        marker = generate_aruco_marker(marker_id, size=200)
        
        # Save as PNG
        output_path = markers_dir / f"aruco_{marker_id}.png"
        cv2.imwrite(str(output_path), marker)
        print(f"  Saved to {output_path}")
    
    print("\nDone! Generated 4 ArUco markers in markers/")
    print("These PNG files can be imported into your Typst template.")

if __name__ == "__main__":
    main()