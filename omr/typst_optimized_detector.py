#!/usr/bin/env python3
"""
Detector optimized specifically for Typst-generated quiz PDFs
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Bubble:
    """Represents a single bubble/circle detected"""
    center: Tuple[int, int]
    radius: float
    row: int = -1
    col: int = -1
    fill_ratio: float = 0.0


@dataclass 
class BubbleGrid:
    """Organized grid of bubbles"""
    bubbles: List[List[Bubble]]
    num_questions: int
    num_options: int


class TypstOptimizedDetector:
    """Detector optimized for Typst quiz format"""
    
    def __init__(self, fill_threshold: float = 0.20, verbose: bool = False):
        self.fill_threshold = fill_threshold
        self.verbose = verbose
        # Typst-specific parameters based on observation
        self.expected_bubble_radius = 13  # Approximate radius in pixels at 300 DPI
        self.expected_row_spacing = 60    # Approximate spacing between question rows
        self.expected_col_spacing = 50    # Approximate spacing between option columns
    
    def preprocess_for_bubbles(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess specifically for Typst bubble detection
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
        
        # Use adaptive threshold optimized for printed text
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,  # Larger block size for cleaner bubbles
            C=5
        )
        
        # Remove small noise (text fragments)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        
        if self.verbose:
            cv2.imwrite("debug_preprocessed.png", cleaned)
        
        return cleaned
    
    def detect_bubbles_targeted(self, img: np.ndarray) -> List[Bubble]:
        """
        Detect bubbles using targeted approach for Typst format
        """
        bubbles = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Preprocess for bubble detection
        preprocessed = self.preprocess_for_bubbles(gray)
        
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours to find bubbles
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Expected area for bubble (pi * r^2)
            expected_area = np.pi * (self.expected_bubble_radius ** 2)
            
            # Allow 40% variance in area
            min_area = expected_area * 0.6
            max_area = expected_area * 1.4
            
            if min_area < area < max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Bubbles should be fairly circular
                    if circularity > 0.7:
                        # Get center and radius
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        # Additional radius check
                        if abs(radius - self.expected_bubble_radius) < 5:
                            bubbles.append(Bubble(
                                center=(int(x), int(y)),
                                radius=radius
                            ))
        
        # If contour method fails, try Hough with specific parameters
        if len(bubbles) < 10:
            logger.info("Contour method found few bubbles, trying Hough circles")
            
            # Very specific Hough parameters for Typst bubbles
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,  # Bubbles are well-spaced
                param1=50,
                param2=25,
                minRadius=10,
                maxRadius=16
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    # Check if this circle is in a reasonable location
                    # (not too close to edges, which might be brackets)
                    if 50 < x < img.shape[1] - 50 and 50 < y < img.shape[0] - 50:
                        bubbles.append(Bubble(center=(x, y), radius=r))
        
        return bubbles
    
    def filter_bubble_region(self, bubbles: List[Bubble], img_shape: Tuple[int, int]) -> List[Bubble]:
        """
        Filter bubbles to remove outliers and focus on main bubble region
        """
        if len(bubbles) < 5:
            return bubbles
        
        # Find the main cluster of bubbles
        x_coords = [b.center[0] for b in bubbles]
        y_coords = [b.center[1] for b in bubbles]
        
        # Use percentiles to find main region
        x_min = np.percentile(x_coords, 10)
        x_max = np.percentile(x_coords, 90)
        y_min = np.percentile(y_coords, 10)
        y_max = np.percentile(y_coords, 90)
        
        # Expand region slightly
        margin = 50
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Filter bubbles within this region
        filtered = []
        for bubble in bubbles:
            x, y = bubble.center
            if x_min <= x <= x_max and y_min <= y <= y_max:
                filtered.append(bubble)
        
        return filtered
    
    def cluster_to_grid(self, bubbles: List[Bubble]) -> BubbleGrid:
        """Organize bubbles into rows and columns using expected spacing"""
        if not bubbles:
            return BubbleGrid([], 0, 0)
        
        # Sort by Y coordinate
        bubbles_sorted = sorted(bubbles, key=lambda b: b.center[1])
        
        # Use expected row spacing for clustering
        rows = []
        current_row = [bubbles_sorted[0]]
        current_y = bubbles_sorted[0].center[1]
        
        for bubble in bubbles_sorted[1:]:
            # Check if this bubble is in a new row
            if bubble.center[1] - current_y > self.expected_row_spacing * 0.5:
                # New row
                rows.append(sorted(current_row, key=lambda b: b.center[0]))
                current_row = [bubble]
                current_y = bubble.center[1]
            else:
                current_row.append(bubble)
        
        # Add last row
        rows.append(sorted(current_row, key=lambda b: b.center[0]))
        
        # Expected 5 options (A-E) for most questions
        expected_cols = 5
        
        # Filter rows to those with expected number of columns
        valid_rows = []
        for row in rows:
            if 3 <= len(row) <= 6:  # Allow some variance
                # Pad or trim to expected columns
                if len(row) < expected_cols:
                    # Row has fewer bubbles (maybe some options missing)
                    valid_rows.append(row)
                elif len(row) > expected_cols:
                    # Too many bubbles, take first 5
                    valid_rows.append(row[:expected_cols])
                else:
                    valid_rows.append(row)
        
        # Assign indices
        for row_idx, row in enumerate(valid_rows):
            for col_idx, bubble in enumerate(row):
                bubble.row = row_idx
                bubble.col = col_idx
        
        if self.verbose:
            logger.info(f"Grid: {len(valid_rows)} questions with up to {expected_cols} options each")
        
        return BubbleGrid(
            bubbles=valid_rows,
            num_questions=len(valid_rows),
            num_options=expected_cols if valid_rows else 0
        )
    
    def read_answers(self, img: np.ndarray, grid: BubbleGrid) -> Dict[int, str]:
        """Determine which bubbles are filled"""
        answers = {}
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Invert image (bubbles should be dark)
        inverted = cv2.bitwise_not(gray)
        
        for row_idx, row in enumerate(grid.bubbles):
            max_fill_ratio = 0
            best_col = None
            
            for bubble in row:
                # Create slightly smaller mask to avoid edges
                mask = np.zeros(inverted.shape, dtype=np.uint8)
                cv2.circle(mask, bubble.center, int(bubble.radius * 0.7), 255, -1)
                
                # Measure darkness in bubble
                masked = cv2.bitwise_and(inverted, mask)
                total_pixels = cv2.countNonZero(mask)
                dark_pixels = cv2.countNonZero(masked)
                
                fill_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
                bubble.fill_ratio = fill_ratio
                
                # Track the darkest bubble in this row
                if fill_ratio > max_fill_ratio:
                    max_fill_ratio = fill_ratio
                    best_col = bubble.col
            
            # Use the darkest bubble if it exceeds threshold
            if max_fill_ratio > self.fill_threshold and best_col is not None:
                answer_letter = chr(ord('A') + best_col)
                answers[row_idx + 1] = answer_letter
            else:
                answers[row_idx + 1] = None
        
        return answers
    
    def process(self, image_path: Path) -> Dict[int, str]:
        """Main processing pipeline"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to find bubble region using brackets
        try:
            from omr.bracket_detector_v2 import find_bubble_region_v2
            bubble_region = find_bubble_region_v2(img)
        except:
            bubble_region = None
        
        if bubble_region:
            x, y, w, h = bubble_region
            roi = gray[y:y+h, x:x+w]
            logger.info(f"Using detected bubble region: {w}x{h}")
        else:
            # Use center portion of image
            h, w = gray.shape
            margin_x = w // 4
            margin_y = h // 8
            roi = gray[margin_y:h-margin_y, margin_x:w-margin_x]
            logger.info("Using center region of image")
        
        # Detect bubbles
        bubbles = self.detect_bubbles_targeted(roi)
        logger.info(f"Found {len(bubbles)} bubble candidates")
        
        # Filter to main region
        bubbles = self.filter_bubble_region(bubbles, roi.shape)
        logger.info(f"Filtered to {len(bubbles)} bubbles")
        
        # Organize into grid
        grid = self.cluster_to_grid(bubbles)
        
        # Read answers
        answers = self.read_answers(roi, grid)
        
        return answers


if __name__ == "__main__":
    # Test on different images
    detector = TypstOptimizedDetector(verbose=True)
    
    # Test synthetic
    synthetic_path = Path("synthetic_test.png")
    if synthetic_path.exists():
        print("\n1. Testing on synthetic image:")
        print("-" * 40)
        answers = detector.process(synthetic_path)
        expected = {1: 'B', 2: 'D', 3: 'A', 4: 'C', 5: 'E'}
        correct = sum(1 for k, v in expected.items() if answers.get(k) == v)
        print(f"Detected: {answers}")
        print(f"Accuracy: {correct}/5 ({correct*20}%)")
    
    # Test random
    random_path = Path("tests/generated/random_test_20250826_202248_001.png")
    if random_path.exists():
        print("\n2. Testing on random Typst quiz:")
        print("-" * 40)
        answers = detector.process(random_path)
        print(f"Detected: {answers}")