#!/usr/bin/env python3
"""
Final optimized detector for Typst-generated quiz PDFs
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


class TypstFinalDetector:
    """Final detector optimized for Typst quiz format"""
    
    def __init__(self, fill_threshold: float = 0.35, verbose: bool = False):
        self.fill_threshold = fill_threshold
        self.verbose = verbose
    
    def detect_bubbles(self, img: np.ndarray) -> List[Bubble]:
        """Detect bubbles using Hough circles"""
        bubbles = []
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Use Hough circles with optimized parameters for Typst
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=25,  # Minimum distance between centers
            param1=100,  # Canny high threshold
            param2=15,   # Accumulator threshold for circle centers
            minRadius=8,
            maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                bubbles.append(Bubble(center=(x, y), radius=float(r)))
            
            if self.verbose:
                logger.info(f"Found {len(bubbles)} bubbles")
        
        return bubbles
    
    def cluster_to_grid(self, bubbles: List[Bubble]) -> BubbleGrid:
        """Organize bubbles into a grid using smart clustering"""
        if not bubbles:
            return BubbleGrid([], 0, 0)
        
        # Sort by Y coordinate (top to bottom)
        bubbles = sorted(bubbles, key=lambda b: b.center[1])
        
        # Group bubbles by Y coordinate with tolerance
        y_tolerance = 20  # Bubbles within 20 pixels are same row
        rows_raw = []
        current_row = [bubbles[0]]
        
        for bubble in bubbles[1:]:
            # Check if bubble is in same row as last bubble in current row
            y_diff = bubble.center[1] - current_row[-1].center[1]
            
            if y_diff < y_tolerance:
                # Same row
                current_row.append(bubble)
            else:
                # New row
                rows_raw.append(current_row)
                current_row = [bubble]
        
        # Add last row
        rows_raw.append(current_row)
        
        if self.verbose:
            logger.info(f"Found {len(rows_raw)} raw rows")
            row_sizes = [len(r) for r in rows_raw]
            logger.info(f"Row sizes: {row_sizes}")
        
        # Find rows with exactly 5 bubbles (complete question rows)
        # Also check for rows that might have 10 bubbles (2 questions side by side)
        rows = []
        for row_bubbles in rows_raw:
            sorted_row = sorted(row_bubbles, key=lambda b: b.center[0])
            
            if len(sorted_row) == 5:
                # Perfect row
                rows.append(sorted_row)
            elif len(sorted_row) == 10:
                # Possibly 2 questions side by side
                # Split in half
                rows.append(sorted_row[:5])
                rows.append(sorted_row[5:])
            elif 4 <= len(sorted_row) <= 6:
                # Close enough, try to use it
                if len(sorted_row) == 4:
                    # Missing one bubble, skip for now
                    if self.verbose:
                        logger.warning(f"Row with 4 bubbles, skipping")
                elif len(sorted_row) == 6:
                    # Extra bubble, take first 5
                    rows.append(sorted_row[:5])
            elif len(sorted_row) >= 15:
                # Might be 3 questions (15 bubbles) 
                # Split into groups of 5
                for i in range(0, len(sorted_row) - 4, 5):
                    if i + 5 <= len(sorted_row):
                        rows.append(sorted_row[i:i+5])
        
        # Keep only first 10 rows (10 questions expected)
        if len(rows) > 10:
            rows = rows[:10]
        
        # Assign grid positions  
        for row_idx, row in enumerate(rows):
            for col_idx, bubble in enumerate(row):
                bubble.row = row_idx
                bubble.col = col_idx
        
        if self.verbose:
            logger.info(f"Grid: {len(rows)} questions x 5 options")
        
        return BubbleGrid(
            bubbles=rows,
            num_questions=len(rows),
            num_options=5
        )
    
    def calculate_fill_ratios(self, img: np.ndarray, grid: BubbleGrid) -> None:
        """Calculate fill ratio for each bubble"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive threshold for better fill detection
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )
        
        for row in grid.bubbles:
            for bubble in row:
                # Create mask for bubble interior
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.circle(mask, bubble.center, int(bubble.radius * 0.8), 255, -1)
                
                # Count dark pixels (filled areas are 0 in binary image)
                masked_region = binary[mask > 0]
                total_pixels = len(masked_region)
                dark_pixels = np.sum(masked_region == 0)
                
                bubble.fill_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
    
    def read_answers(self, grid: BubbleGrid) -> Dict[int, str]:
        """Determine which bubbles are filled based on fill ratios"""
        answers = {}
        
        for row_idx, row in enumerate(grid.bubbles):
            # Find bubble with highest fill ratio in this row
            max_fill = 0
            best_col = None
            
            for bubble in row:
                if bubble.fill_ratio > max_fill:
                    max_fill = bubble.fill_ratio
                    best_col = bubble.col
            
            # Only mark as answered if fill ratio exceeds threshold
            if max_fill > self.fill_threshold and best_col is not None:
                answers[row_idx + 1] = chr(ord('A') + best_col)
            else:
                answers[row_idx + 1] = None
        
        return answers
    
    def process(self, image_path: Path) -> Dict[int, str]:
        """Main processing pipeline"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Try to detect bubble region using brackets
        try:
            from omr.bracket_detector_v2 import find_bubble_region_v2
            bubble_region = find_bubble_region_v2(img)
        except:
            bubble_region = None
        
        if bubble_region:
            x, y, w, h = bubble_region
            # Extend region to ensure we don't cut off bubbles
            # Extend down to capture all questions
            h = min(h + 100, img.shape[0] - y)
            roi = img[y:y+h, x:x+w]
            if self.verbose:
                logger.info(f"Using bubble region (extended): {w}x{h}")
        else:
            # Use center region as fallback
            h, w = img.shape[:2]
            margin_x = w // 4
            margin_y = h // 8  
            roi = img[margin_y:h-margin_y, margin_x:w-margin_x]
            if self.verbose:
                logger.info("Using center region")
        
        # Detect bubbles
        bubbles = self.detect_bubbles(roi)
        
        # Organize into grid
        grid = self.cluster_to_grid(bubbles)
        
        # Calculate fill ratios
        self.calculate_fill_ratios(roi, grid)
        
        # Read answers
        answers = self.read_answers(grid)
        
        return answers


if __name__ == "__main__":
    # Test the detector
    detector = TypstFinalDetector(verbose=True)
    
    # Test on synthetic
    synthetic_path = Path("synthetic_test.png")
    if synthetic_path.exists():
        print("Testing on synthetic image:")
        print("-" * 40)
        answers = detector.process(synthetic_path)
        expected = {1: 'B', 2: 'D', 3: 'A', 4: 'C', 5: 'E'}
        correct = sum(1 for k, v in expected.items() if answers.get(k) == v)
        print(f"Detected: {answers}")
        print(f"Accuracy: {correct}/5 ({correct*20}%)")
    
    # Test on random Typst quiz
    random_path = Path("tests/generated/random_test_20250826_202248_001.png")
    if random_path.exists():
        print("\nTesting on Typst quiz:")
        print("-" * 40)
        answers = detector.process(random_path)
        print(f"Detected: {answers}")