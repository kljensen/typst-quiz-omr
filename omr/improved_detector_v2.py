#!/usr/bin/env python3
"""
Improved OMR detector using proven techniques from exam-maker
Adapts morphological operations and template matching for better accuracy
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


class ImprovedOMRDetector:
    """Improved detector using proven morphological techniques"""
    
    def __init__(self, fill_threshold: float = 0.30, verbose: bool = False):
        self.fill_threshold = fill_threshold
        self.verbose = verbose
    
    def adaptive_threshold(self, img: np.ndarray, w: int = 5, h: int = 5, 
                          blur: bool = True) -> np.ndarray:
        """
        Apply adaptive thresholding with optional blur.
        Adapted from exam-maker's proven method.
        """
        if blur:
            stddev = 1
            blurred_img = cv2.GaussianBlur(img, (w, h), stddev)
        else:
            blurred_img = img
        
        # Use OTSU for automatic threshold selection
        _, thresholded_img = cv2.threshold(
            blurred_img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded_img
    
    def isolate_bubble_region(self, img: np.ndarray) -> np.ndarray:
        """
        Isolate the bubble region using morphological operations.
        Based on exam-maker's isolate_grading_box technique.
        """
        # Apply adaptive threshold
        img_thresh = self.adaptive_threshold(img)
        
        if self.verbose:
            cv2.imwrite("debug_threshold.png", img_thresh)
        
        # Kernel sizes based on image dimensions
        h_kernel_length = img.shape[1] // 50  # Horizontal lines
        v_kernel_length = img.shape[0] // 50  # Vertical lines
        
        # Create kernels for detecting lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_length))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
        
        # Detect vertical lines
        v_eroded = cv2.erode(img_thresh, v_kernel, iterations=3)
        v_dilated = cv2.dilate(v_eroded, v_kernel, iterations=3)
        
        # Detect horizontal lines
        h_eroded = cv2.erode(img_thresh, h_kernel, iterations=3)
        h_dilated = cv2.dilate(h_eroded, h_kernel, iterations=3)
        
        # Combine lines to form grid
        combined = cv2.add(h_dilated, v_dilated)
        
        # Clean up with morphological operations
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        small_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        combined_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, closed_kernel)
        combined_eroded = cv2.erode(combined_closed, open_kernel)
        combined_dilated = cv2.dilate(combined_eroded, small_dilation)
        
        # Use as mask
        final = cv2.bitwise_and(img_thresh, combined_dilated)
        
        if self.verbose:
            cv2.imwrite("debug_isolated.png", final)
        
        return final
    
    def detect_bubbles_contours(self, img: np.ndarray) -> List[Bubble]:
        """
        Detect bubbles using contour detection.
        More robust than Hough circles for printed documents.
        """
        bubbles = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Invert if needed (bubbles should be dark)
        mean_val = np.mean(gray)
        if mean_val > 127:
            gray = cv2.bitwise_not(gray)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for circular contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small or too large
            if area < 100 or area > 2000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Accept reasonably circular shapes
            if circularity > 0.5:
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Additional size filtering
                if 5 < radius < 30:
                    bubbles.append(Bubble(
                        center=(int(x), int(y)),
                        radius=radius
                    ))
        
        return bubbles
    
    def detect_bubbles_hybrid(self, img: np.ndarray) -> List[Bubble]:
        """
        Hybrid approach: Try both Hough circles and contours
        """
        # Method 1: Contour detection (better for clean scans)
        contour_bubbles = self.detect_bubbles_contours(img)
        
        # Method 2: Hough circles (better for some cases)
        hough_bubbles = self.detect_bubbles_hough(img)
        
        # Choose the method that found more bubbles
        if len(contour_bubbles) > len(hough_bubbles):
            logger.info(f"Using contour detection: {len(contour_bubbles)} bubbles")
            return contour_bubbles
        else:
            logger.info(f"Using Hough circles: {len(hough_bubbles)} bubbles")
            return hough_bubbles
    
    def detect_bubbles_hough(self, img: np.ndarray) -> List[Bubble]:
        """
        Detect bubbles using improved Hough circle parameters
        """
        bubbles = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply bilateral filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Try multiple parameter sets
        param_sets = [
            # (dp, minDist, param1, param2, minRadius, maxRadius)
            (1.0, 20, 30, 15, 8, 20),   # Standard
            (1.2, 25, 40, 20, 10, 25),  # Larger bubbles
            (1.5, 15, 25, 12, 6, 15),   # Smaller bubbles
        ]
        
        all_circles = []
        
        for dp, minDist, p1, p2, minR, maxR in param_sets:
            circles = cv2.HoughCircles(
                filtered,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=p1,
                param2=p2,
                minRadius=minR,
                maxRadius=maxR
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    all_circles.append((x, y, r))
        
        # Remove duplicates
        unique_circles = []
        for circle in all_circles:
            is_duplicate = False
            for unique in unique_circles:
                dist = np.sqrt((circle[0] - unique[0])**2 + (circle[1] - unique[1])**2)
                if dist < 10:  # Too close
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_circles.append(circle)
                bubbles.append(Bubble(
                    center=(circle[0], circle[1]),
                    radius=circle[2]
                ))
        
        return bubbles
    
    def cluster_to_grid(self, bubbles: List[Bubble]) -> BubbleGrid:
        """Organize bubbles into rows and columns"""
        if not bubbles:
            return BubbleGrid([], 0, 0)
        
        # Sort by Y coordinate (rows)
        bubbles_sorted = sorted(bubbles, key=lambda b: b.center[1])
        
        # Dynamic row threshold based on bubble spacing
        if len(bubbles) > 1:
            y_diffs = []
            for i in range(len(bubbles_sorted) - 1):
                y_diff = abs(bubbles_sorted[i+1].center[1] - bubbles_sorted[i].center[1])
                if y_diff > 5:  # Ignore very small differences
                    y_diffs.append(y_diff)
            
            if y_diffs:
                # Use median difference as threshold
                row_threshold = np.median(y_diffs) * 0.5
            else:
                row_threshold = 25
        else:
            row_threshold = 25
        
        # Cluster into rows
        rows = []
        current_row = [bubbles_sorted[0]]
        
        for bubble in bubbles_sorted[1:]:
            y_diff = abs(bubble.center[1] - current_row[-1].center[1])
            if y_diff > row_threshold:
                # New row - sort current row by X
                rows.append(sorted(current_row, key=lambda b: b.center[0]))
                current_row = [bubble]
            else:
                current_row.append(bubble)
        
        # Don't forget last row
        rows.append(sorted(current_row, key=lambda b: b.center[0]))
        
        # Assign row/col indices
        for row_idx, row in enumerate(rows):
            for col_idx, bubble in enumerate(row):
                bubble.row = row_idx
                bubble.col = col_idx
        
        # Find most common column count (mode)
        col_counts = [len(row) for row in rows]
        if col_counts:
            from collections import Counter
            mode_cols = Counter(col_counts).most_common(1)[0][0]
        else:
            mode_cols = 0
        
        # Filter out rows with wrong number of columns
        filtered_rows = [row for row in rows if len(row) == mode_cols]
        
        if self.verbose:
            logger.info(f"Grid: {len(filtered_rows)} rows x {mode_cols} cols")
            logger.info(f"Filtered out {len(rows) - len(filtered_rows)} irregular rows")
        
        return BubbleGrid(
            bubbles=filtered_rows,
            num_questions=len(filtered_rows),
            num_options=mode_cols
        )
    
    def read_answers(self, img: np.ndarray, grid: BubbleGrid) -> Dict[int, str]:
        """Determine which bubbles are filled"""
        answers = {}
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Use adaptive threshold for better results
        binary = self.adaptive_threshold(gray)
        
        # Invert if needed (filled bubbles should be white)
        mean_val = np.mean(binary)
        if mean_val > 127:
            binary = cv2.bitwise_not(binary)
        
        for row_idx, row in enumerate(grid.bubbles):
            filled_cols = []
            fill_scores = []
            
            for bubble in row:
                # Create circular mask
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.circle(mask, bubble.center, int(bubble.radius * 0.8), 255, -1)
                
                # Count filled pixels
                filled_pixels = cv2.bitwise_and(binary, mask)
                total_pixels = cv2.countNonZero(mask)
                filled_count = cv2.countNonZero(filled_pixels)
                
                fill_ratio = filled_count / total_pixels if total_pixels > 0 else 0
                bubble.fill_ratio = fill_ratio
                
                if fill_ratio > self.fill_threshold:
                    filled_cols.append(bubble.col)
                    fill_scores.append(fill_ratio)
            
            # Convert to letters
            if filled_cols:
                # Take highest confidence if multiple
                if len(filled_cols) > 1:
                    best_idx = fill_scores.index(max(fill_scores))
                    filled_cols = [filled_cols[best_idx]]
                
                answer_letter = chr(ord('A') + filled_cols[0])
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
        
        # Find bubble region using bracket detection
        from omr.bracket_detector_v2 import find_bubble_region_v2
        bubble_region = find_bubble_region_v2(img)
        
        if bubble_region is None:
            logger.warning("Could not find bubble region via brackets")
            # Try to detect bubbles in full image
            roi = gray
        else:
            x, y, w, h = bubble_region
            roi = gray[y:y+h, x:x+w]
            logger.info(f"Found bubble region: {w}x{h} at ({x},{y})")
        
        # Detect bubbles using hybrid approach
        bubbles = self.detect_bubbles_hybrid(roi)
        logger.info(f"Detected {len(bubbles)} bubbles")
        
        # Organize into grid
        grid = self.cluster_to_grid(bubbles)
        logger.info(f"Organized into {grid.num_questions}x{grid.num_options} grid")
        
        # Read answers
        answers = self.read_answers(roi, grid)
        
        return answers


if __name__ == "__main__":
    # Test on synthetic image
    from pathlib import Path
    
    detector = ImprovedOMRDetector(verbose=True)
    
    # Test synthetic
    synthetic_path = Path("synthetic_test.png")
    if synthetic_path.exists():
        print("Testing on synthetic image...")
        answers = detector.process(synthetic_path)
        print(f"Detected answers: {answers}")
        
        # Expected: B, D, A, C, E
        expected = {1: 'B', 2: 'D', 3: 'A', 4: 'C', 5: 'E'}
        correct = sum(1 for k, v in expected.items() if answers.get(k) == v)
        print(f"Accuracy: {correct}/5 ({correct*20}%)")