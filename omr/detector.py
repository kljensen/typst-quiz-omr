"""
Dynamic OMR bubble detection without templates.
Detects corner brackets to locate bubble region, then finds and analyzes bubbles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
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
    

@dataclass
class BracketCorner:
    """Detected L-shaped corner bracket"""
    position: Tuple[int, int]
    corner_type: str  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    confidence: float


class DynamicOMRDetector:
    """Detect and read OMR bubbles without templates"""
    
    def __init__(self, fill_threshold: float = 0.30):
        self.fill_threshold = fill_threshold
        
    def process(self, image_path: Path) -> Dict[int, str]:
        """Main processing pipeline"""
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Step 1: Find bubble region using corner brackets
        bubble_region = self.find_bubble_region(img)
        if bubble_region is None:
            logger.warning("Could not find bubble region via brackets, trying fallback")
            bubble_region = self.find_bubble_region_fallback(img)
            
        if bubble_region is None:
            raise ValueError("Could not locate bubble region")
            
        # Step 2: Extract and align the region
        x, y, w, h = bubble_region
        aligned = self.extract_and_align(img, bubble_region)
        
        # Step 3: Detect all bubbles in the region
        bubbles = self.detect_bubbles(aligned)
        
        # Step 4: Organize bubbles into grid
        grid = self.cluster_to_grid(bubbles)
        
        # Step 5: Analyze fill status
        answers = self.read_answers(aligned, grid)
        
        return answers
    
    def find_bubble_region(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find region bounded by corner brackets"""
        corners = self.detect_corner_brackets(img)
        
        if len(corners) == 4:
            # Found all 4 corners, compute bounding box
            xs = [c.position[0] for c in corners]
            ys = [c.position[1] for c in corners]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Add some padding
            padding = 10
            x = max(0, x_min - padding)
            y = max(0, y_min - padding)
            w = min(img.shape[1] - x, x_max - x_min + 2 * padding)
            h = min(img.shape[0] - y, y_max - y_min + 2 * padding)
            
            return (x, y, w, h)
        
        return None
    
    def detect_corner_brackets(self, img: np.ndarray) -> List[BracketCorner]:
        """Detect L-shaped corner markers"""
        corners = []
        
        # Preprocess image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Define L-shaped kernels for each corner orientation
        kernel_size = 15
        thickness = 3
        
        # Top-left L shape
        kernel_tl = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        kernel_tl[:thickness, :] = 255  # Horizontal line
        kernel_tl[:, :thickness] = 255  # Vertical line
        
        # Top-right L shape (flipped horizontally)
        kernel_tr = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        kernel_tr[:thickness, :] = 255
        kernel_tr[:, -thickness:] = 255
        
        # Bottom-left L shape (flipped vertically)
        kernel_bl = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        kernel_bl[-thickness:, :] = 255
        kernel_bl[:, :thickness] = 255
        
        # Bottom-right L shape (flipped both)
        kernel_br = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        kernel_br[-thickness:, :] = 255
        kernel_br[:, -thickness:] = 255
        
        # Match each kernel
        kernels = [
            (kernel_tl, 'top-left'),
            (kernel_tr, 'top-right'),
            (kernel_bl, 'bottom-left'),
            (kernel_br, 'bottom-right')
        ]
        
        for kernel, corner_type in kernels:
            result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            
            # Find peaks
            locations = np.where(result >= threshold)
            
            # Get best match for this corner type
            if len(locations[0]) > 0:
                idx = np.argmax(result[locations])
                y, x = locations[0][idx], locations[1][idx]
                confidence = result[y, x]
                
                corners.append(BracketCorner(
                    position=(x + kernel_size//2, y + kernel_size//2),
                    corner_type=corner_type,
                    confidence=confidence
                ))
                
        return corners
    
    def find_bubble_region_fallback(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback: Find region with highest density of circles"""
        # Use Hough circles to find all bubbles
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Find bounding box of all circles
            x_coords = circles[:, 0]
            y_coords = circles[:, 1]
            
            padding = 20
            x = max(0, x_coords.min() - padding)
            y = max(0, y_coords.min() - padding)
            w = min(img.shape[1] - x, x_coords.max() - x + padding)
            h = min(img.shape[0] - y, y_coords.max() - y + padding)
            
            return (x, y, w, h)
            
        return None
    
    def extract_and_align(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract region and correct any rotation"""
        x, y, w, h = region
        extracted = img[y:y+h, x:x+w].copy()
        
        # TODO: Add rotation correction if needed
        # For now, return as-is
        return extracted
    
    def detect_bubbles(self, img: np.ndarray) -> List[Bubble]:
        """Detect all circular bubbles in image"""
        bubbles = []
        
        # Preprocess
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        
        # Method 1: Hough Circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=25,
            minRadius=12,
            maxRadius=20
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                bubbles.append(Bubble(center=(x, y), radius=r))
                
        return bubbles
    
    def cluster_to_grid(self, bubbles: List[Bubble]) -> BubbleGrid:
        """Organize bubbles into rows and columns"""
        if not bubbles:
            return BubbleGrid([], 0, 0)
            
        # Sort by Y coordinate (rows)
        bubbles_sorted = sorted(bubbles, key=lambda b: b.center[1])
        
        # Cluster into rows
        rows = []
        current_row = [bubbles_sorted[0]]
        row_threshold = 25  # Max Y difference within a row
        
        for bubble in bubbles_sorted[1:]:
            y_diff = abs(bubble.center[1] - current_row[-1].center[1])
            if y_diff > row_threshold:
                # New row - sort current row by X before saving
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
        
        # Validate grid consistency
        num_cols = len(rows[0]) if rows else 0
        for row in rows:
            if len(row) != num_cols:
                logger.warning(f"Inconsistent columns: expected {num_cols}, got {len(row)}")
                
        return BubbleGrid(
            bubbles=rows,
            num_questions=len(rows),
            num_options=num_cols
        )
    
    def read_answers(self, img: np.ndarray, grid: BubbleGrid) -> Dict[int, str]:
        """Determine which bubbles are filled"""
        answers = {}
        
        # Convert to binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        for row_idx, row in enumerate(grid.bubbles):
            filled_cols = []
            fill_scores = []
            
            for bubble in row:
                # Create circular mask for this bubble
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
            
            # Convert column indices to letters (0=A, 1=B, etc.)
            if filled_cols:
                # If multiple bubbles filled, take the one with highest fill ratio
                if len(filled_cols) > 1:
                    best_idx = fill_scores.index(max(fill_scores))
                    filled_cols = [filled_cols[best_idx]]
                    
                answer_letter = chr(ord('A') + filled_cols[0])
                answers[row_idx + 1] = answer_letter
            else:
                answers[row_idx + 1] = None
                
        return answers


if __name__ == "__main__":
    # Test with generated data
    detector = DynamicOMRDetector()
    
    test_file = Path("test_data.pdf")
    if test_file.exists():
        # Note: Would need to convert PDF pages to images first
        print(f"Testing with {test_file}")
        # results = detector.process(test_file)
        # print(results)