"""
Improved OMR detector with better corner bracket detection.
Multiple detection strategies for robustness.
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
class Corner:
    """Represents a detected corner"""
    position: Tuple[int, int]
    corner_type: str  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    confidence: float
    method: str  # Detection method used


class ImprovedBracketDetector:
    """Multiple methods for detecting L-shaped corner brackets"""
    
    def __init__(self):
        self.debug_mode = False
        
    def detect_corners(self, img: np.ndarray) -> List[Corner]:
        """Try multiple detection methods and combine results"""
        corners = []
        
        # Method 1: Harris corner detection for L-shapes
        harris_corners = self.detect_harris_corners(img)
        corners.extend(harris_corners)
        
        # Method 2: Template matching with generated L-shapes
        template_corners = self.detect_template_corners(img)
        corners.extend(template_corners)
        
        # Method 3: Contour-based detection
        contour_corners = self.detect_contour_corners(img)
        corners.extend(contour_corners)
        
        # Method 4: Line intersection detection
        line_corners = self.detect_line_intersection_corners(img)
        corners.extend(line_corners)
        
        # Merge and filter corners
        return self.merge_corners(corners)
    
    def detect_harris_corners(self, img: np.ndarray) -> List[Corner]:
        """Use Harris corner detection to find L-shaped corners"""
        corners = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply Harris corner detection
        dst = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.04)
        
        # Threshold for corner response
        dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        threshold = 0.1 * dst_norm.max()
        
        # Find corner points
        corner_points = np.where(dst_norm > threshold)
        
        # Cluster nearby corners and classify them
        if len(corner_points[0]) > 0:
            points = list(zip(corner_points[1], corner_points[0]))
            clustered = self.cluster_points(points)
            
            for cluster in clustered:
                corner_type = self.classify_corner_position(cluster, img.shape)
                corners.append(Corner(
                    position=cluster,
                    corner_type=corner_type,
                    confidence=0.6,
                    method='harris'
                ))
                
        return corners
    
    def detect_template_corners(self, img: np.ndarray) -> List[Corner]:
        """Template matching with generated L-shaped patterns"""
        corners = []
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Generate L-shaped templates at different scales
        scales = [20, 30, 40, 50]  # Different sizes in pixels
        
        for scale in scales:
            templates = self.generate_l_templates(scale)
            
            for template, corner_type in templates:
                # Template matching
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.7
                
                # Find matches
                locations = np.where(result >= threshold)
                
                for y, x in zip(locations[0], locations[1]):
                    confidence = result[y, x]
                    corners.append(Corner(
                        position=(x + scale//2, y + scale//2),
                        corner_type=corner_type,
                        confidence=confidence,
                        method='template'
                    ))
                    
        return corners
    
    def detect_contour_corners(self, img: np.ndarray) -> List[Corner]:
        """Detect L-shapes using contour analysis"""
        corners = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for L-shaped contours (should have 6 vertices ideally)
            if 5 <= len(approx) <= 7:
                # Check if it could be an L-shape
                if self.is_l_shaped(approx):
                    # Find the corner point of the L
                    corner_point = self.find_l_corner(approx)
                    corner_type = self.classify_corner_position(corner_point, img.shape)
                    
                    corners.append(Corner(
                        position=corner_point,
                        corner_type=corner_type,
                        confidence=0.7,
                        method='contour'
                    ))
                    
        return corners
    
    def detect_line_intersection_corners(self, img: np.ndarray) -> List[Corner]:
        """Detect corners by finding perpendicular line intersections"""
        corners = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Find perpendicular line intersections
            intersections = self.find_perpendicular_intersections(lines)
            
            for point in intersections:
                corner_type = self.classify_corner_position(point, img.shape)
                corners.append(Corner(
                    position=point,
                    corner_type=corner_type,
                    confidence=0.5,
                    method='lines'
                ))
                
        return corners
    
    def generate_l_templates(self, size: int) -> List[Tuple[np.ndarray, str]]:
        """Generate L-shaped templates for all four orientations"""
        templates = []
        thickness = max(2, size // 5)  # Proportional thickness
        
        # Top-left L
        tl = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(tl, (0, 0), (thickness, size), 255, -1)
        cv2.rectangle(tl, (0, 0), (size, thickness), 255, -1)
        templates.append((tl, 'top-left'))
        
        # Top-right L
        tr = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(tr, (size-thickness, 0), (size, size), 255, -1)
        cv2.rectangle(tr, (0, 0), (size, thickness), 255, -1)
        templates.append((tr, 'top-right'))
        
        # Bottom-left L
        bl = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(bl, (0, 0), (thickness, size), 255, -1)
        cv2.rectangle(bl, (0, size-thickness), (size, size), 255, -1)
        templates.append((bl, 'bottom-left'))
        
        # Bottom-right L
        br = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(br, (size-thickness, 0), (size, size), 255, -1)
        cv2.rectangle(br, (0, size-thickness), (size, size), 255, -1)
        templates.append((br, 'bottom-right'))
        
        return templates
    
    def is_l_shaped(self, contour: np.ndarray) -> bool:
        """Check if a contour could be L-shaped"""
        # Calculate moments
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
            
        # Check aspect ratio and solidity
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # L-shapes should have aspect ratio close to 1
        if not (0.7 < aspect_ratio < 1.3):
            return False
            
        # Check solidity (L-shape has lower solidity than rectangle)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area == 0:
            return False
            
        solidity = contour_area / hull_area
        
        # L-shape typically has solidity between 0.4 and 0.7
        return 0.4 < solidity < 0.7
    
    def find_l_corner(self, contour: np.ndarray) -> Tuple[int, int]:
        """Find the corner point of an L-shaped contour"""
        # Find the point that is closest to the centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return contour[0][0]
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Find vertex closest to centroid
        min_dist = float('inf')
        corner_point = None
        
        for point in contour:
            px, py = point[0]
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                corner_point = (px, py)
                
        return corner_point
    
    def find_perpendicular_intersections(self, lines: np.ndarray) -> List[Tuple[int, int]]:
        """Find intersections of perpendicular lines"""
        intersections = []
        
        for i, line1 in enumerate(lines):
            x1, y1, x2, y2 = line1[0]
            angle1 = np.arctan2(y2 - y1, x2 - x1)
            
            for line2 in lines[i+1:]:
                x3, y3, x4, y4 = line2[0]
                angle2 = np.arctan2(y4 - y3, x4 - x3)
                
                # Check if lines are perpendicular (within tolerance)
                angle_diff = abs(angle1 - angle2)
                if abs(angle_diff - np.pi/2) < 0.1 or abs(angle_diff - 3*np.pi/2) < 0.1:
                    # Find intersection point
                    intersection = self.line_intersection(
                        (x1, y1), (x2, y2), (x3, y3), (x4, y4)
                    )
                    if intersection:
                        intersections.append(intersection)
                        
        return intersections
    
    def line_intersection(self, p1, p2, p3, p4):
        """Find intersection point of two lines"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))
            
        return None
    
    def cluster_points(self, points: List[Tuple[int, int]], 
                      max_distance: int = 50) -> List[Tuple[int, int]]:
        """Cluster nearby points and return centroids"""
        if not points:
            return []
            
        clusters = []
        used = set()
        
        for i, p1 in enumerate(points):
            if i in used:
                continue
                
            cluster = [p1]
            used.add(i)
            
            for j, p2 in enumerate(points[i+1:], i+1):
                if j not in used:
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    if dist < max_distance:
                        cluster.append(p2)
                        used.add(j)
                        
            # Calculate centroid of cluster
            if cluster:
                cx = int(np.mean([p[0] for p in cluster]))
                cy = int(np.mean([p[1] for p in cluster]))
                clusters.append((cx, cy))
                
        return clusters
    
    def classify_corner_position(self, point: Tuple[int, int], 
                                img_shape: Tuple[int, int]) -> str:
        """Classify which corner a point belongs to"""
        x, y = point
        h, w = img_shape[:2]
        
        # Determine quadrant
        if x < w // 2 and y < h // 2:
            return 'top-left'
        elif x >= w // 2 and y < h // 2:
            return 'top-right'
        elif x < w // 2 and y >= h // 2:
            return 'bottom-left'
        else:
            return 'bottom-right'
    
    def merge_corners(self, corners: List[Corner]) -> List[Corner]:
        """Merge nearby corners and select best ones for each position"""
        if not corners:
            return []
            
        # Group by corner type
        grouped = {}
        for corner in corners:
            if corner.corner_type not in grouped:
                grouped[corner.corner_type] = []
            grouped[corner.corner_type].append(corner)
            
        # Select best corner for each type
        final_corners = []
        for corner_type, group in grouped.items():
            if group:
                # Sort by confidence and take the best
                best = max(group, key=lambda c: c.confidence)
                final_corners.append(best)
                
        return final_corners


def find_bubble_region_with_improved_brackets(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find bubble region using improved bracket detection"""
    detector = ImprovedBracketDetector()
    corners = detector.detect_corners(img)
    
    if len(corners) >= 4:
        # Extract corner positions
        positions = [c.position for c in corners]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Calculate bounding box with padding
        padding = 20
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(img.shape[1], max(xs) + padding)
        y_max = min(img.shape[0], max(ys) + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    logger.warning(f"Only found {len(corners)} corners")
    return None


if __name__ == "__main__":
    # Test the improved detector
    test_image = Path("tests/generated/test_data_page_1.png")
    if test_image.exists():
        img = cv2.imread(str(test_image))
        region = find_bubble_region_with_improved_brackets(img)
        if region:
            print(f"Found bubble region: {region}")
        else:
            print("Could not find bubble region")