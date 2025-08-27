"""
Tests for corner bracket detection
"""

import sys
import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omr.improved_detector import (
    ImprovedBracketDetector, 
    find_bubble_region_with_improved_brackets,
    Corner
)


class TestBracketGeneration:
    """Test L-shaped bracket template generation"""
    
    def test_generate_l_templates(self):
        """Test that L-shaped templates are generated correctly"""
        detector = ImprovedBracketDetector()
        templates = detector.generate_l_templates(size=30)
        
        # Should generate 4 templates (one for each corner)
        assert len(templates) == 4
        
        # Check corner types
        corner_types = [t[1] for t in templates]
        assert 'top-left' in corner_types
        assert 'top-right' in corner_types
        assert 'bottom-left' in corner_types
        assert 'bottom-right' in corner_types
        
        # Check template dimensions
        for template, _ in templates:
            assert template.shape == (30, 30)
            assert template.dtype == np.uint8
            assert template.max() == 255
            assert template.min() == 0


class TestSyntheticBrackets:
    """Test detection on synthetic images with perfect brackets"""
    
    @pytest.fixture
    def synthetic_image_with_brackets(self) -> np.ndarray:
        """Create a synthetic image with corner brackets"""
        img = np.ones((500, 400), dtype=np.uint8) * 255  # White background
        
        # Add corner brackets (black)
        bracket_size = 30
        thickness = 5
        
        # Top-left
        cv2.rectangle(img, (50, 50), (50 + thickness, 50 + bracket_size), 0, -1)
        cv2.rectangle(img, (50, 50), (50 + bracket_size, 50 + thickness), 0, -1)
        
        # Top-right
        cv2.rectangle(img, (350 - thickness, 50), (350, 50 + bracket_size), 0, -1)
        cv2.rectangle(img, (350 - bracket_size, 50), (350, 50 + thickness), 0, -1)
        
        # Bottom-left
        cv2.rectangle(img, (50, 450 - thickness), (50 + bracket_size, 450), 0, -1)
        cv2.rectangle(img, (50, 450 - bracket_size), (50 + thickness, 450), 0, -1)
        
        # Bottom-right
        cv2.rectangle(img, (350 - thickness, 450 - bracket_size), (350, 450), 0, -1)
        cv2.rectangle(img, (350 - bracket_size, 450 - thickness), (350, 450), 0, -1)
        
        return img
    
    def test_detect_on_synthetic_image(self, synthetic_image_with_brackets):
        """Test detection on perfect synthetic brackets"""
        detector = ImprovedBracketDetector()
        corners = detector.detect_corners(synthetic_image_with_brackets)
        
        # Should find at least 4 corners
        assert len(corners) >= 4
        
        # Check that we have corners in each quadrant
        corner_types = [c.corner_type for c in corners]
        assert any('top-left' in ct for ct in corner_types)
        assert any('top-right' in ct for ct in corner_types)
        assert any('bottom-left' in ct for ct in corner_types)
        assert any('bottom-right' in ct for ct in corner_types)
    
    def test_find_bubble_region_synthetic(self, synthetic_image_with_brackets):
        """Test finding bubble region on synthetic image"""
        region = find_bubble_region_with_improved_brackets(synthetic_image_with_brackets)
        
        assert region is not None
        x, y, w, h = region
        
        # Region should be roughly in the middle
        assert 30 < x < 70
        assert 30 < y < 70
        assert 250 < w < 350
        assert 350 < h < 450


class TestNoiseRobustness:
    """Test robustness to various image artifacts"""
    
    def add_gaussian_noise(self, img: np.ndarray, sigma: float = 10) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, sigma, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_rotation(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]),
                                 borderValue=255)
        return rotated
    
    @pytest.fixture
    def synthetic_image(self) -> np.ndarray:
        """Create a basic synthetic image with brackets"""
        img = np.ones((500, 400), dtype=np.uint8) * 255
        
        # Simplified bracket creation
        thickness = 5
        size = 30
        
        # Draw L-shapes at corners
        corners = [(50, 50), (350, 50), (50, 450), (350, 450)]
        for i, (x, y) in enumerate(corners):
            if i == 0:  # Top-left
                cv2.line(img, (x, y), (x + size, y), 0, thickness)
                cv2.line(img, (x, y), (x, y + size), 0, thickness)
            elif i == 1:  # Top-right
                cv2.line(img, (x - size, y), (x, y), 0, thickness)
                cv2.line(img, (x, y), (x, y + size), 0, thickness)
            elif i == 2:  # Bottom-left
                cv2.line(img, (x, y), (x + size, y), 0, thickness)
                cv2.line(img, (x, y - size), (x, y), 0, thickness)
            else:  # Bottom-right
                cv2.line(img, (x - size, y), (x, y), 0, thickness)
                cv2.line(img, (x, y - size), (x, y), 0, thickness)
        
        return img
    
    def test_noise_robustness(self, synthetic_image):
        """Test detection with added noise"""
        detector = ImprovedBracketDetector()
        
        # Test with different noise levels
        for sigma in [5, 10, 15]:
            noisy = self.add_gaussian_noise(synthetic_image, sigma)
            corners = detector.detect_corners(noisy)
            
            # Should still find some corners even with noise
            assert len(corners) > 0, f"Failed to find corners with sigma={sigma}"
    
    def test_rotation_robustness(self, synthetic_image):
        """Test detection with rotation"""
        detector = ImprovedBracketDetector()
        
        # Test small rotations
        for angle in [-5, -2, 0, 2, 5]:
            rotated = self.add_rotation(synthetic_image, angle)
            corners = detector.detect_corners(rotated)
            
            # Should find corners even with rotation
            assert len(corners) > 0, f"Failed to find corners with angle={angle}"


class TestRealImages:
    """Test on actual generated quiz images"""
    
    @pytest.fixture
    def test_image_path(self) -> Path:
        """Path to test image (generate if needed)"""
        test_dir = Path("tests/generated")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have a test image
        test_image = test_dir / "test_data_page_1.png"
        
        if not test_image.exists():
            # Generate test data first
            import subprocess
            subprocess.run(["typst", "compile", "generate_test_data.typ", "test_data.pdf"])
            
            # Convert to images
            from pdf2image import convert_from_path
            images = convert_from_path("test_data.pdf", dpi=300)
            if images:
                images[0].save(test_image)
        
        return test_image
    
    def test_detect_on_real_image(self, test_image_path):
        """Test detection on real generated quiz image"""
        if not test_image_path.exists():
            pytest.skip("Test image not available")
        
        img = cv2.imread(str(test_image_path))
        detector = ImprovedBracketDetector()
        corners = detector.detect_corners(img)
        
        # Log what we found
        print(f"\nFound {len(corners)} corners on real image:")
        for corner in corners:
            print(f"  - {corner.corner_type} at {corner.position} "
                  f"(confidence: {corner.confidence:.2f}, method: {corner.method})")
        
        # We should find at least some corners
        assert len(corners) > 0, "No corners found on real image"
    
    def test_bubble_region_on_real_image(self, test_image_path):
        """Test finding bubble region on real image"""
        if not test_image_path.exists():
            pytest.skip("Test image not available")
        
        img = cv2.imread(str(test_image_path))
        region = find_bubble_region_with_improved_brackets(img)
        
        if region:
            x, y, w, h = region
            print(f"\nFound bubble region: x={x}, y={y}, w={w}, h={h}")
            
            # Basic sanity checks
            assert w > 100, "Region too narrow"
            assert h > 100, "Region too short"
            assert x >= 0 and y >= 0, "Invalid coordinates"
        else:
            # If detection fails, at least log why
            detector = ImprovedBracketDetector()
            corners = detector.detect_corners(img)
            print(f"\nRegion detection failed. Found {len(corners)} corners")


class TestDetectorMethods:
    """Test individual detection methods"""
    
    def test_line_intersection(self):
        """Test line intersection calculation"""
        detector = ImprovedBracketDetector()
        
        # Test perpendicular lines that intersect
        intersection = detector.line_intersection((0, 0), (10, 0), (5, -5), (5, 5))
        assert intersection is not None
        assert intersection == (5, 0)
        
        # Test parallel lines (no intersection)
        intersection = detector.line_intersection((0, 0), (10, 0), (0, 5), (10, 5))
        assert intersection is None
    
    def test_cluster_points(self):
        """Test point clustering"""
        detector = ImprovedBracketDetector()
        
        # Test clustering nearby points
        points = [(10, 10), (12, 11), (50, 50), (52, 48), (100, 100)]
        clusters = detector.cluster_points(points, max_distance=10)
        
        # Should get 3 clusters
        assert len(clusters) == 3
    
    def test_classify_corner_position(self):
        """Test corner position classification"""
        detector = ImprovedBracketDetector()
        img_shape = (600, 800)  # height, width
        
        assert detector.classify_corner_position((100, 100), img_shape) == 'top-left'
        assert detector.classify_corner_position((700, 100), img_shape) == 'top-right'
        assert detector.classify_corner_position((100, 500), img_shape) == 'bottom-left'
        assert detector.classify_corner_position((700, 500), img_shape) == 'bottom-right'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])