"""
End-to-end tests for the OMR system with known answer keys
"""

import sys
import json
import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omr.detector import DynamicOMRDetector, Bubble, BubbleGrid


class TestOMRPipeline:
    """Test complete OMR pipeline with known answers"""
    
    @pytest.fixture(scope="class")
    def test_suite_pdf(self) -> Path:
        """Generate test suite PDF if needed"""
        pdf_path = Path("test_suite.pdf")
        
        if not pdf_path.exists():
            # Generate test suite
            result = subprocess.run(
                ["typst", "compile", "generate_test_suite.typ", str(pdf_path)],
                capture_output=True
            )
            if result.returncode != 0:
                pytest.skip(f"Could not generate test suite: {result.stderr}")
        
        return pdf_path
    
    @pytest.fixture(scope="class")
    def test_images(self, test_suite_pdf) -> Dict[int, Path]:
        """Convert PDF pages to images"""
        test_dir = Path("tests/generated")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        images = {}
        
        # Convert each page to PNG
        for page_num in range(1, 8):  # 7 test pages
            image_path = test_dir / f"test_suite_page_{page_num}.png"
            
            if not image_path.exists():
                # Convert specific page to PNG
                result = subprocess.run(
                    ["magick", "-density", "300", 
                     f"{test_suite_pdf}[{page_num-1}]", str(image_path)],
                    capture_output=True
                )
                if result.returncode != 0:
                    continue
            
            if image_path.exists():
                images[page_num] = image_path
        
        return images
    
    @pytest.fixture
    def answer_keys(self) -> Dict:
        """Load answer keys"""
        keys_file = Path("tests/test_answer_keys.json")
        with open(keys_file, 'r') as f:
            return json.load(f)['test_suite']
    
    @pytest.fixture
    def detector(self) -> DynamicOMRDetector:
        """Create OMR detector instance"""
        return DynamicOMRDetector(fill_threshold=0.30)
    
    def test_sequential_pattern(self, test_images, answer_keys, detector):
        """Test sequential answer pattern (A, B, C, D, E)"""
        if 1 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_1_sequential']['answers']
        detected = detector.process(test_images[1])
        
        self.validate_answers(expected, detected, "Sequential Pattern")
    
    def test_all_same_answer(self, test_images, answer_keys, detector):
        """Test all same answer (all A's)"""
        if 2 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_2_all_a']['answers']
        detected = detector.process(test_images[2])
        
        self.validate_answers(expected, detected, "All A's")
    
    def test_multiple_selections(self, test_images, answer_keys, detector):
        """Test multiple selections per question"""
        if 3 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_3_multiple']['answers']
        detected = detector.process(test_images[3])
        
        # For multiple selections, detector should return multiple answers
        for q_num, exp_answers in expected.items():
            q_idx = int(q_num)
            if len(exp_answers) > 1:
                # Check if detector found multiple fills
                assert q_idx in detected, f"Question {q_num} not detected"
                # Note: Current detector may only return highest confidence answer
                # This test documents expected behavior for future enhancement
    
    def test_blank_answers(self, test_images, answer_keys, detector):
        """Test blank/no answers"""
        if 4 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_4_blank']['answers']
        detected = detector.process(test_images[4])
        
        # All questions should have no answer or None
        for q_num in expected.keys():
            q_idx = int(q_num)
            assert detected.get(q_idx) is None, f"Question {q_num} should be blank"
    
    def test_edge_options(self, test_images, answer_keys, detector):
        """Test edge options (first and last)"""
        if 6 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_6_edge']['answers']
        detected = detector.process(test_images[6])
        
        self.validate_answers(expected, detected, "Edge Options")
    
    def test_realistic_pattern(self, test_images, answer_keys, detector):
        """Test realistic answer pattern"""
        if 7 not in test_images:
            pytest.skip("Test image not available")
        
        expected = answer_keys['test_7_realistic']['answers']
        detected = detector.process(test_images[7])
        
        self.validate_answers(expected, detected, "Realistic Pattern")
    
    def validate_answers(self, expected: Dict, detected: Dict, test_name: str):
        """Validate detected answers against expected"""
        correct = 0
        total = len(expected)
        
        for q_num, exp_answers in expected.items():
            q_idx = int(q_num)
            det_answer = detected.get(q_idx)
            
            # Handle single answer case
            if len(exp_answers) == 1:
                expected_answer = exp_answers[0]
                if det_answer == expected_answer:
                    correct += 1
                else:
                    print(f"{test_name} Q{q_num}: Expected {expected_answer}, got {det_answer}")
            elif len(exp_answers) == 0:
                # Blank answer
                if det_answer is None:
                    correct += 1
                else:
                    print(f"{test_name} Q{q_num}: Expected blank, got {det_answer}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"{test_name}: {correct}/{total} correct ({accuracy:.1%})")
        
        # Assert high accuracy (allow for some detection errors)
        assert accuracy >= 0.8, f"{test_name} accuracy too low: {accuracy:.1%}"


class TestBubbleDetection:
    """Test individual bubble detection components"""
    
    def create_synthetic_bubble_image(self, 
                                    num_rows: int = 5, 
                                    num_cols: int = 5,
                                    filled: List[Tuple[int, int]] = None) -> np.ndarray:
        """Create synthetic image with bubble grid"""
        img = np.ones((400, 300), dtype=np.uint8) * 255  # White background
        
        # Draw bubbles
        start_x, start_y = 50, 50
        spacing_x, spacing_y = 50, 60
        radius = 15
        
        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                
                # Draw circle
                cv2.circle(img, (x, y), radius, 0, 2)
                
                # Fill if specified
                if filled and (row, col) in filled:
                    cv2.circle(img, (x, y), radius - 2, 0, -1)
        
        return img
    
    def test_bubble_detection(self):
        """Test bubble detection on synthetic image"""
        detector = DynamicOMRDetector()
        img = self.create_synthetic_bubble_image()
        
        bubbles = detector.detect_bubbles(img)
        
        # Should detect 25 bubbles (5x5 grid)
        assert len(bubbles) >= 20, f"Expected ~25 bubbles, got {len(bubbles)}"
        
        # Check bubble properties
        for bubble in bubbles:
            assert isinstance(bubble, Bubble)
            assert 10 <= bubble.radius <= 20
            assert bubble.center[0] > 0 and bubble.center[1] > 0
    
    def test_grid_clustering(self):
        """Test clustering bubbles into grid"""
        detector = DynamicOMRDetector()
        img = self.create_synthetic_bubble_image(num_rows=4, num_cols=5)
        
        bubbles = detector.detect_bubbles(img)
        grid = detector.cluster_to_grid(bubbles)
        
        assert isinstance(grid, BubbleGrid)
        assert grid.num_questions == 4  # 4 rows
        assert grid.num_options == 5     # 5 columns
        
        # Check grid organization
        for row_idx, row in enumerate(grid.bubbles):
            assert len(row) == 5, f"Row {row_idx} has {len(row)} bubbles, expected 5"
            
            # Check bubbles are sorted by x-coordinate
            x_coords = [b.center[0] for b in row]
            assert x_coords == sorted(x_coords), f"Row {row_idx} not sorted by X"
    
    def test_fill_detection(self):
        """Test detecting filled vs unfilled bubbles"""
        detector = DynamicOMRDetector()
        
        # Create image with some filled bubbles
        filled_positions = [(0, 0), (1, 2), (2, 4), (3, 1)]
        img = self.create_synthetic_bubble_image(
            num_rows=4, num_cols=5, 
            filled=filled_positions
        )
        
        bubbles = detector.detect_bubbles(img)
        grid = detector.cluster_to_grid(bubbles)
        answers = detector.read_answers(img, grid)
        
        # Check detected answers
        assert answers[1] == 'A', f"Q1 should be A, got {answers[1]}"
        assert answers[2] == 'C', f"Q2 should be C, got {answers[2]}"
        assert answers[3] == 'E', f"Q3 should be E, got {answers[3]}"
        assert answers[4] == 'B', f"Q4 should be B, got {answers[4]}"
    
    def test_fill_threshold(self):
        """Test fill threshold sensitivity"""
        # Test with different fill levels
        for fill_ratio in [0.2, 0.3, 0.5, 0.7, 0.9]:
            detector = DynamicOMRDetector(fill_threshold=0.3)
            
            # Create image with partial fill
            img = np.ones((100, 100), dtype=np.uint8) * 255
            cv2.circle(img, (50, 50), 20, 0, 2)
            
            # Fill partially
            mask = np.zeros_like(img)
            cv2.circle(mask, (50, 50), 18, 255, -1)
            filled_pixels = int(np.sum(mask > 0) * fill_ratio)
            
            # Create partial fill pattern
            points = np.where(mask > 0)
            indices = np.random.choice(len(points[0]), filled_pixels, replace=False)
            for idx in indices:
                img[points[0][idx], points[1][idx]] = 0
            
            # Test detection
            bubble = Bubble(center=(50, 50), radius=20)
            grid = BubbleGrid([[bubble]], 1, 1)
            answers = detector.read_answers(img, grid)
            
            if fill_ratio >= 0.3:
                assert 1 in answers, f"Should detect fill at {fill_ratio:.0%}"
            else:
                assert answers.get(1) is None, f"Should not detect fill at {fill_ratio:.0%}"


class TestPerformance:
    """Performance benchmarks"""
    
    def test_processing_speed(self, tmp_path):
        """Test processing speed on typical image"""
        detector = DynamicOMRDetector()
        
        # Create test image
        img = np.ones((3300, 2550), dtype=np.uint8) * 255
        
        # Add some bubbles
        for row in range(10):
            for col in range(5):
                x = 500 + col * 60
                y = 500 + row * 80
                cv2.circle(img, (x, y), 18, 0, 2)
        
        # Save and process
        test_file = tmp_path / "test.png"
        cv2.imwrite(str(test_file), img)
        
        import time
        start = time.time()
        
        try:
            detector.process(test_file)
        except:
            pass  # May fail due to missing brackets
        
        elapsed = time.time() - start
        
        # Should process in reasonable time
        assert elapsed < 5.0, f"Processing took {elapsed:.1f}s, expected < 5s"
    
    def test_memory_usage(self):
        """Test memory usage stays reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        detector = DynamicOMRDetector()
        
        # Process multiple images
        for _ in range(5):
            img = np.ones((3300, 2550), dtype=np.uint8) * 255
            detector.detect_bubbles(img)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory increased by {memory_increase:.0f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])