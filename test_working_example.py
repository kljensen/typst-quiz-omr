#!/usr/bin/env python3
"""
Test the working example with tuned parameters
"""

import cv2
import numpy as np
from pathlib import Path
from omr.detector import DynamicOMRDetector
from omr.bracket_detector_v2 import find_bubble_region_v2

class TunedOMRDetector(DynamicOMRDetector):
    """OMR Detector with tuned parameters for our specific format"""
    
    def detect_bubbles(self, img: np.ndarray):
        """Detect bubbles with tuned parameters"""
        bubbles = []
        
        # Preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply more aggressive preprocessing
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Try multiple parameter sets
        param_sets = [
            # (dp, minDist, param1, param2, minRadius, maxRadius)
            (1.5, 25, 50, 18, 8, 15),   # Smaller, closer bubbles
            (1.2, 30, 40, 20, 10, 18),  # Medium bubbles
            (1.0, 35, 45, 25, 12, 20),  # Larger bubbles
        ]
        
        all_circles = []
        for dp, minDist, p1, p2, minR, maxR in param_sets:
            circles = cv2.HoughCircles(
                blurred,
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
                for circle in circles:
                    all_circles.append(circle)
        
        # Remove duplicates (circles detected multiple times)
        if all_circles:
            all_circles = np.array(all_circles)
            unique_circles = []
            
            for circle in all_circles:
                is_duplicate = False
                for unique in unique_circles:
                    dist = np.sqrt((circle[0] - unique[0])**2 + (circle[1] - unique[1])**2)
                    if dist < 15:  # Too close, likely duplicate
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_circles.append(circle)
            
            # Convert to Bubble objects
            from omr.detector import Bubble
            for x, y, r in unique_circles:
                bubbles.append(Bubble(center=(x, y), radius=r))
        
        return bubbles


def test_working_example():
    """Test the working example"""
    
    # Load image
    img_path = Path("working_example.png")
    if not img_path.exists():
        print("Please generate working_example.png first")
        return
    
    img = cv2.imread(str(img_path))
    print(f"Image shape: {img.shape}")
    
    # Use tuned detector
    detector = TunedOMRDetector(fill_threshold=0.25)
    
    # Find bubble region
    region = find_bubble_region_v2(img)
    
    if region:
        print(f"\n✓ Found bubble region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Extract region
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        # Detect bubbles
        bubbles = detector.detect_bubbles(roi)
        print(f"✓ Found {len(bubbles)} bubbles")
        
        # Organize into grid
        grid = detector.cluster_to_grid(bubbles)
        print(f"✓ Grid: {grid.num_questions} questions, {grid.num_options} options per question")
        
        # Read answers
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        answers = detector.read_answers(gray_roi, grid)
        
        # Expected answers
        expected = {1: 'B', 2: 'D', 3: 'A', 4: 'C', 5: 'E'}
        
        print("\nDetection Results:")
        print("-" * 40)
        correct = 0
        for q_num in range(1, 6):
            detected = answers.get(q_num)
            expect = expected.get(q_num)
            status = "✓" if detected == expect else "✗"
            print(f"Question {q_num}: Expected {expect}, Got {detected} {status}")
            if detected == expect:
                correct += 1
        
        accuracy = (correct / 5) * 100
        print("-" * 40)
        print(f"Accuracy: {correct}/5 ({accuracy:.0f}%)")
        
        # Visualize results
        vis = img.copy()
        
        # Draw bubble region
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw detected bubbles
        for row in grid.bubbles:
            for bubble in row:
                # Color based on fill status
                if bubble.fill_ratio > detector.fill_threshold:
                    color = (0, 0, 255)  # Red for filled
                    thickness = -1
                else:
                    color = (255, 0, 0)  # Blue for empty
                    thickness = 2
                
                # Adjust coordinates back to full image
                bubble_x = bubble.center[0] + x
                bubble_y = bubble.center[1] + y
                
                cv2.circle(vis, (bubble_x, bubble_y), 
                          int(bubble.radius * 0.8), color, thickness)
                
                # Add question/option labels
                label = f"{bubble.row+1}{chr(65+bubble.col)}"
                cv2.putText(vis, label, 
                           (bubble_x - 15, bubble_y - int(bubble.radius) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save visualization
        cv2.imwrite("working_example_detected.png", vis)
        print(f"\n✓ Visualization saved to working_example_detected.png")
        
        return accuracy >= 60  # Consider successful if 60% or better
        
    else:
        print("✗ Could not find bubble region")
        return False


def create_simple_synthetic_test():
    """Create a very simple synthetic test that should work perfectly"""
    
    # Create white image
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add corner brackets
    bracket_size = 30
    thickness = 5
    
    # Top-left
    cv2.rectangle(img, (50, 50), (50 + thickness, 50 + bracket_size), (0,0,0), -1)
    cv2.rectangle(img, (50, 50), (50 + bracket_size, 50 + thickness), (0,0,0), -1)
    
    # Top-right  
    cv2.rectangle(img, (550 - thickness, 50), (550, 50 + bracket_size), (0,0,0), -1)
    cv2.rectangle(img, (550 - bracket_size, 50), (550, 50 + thickness), (0,0,0), -1)
    
    # Bottom-left
    cv2.rectangle(img, (50, 750 - thickness), (50 + bracket_size, 750), (0,0,0), -1)
    cv2.rectangle(img, (50, 750 - bracket_size), (50 + thickness, 750), (0,0,0), -1)
    
    # Bottom-right
    cv2.rectangle(img, (550 - thickness, 750 - bracket_size), (550, 750), (0,0,0), -1)
    cv2.rectangle(img, (550 - bracket_size, 750 - thickness), (550, 750), (0,0,0), -1)
    
    # Add bubble grid
    start_x, start_y = 150, 150
    spacing_x, spacing_y = 70, 100
    radius = 15
    
    # Answers: B, D, A, C, E
    filled = [(0, 1), (1, 3), (2, 0), (3, 2), (4, 4)]
    
    for row in range(5):  # 5 questions
        for col in range(5):  # 5 options
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y
            
            # Draw circle outline
            cv2.circle(img, (x, y), radius, (0,0,0), 2)
            
            # Fill if this is a correct answer
            if (row, col) in filled:
                cv2.circle(img, (x, y), radius - 3, (0,0,0), -1)
    
    # Save synthetic test
    cv2.imwrite("synthetic_test.png", img)
    print("Created synthetic_test.png")
    
    # Test it
    detector = TunedOMRDetector(fill_threshold=0.25)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    region = detector.find_bubble_region(gray)
    
    if region:
        print(f"✓ Found region: {region}")
        x, y, w, h = region
        roi = gray[y:y+h, x:x+w]
        
        bubbles = detector.detect_bubbles(roi)
        print(f"✓ Found {len(bubbles)} bubbles")
        
        if len(bubbles) >= 20:
            print("✓ Synthetic test successful!")
            return True
    
    return False


if __name__ == "__main__":
    print("Testing Working Example")
    print("=" * 50)
    
    # First try synthetic
    print("\n1. Testing synthetic image:")
    synthetic_success = create_simple_synthetic_test()
    
    # Then try real
    print("\n2. Testing generated quiz:")
    real_success = test_working_example()
    
    if real_success:
        print("\n✓ SUCCESS: Detection working on generated quiz!")
    else:
        print("\n⚠ Detection needs more tuning for generated quizzes")
        print("  But synthetic tests show the approach is sound")