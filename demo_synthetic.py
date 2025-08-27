#!/usr/bin/env python3
"""
Demonstrate successful OMR detection on synthetic test
"""

import cv2
import numpy as np
from pathlib import Path
from omr.detector import DynamicOMRDetector

def demo_synthetic():
    """Demonstrate detection on synthetic image"""
    
    # Load synthetic test
    img_path = Path("synthetic_test.png")
    img = cv2.imread(str(img_path))
    
    # Create detector
    detector = DynamicOMRDetector(fill_threshold=0.30)
    
    # Process the image
    try:
        answers = detector.process(img_path)
        
        print("Successfully Detected Answers:")
        print("=" * 40)
        
        # Expected answers based on what we filled
        expected = {1: 'B', 2: 'D', 3: 'A', 4: 'C', 5: 'E'}
        
        for q_num in range(1, 6):
            detected = answers.get(q_num)
            expect = expected.get(q_num)
            status = "✓" if detected == expect else "✗"
            print(f"Question {q_num}: {detected} (Expected: {expect}) {status}")
        
        # Calculate accuracy
        correct = sum(1 for q in range(1, 6) if answers.get(q) == expected.get(q))
        accuracy = (correct / 5) * 100
        
        print("=" * 40)
        print(f"Accuracy: {correct}/5 ({accuracy:.0f}%)")
        
        if accuracy == 100:
            print("\n✅ PERFECT DETECTION!")
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        return 0


if __name__ == "__main__":
    print("Synthetic Test Demonstration")
    print("-" * 40)
    print("This synthetic image has:")
    print("• Clear corner brackets (L-shaped)")
    print("• 5 questions with 5 options each")
    print("• Filled answers: B, D, A, C, E")
    print("• High contrast bubbles")
    print("-" * 40)
    print()
    
    accuracy = demo_synthetic()
    
    if accuracy >= 80:
        print("\n✅ The OMR system works successfully on this test!")