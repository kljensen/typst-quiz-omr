#!/usr/bin/env python3
"""
Test the Typst-optimized detector
"""

import json
from pathlib import Path
from omr.typst_optimized_detector import TypstOptimizedDetector
import sys


def test_on_batch():
    """Test optimized detector on random batch"""
    
    # Load batch metadata
    batch_file = Path("test_batch_20250826_202248.json")
    if not batch_file.exists():
        print("Batch file not found")
        return 0
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    detector = TypstOptimizedDetector(fill_threshold=0.30, verbose=False)
    
    print("Testing Typst-Optimized Detector")
    print("=" * 60)
    
    total_correct = 0
    total_questions = 0
    
    for test_data in batch_data['tests']:
        test_id = test_data['test_id']
        png_file = Path(f"tests/generated/random_test_{test_id}.png")
        
        if not png_file.exists():
            continue
        
        print(f"\nTest {test_id}:", end=" ")
        
        try:
            detected = detector.process(png_file)
            expected = test_data['answers']
            
            correct = 0
            for q_str, exp_list in expected.items():
                q_num = int(q_str)
                det = detected.get(q_num)
                exp = exp_list[0] if exp_list else None
                
                if det == exp:
                    correct += 1
            
            accuracy = (correct / len(expected)) * 100
            total_correct += correct
            total_questions += len(expected)
            
            print(f"{accuracy:.0f}% ({correct}/{len(expected)})")
            
        except Exception as e:
            print(f"Error: {e}")
    
    if total_questions > 0:
        overall = (total_correct / total_questions) * 100
        print("\n" + "=" * 60)
        print(f"OVERALL: {overall:.1f}% ({total_correct}/{total_questions})")
        return overall
    return 0


def test_single():
    """Test on single image with verbose output"""
    detector = TypstOptimizedDetector(fill_threshold=0.30, verbose=True)
    
    # Test on first random test
    test_path = Path("tests/generated/random_test_20250826_202248_001.png")
    
    if test_path.exists():
        print("Testing single image...")
        answers = detector.process(test_path)
        print(f"\nDetected: {answers}")
        
        # Compare with expected
        with open("test_batch_20250826_202248.json", 'r') as f:
            batch = json.load(f)
        expected = batch['tests'][0]['answers']
        
        correct = sum(1 for q, exp in expected.items() 
                     if exp and answers.get(int(q)) == exp[0])
        print(f"Accuracy: {correct}/{len(expected)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        test_single()
    else:
        accuracy = test_on_batch()
        print(f"\nTypst-Optimized Detector: {accuracy:.1f}%")