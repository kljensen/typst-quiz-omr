#!/usr/bin/env python3
"""
Test the improved detector on random batch
"""

import json
from pathlib import Path
from omr.improved_detector_v2 import ImprovedOMRDetector


def test_batch_with_improved_detector():
    """Test the improved detector on our random batch"""
    
    # Load batch metadata
    batch_file = Path("test_batch_20250826_202248.json")
    if not batch_file.exists():
        print("Batch file not found. Generate it first with generate_random_tests.py")
        return
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    # Create improved detector
    detector = ImprovedOMRDetector(fill_threshold=0.25, verbose=False)
    
    print("Testing Improved Detector on Random Batch")
    print("=" * 60)
    
    total_correct = 0
    total_questions = 0
    
    for test_data in batch_data['tests']:
        test_id = test_data['test_id']
        png_file = Path(f"tests/generated/random_test_{test_id}.png")
        
        if not png_file.exists():
            print(f"❌ {test_id}: Image not found")
            continue
        
        print(f"\nTest {test_id}:")
        print("-" * 40)
        
        try:
            # Process with improved detector
            detected = detector.process(png_file)
            
            # Compare with expected
            expected_answers = test_data['answers']
            correct = 0
            
            for q_num_str, expected_letters in expected_answers.items():
                q_num = int(q_num_str)
                detected_answer = detected.get(q_num)
                
                # Check correctness
                is_correct = False
                if len(expected_letters) == 0:
                    # Blank expected
                    is_correct = (detected_answer is None)
                elif len(expected_letters) == 1:
                    # Single answer
                    is_correct = (detected_answer == expected_letters[0])
                else:
                    # Multi-answer - check if detected is one of them
                    is_correct = (detected_answer in expected_letters)
                
                if is_correct:
                    correct += 1
                else:
                    # Show errors for debugging
                    exp = expected_letters[0] if expected_letters else "blank"
                    det = detected_answer if detected_answer else "blank"
                    print(f"  Q{q_num}: Expected {exp}, Got {det}")
            
            accuracy = (correct / len(expected_answers)) * 100
            total_correct += correct
            total_questions += len(expected_answers)
            
            status = "✅" if accuracy >= 70 else "⚠️"
            print(f"{status} Accuracy: {accuracy:.1f}% ({correct}/{len(expected_answers)})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    
    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"Total accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")
        
        if overall_accuracy >= 80:
            print("✅ EXCELLENT: Improved detector works well!")
        elif overall_accuracy >= 60:
            print("⚠️ GOOD: Significant improvement over baseline")
        else:
            print("❌ Needs more tuning")
    
    return overall_accuracy if total_questions > 0 else 0


def test_on_single_image():
    """Test on a single specific image for debugging"""
    
    detector = ImprovedOMRDetector(fill_threshold=0.25, verbose=True)
    
    # Test on first random test
    test_path = Path("tests/generated/random_test_20250826_202248_001.png")
    
    if test_path.exists():
        print("Testing on single image with verbose output...")
        print("-" * 60)
        
        answers = detector.process(test_path)
        print(f"\nDetected answers: {answers}")
        
        # Load expected
        with open("test_batch_20250826_202248.json", 'r') as f:
            batch = json.load(f)
        
        expected = batch['tests'][0]['answers']
        print(f"\nExpected answers: {expected}")
        
        correct = 0
        for q_str, exp_list in expected.items():
            q_num = int(q_str)
            if exp_list and answers.get(q_num) == exp_list[0]:
                correct += 1
        
        print(f"\nAccuracy: {correct}/{len(expected)} ({correct/len(expected)*100:.0f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        test_on_single_image()
    else:
        overall = test_batch_with_improved_detector()
        
        # Compare with baseline
        print("\nComparison:")
        print("- Baseline detector: 12% accuracy")
        print(f"- Improved detector: {overall:.1f}% accuracy")
        if overall > 12:
            improvement = overall - 12
            print(f"- Improvement: +{improvement:.1f}% 🎉")