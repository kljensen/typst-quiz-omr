#!/usr/bin/env python3
"""
Test the final detector on the random batch
"""

import json
from pathlib import Path
from omr.typst_final_detector import TypstFinalDetector
import sys


def test_batch():
    """Test final detector on batch"""
    
    # Load batch metadata
    batch_file = Path("test_batch_20250826_202248.json")
    if not batch_file.exists():
        print("Batch file not found")
        return 0
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    detector = TypstFinalDetector(fill_threshold=0.35, verbose=False)
    
    print("Testing Final Typst Detector")
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
            errors = []
            for q_str, exp_list in expected.items():
                q_num = int(q_str)
                det = detected.get(q_num)
                exp = exp_list[0] if exp_list else None
                
                if det == exp:
                    correct += 1
                elif det != exp:
                    errors.append(f"Q{q_num}: {exp}→{det}")
            
            accuracy = (correct / len(expected)) * 100
            total_correct += correct
            total_questions += len(expected)
            
            print(f"{accuracy:.0f}% ({correct}/{len(expected)})", end="")
            if errors and len(errors) <= 3:
                print(f" Errors: {', '.join(errors)}", end="")
            print()
            
        except Exception as e:
            print(f"Error: {e}")
    
    if total_questions > 0:
        overall = (total_correct / total_questions) * 100
        print("\n" + "=" * 60)
        print(f"OVERALL ACCURACY: {overall:.1f}% ({total_correct}/{total_questions})")
        
        if overall >= 70:
            print("✅ GOOD: Detector working well!")
        elif overall >= 50:
            print("⚠️ MODERATE: Needs tuning")
        else:
            print("❌ LOW: Significant improvements needed")
        
        return overall
    return 0


if __name__ == "__main__":
    accuracy = test_batch()
    
    # Compare with previous attempts
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"- Baseline detector: 12%")
    print(f"- Improved detector v2: 18%") 
    print(f"- Final detector: {accuracy:.1f}%")
    
    if accuracy > 18:
        improvement = accuracy - 18
        print(f"\n🎉 Improvement over v2: +{improvement:.1f}%")