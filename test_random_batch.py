#!/usr/bin/env python3
"""
Test OMR detection on randomly generated test batches
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from omr.detector import DynamicOMRDetector
import time


class BatchTestRunner:
    """Run OMR detection on batches of tests and analyze results"""
    
    def __init__(self, detector: DynamicOMRDetector = None):
        self.detector = detector or DynamicOMRDetector(fill_threshold=0.30)
        self.results = []
    
    def test_single_image(self, image_path: Path, expected_answers: Dict[str, List[str]]) -> Dict:
        """Test a single image and return results"""
        
        start_time = time.time()
        
        try:
            # Process image
            detected = self.detector.process(image_path)
            
            # Compare with expected
            correct = 0
            total = len(expected_answers)
            details = []
            
            for q_num_str, expected_letters in expected_answers.items():
                q_num = int(q_num_str)
                detected_answer = detected.get(q_num)
                
                # Convert detected to letter if exists
                detected_letter = detected_answer if detected_answer else "None"
                
                # Check if correct
                is_correct = False
                if len(expected_letters) == 0:
                    # Blank expected
                    is_correct = (detected_answer is None)
                elif len(expected_letters) == 1:
                    # Single answer
                    is_correct = (detected_letter == expected_letters[0])
                else:
                    # Multi-answer - for now just check if detected is one of them
                    is_correct = (detected_letter in expected_letters)
                
                if is_correct:
                    correct += 1
                
                details.append({
                    "question": q_num,
                    "expected": expected_letters,
                    "detected": detected_letter,
                    "correct": is_correct
                })
            
            accuracy = (correct / total * 100) if total > 0 else 0
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "processing_time": processing_time,
                "details": details
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def test_batch(self, batch_file: Path) -> Dict:
        """Test all images in a batch"""
        
        # Load batch metadata
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        print(f"Testing batch: {batch_data.get('batch_id', 'unknown')}")
        print(f"Number of tests: {len(batch_data['tests'])}")
        print("-" * 60)
        
        batch_results = {
            "batch_id": batch_data.get('batch_id'),
            "num_tests": len(batch_data['tests']),
            "test_results": [],
            "summary": {}
        }
        
        for test_data in batch_data['tests']:
            test_id = test_data['test_id']
            png_file = Path(test_data.get('png_file', f"tests/generated/random_test_{test_id}.png"))
            
            if not png_file.exists():
                print(f"❌ Test {test_id}: Image not found")
                continue
            
            print(f"Testing {test_id}...", end=" ")
            
            # Run test
            result = self.test_single_image(png_file, test_data['answers'])
            result['test_id'] = test_id
            result['pattern_stats'] = test_data.get('pattern_stats', {})
            
            batch_results['test_results'].append(result)
            
            if result['success']:
                status = "✅" if result['accuracy'] >= 70 else "⚠️"
                print(f"{status} Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
                
                # Show errors for low accuracy
                if result['accuracy'] < 70:
                    errors = [d for d in result['details'] if not d['correct']][:3]
                    for err in errors:
                        print(f"    Q{err['question']}: Expected {err['expected']}, Got {err['detected']}")
            else:
                print(f"❌ Failed: {result['error']}")
        
        # Calculate summary statistics
        successful_tests = [r for r in batch_results['test_results'] if r['success']]
        
        if successful_tests:
            accuracies = [r['accuracy'] for r in successful_tests]
            batch_results['summary'] = {
                "num_successful": len(successful_tests),
                "num_failed": len(batch_results['test_results']) - len(successful_tests),
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "median_accuracy": np.median(accuracies),
                "tests_above_70": sum(1 for a in accuracies if a >= 70),
                "tests_above_80": sum(1 for a in accuracies if a >= 80),
                "tests_above_90": sum(1 for a in accuracies if a >= 90),
                "mean_processing_time": np.mean([r['processing_time'] for r in successful_tests])
            }
        
        return batch_results
    
    def analyze_error_patterns(self, batch_results: Dict) -> Dict:
        """Analyze common error patterns across the batch"""
        
        error_analysis = {
            "by_question_position": {},
            "by_answer_option": {},
            "by_pattern_type": {},
            "common_confusions": {}
        }
        
        # Collect all errors
        all_errors = []
        for test_result in batch_results['test_results']:
            if test_result['success']:
                for detail in test_result['details']:
                    if not detail['correct']:
                        all_errors.append({
                            "test_id": test_result['test_id'],
                            "question": detail['question'],
                            "expected": detail['expected'],
                            "detected": detail['detected'],
                            "pattern_stats": test_result.get('pattern_stats', {})
                        })
        
        # Analyze by question position
        for error in all_errors:
            q_pos = error['question']
            if q_pos not in error_analysis['by_question_position']:
                error_analysis['by_question_position'][q_pos] = 0
            error_analysis['by_question_position'][q_pos] += 1
        
        # Analyze by answer option
        for error in all_errors:
            for expected in error['expected']:
                key = f"Expected_{expected}"
                if key not in error_analysis['by_answer_option']:
                    error_analysis['by_answer_option'][key] = 0
                error_analysis['by_answer_option'][key] += 1
        
        # Find common confusions (what was detected instead)
        confusion_pairs = {}
        for error in all_errors:
            if len(error['expected']) == 1:  # Single answer only
                pair = f"{error['expected'][0]}->{error['detected']}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        error_analysis['common_confusions'] = dict(
            sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return error_analysis


def print_batch_summary(batch_results: Dict):
    """Print a nice summary of batch test results"""
    
    summary = batch_results['summary']
    
    print("\n" + "=" * 60)
    print("BATCH TEST SUMMARY")
    print("=" * 60)
    
    print(f"Batch ID: {batch_results['batch_id']}")
    print(f"Tests run: {batch_results['num_tests']}")
    print(f"Successful: {summary['num_successful']}")
    print(f"Failed: {summary['num_failed']}")
    
    print("\nAccuracy Statistics:")
    print(f"  Mean:   {summary['mean_accuracy']:.1f}%")
    print(f"  Median: {summary['median_accuracy']:.1f}%")
    print(f"  StdDev: {summary['std_accuracy']:.1f}%")
    print(f"  Range:  {summary['min_accuracy']:.1f}% - {summary['max_accuracy']:.1f}%")
    
    print("\nPerformance Distribution:")
    print(f"  >= 90% accuracy: {summary['tests_above_90']} tests")
    print(f"  >= 80% accuracy: {summary['tests_above_80']} tests")
    print(f"  >= 70% accuracy: {summary['tests_above_70']} tests")
    
    print(f"\nMean processing time: {summary['mean_processing_time']:.2f} seconds")
    
    # Success rate interpretation
    success_rate = summary['tests_above_70'] / summary['num_successful'] * 100
    print("\n" + "=" * 60)
    if success_rate >= 90:
        print("✅ EXCELLENT: System performs very well on random tests!")
    elif success_rate >= 70:
        print("⚠️ GOOD: System works but needs some tuning")
    else:
        print("❌ NEEDS WORK: System requires significant improvements")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OMR on random test batches")
    parser.add_argument("batch_file", nargs="?", 
                       help="Batch JSON file to test")
    parser.add_argument("--generate", type=int, default=0,
                       help="Generate N new random tests first")
    parser.add_argument("--edge-cases", action="store_true",
                       help="Test edge cases")
    
    args = parser.parse_args()
    
    # Generate tests if requested
    if args.generate > 0:
        print(f"Generating {args.generate} random tests...")
        from generate_random_tests import generate_test_batch
        batch = generate_test_batch(num_tests=args.generate)
        batch_file = Path(f"test_batch_{batch['batch_id']}.json")
    elif args.edge_cases:
        print("Testing edge cases...")
        from generate_random_tests import generate_edge_case_tests
        batch = generate_edge_case_tests()
        batch_file = Path("test_batch_edge_cases.json")
    elif args.batch_file:
        batch_file = Path(args.batch_file)
    else:
        # Find most recent batch file
        batch_files = list(Path(".").glob("test_batch_*.json"))
        if not batch_files:
            print("No batch files found. Generate some first with --generate N")
            exit(1)
        batch_file = max(batch_files, key=lambda f: f.stat().st_mtime)
        print(f"Using most recent batch: {batch_file}")
    
    if not batch_file.exists():
        print(f"Batch file not found: {batch_file}")
        exit(1)
    
    # Run tests
    runner = BatchTestRunner()
    results = runner.test_batch(batch_file)
    
    # Analyze errors
    if results['summary']:
        print_batch_summary(results)
        
        # Show error analysis
        errors = runner.analyze_error_patterns(results)
        
        print("\nError Analysis:")
        print("-" * 60)
        
        # Questions with most errors
        if errors['by_question_position']:
            print("Questions with most errors:")
            for q, count in sorted(errors['by_question_position'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
                print(f"  Question {q}: {count} errors")
        
        # Common confusions
        if errors['common_confusions']:
            print("\nMost common confusions:")
            for confusion, count in list(errors['common_confusions'].items())[:5]:
                print(f"  {confusion}: {count} times")
    
    # Save detailed results
    results_file = f"test_results_{batch_file.stem}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")