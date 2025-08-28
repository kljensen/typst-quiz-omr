#!/usr/bin/env python3
"""Comprehensive test covering 1-16 questions with random answers."""

import random
import subprocess
from pathlib import Path
import sys
import json

# Add parent directory to path to import omr_detector
sys.path.insert(0, str(Path(__file__).parent))
from omr_detector import detect_answers

def generate_quiz(num_questions: int, seed: int = None) -> tuple:
    """Generate a quiz with random answers."""
    if seed:
        random.seed(seed)
    
    # Generate random answers
    expected_answers = {}
    for q in range(1, num_questions + 1):
        # 80% single answer, 20% multiple answers
        if random.random() < 0.8:
            expected_answers[q] = [random.choice(['A', 'B', 'C', 'D'])]
        else:
            # Multiple answers (2-3)
            num_answers = random.randint(2, 3)
            expected_answers[q] = sorted(random.sample(['A', 'B', 'C', 'D'], num_answers))
    
    # Generate Typst content
    lines = [
        '#import "quiz_template.typ": quiz',
        '',
        '#show: quiz.with(',
        f'  title: "Comprehensive Test: {num_questions} Questions",',
        '  show_filled: true',
        ')',
        ''
    ]
    
    for q in range(1, num_questions + 1):
        lines.append(f'{q}. Question {q}')
        selected = expected_answers.get(q, [])
        for option in ['A', 'B', 'C', 'D']:
            mark = '[x]' if option in selected else '[ ]'
            lines.append(f'  - {mark} Option {option}')
        lines.append('')
    
    return '\n'.join(lines), expected_answers

def run_test(num_questions: int, verbose: bool = False) -> dict:
    """Run a test with the specified number of questions."""
    print(f"\nTesting {num_questions} questions...")
    
    # Generate quiz
    typst_content, expected_answers = generate_quiz(num_questions, seed=num_questions * 100)
    
    # Write Typst file
    typst_file = Path(f'temp_test_{num_questions}.typ')
    typst_file.write_text(typst_content)
    
    # Compile to PDF
    pdf_file = typst_file.with_suffix('.pdf')
    try:
        result = subprocess.run(
            ['typst', 'compile', str(typst_file), str(pdf_file)],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Typst compilation failed: {e.stderr}")
        return {'passed': False, 'error': 'Compilation failed'}
    
    # Run OMR detection
    try:
        result = detect_answers(pdf_file)
        detected_answers = result.answers
        
        # Compare answers
        passed = True
        errors = []
        
        # Check all expected answers are detected
        for q, expected in expected_answers.items():
            detected = detected_answers.get(q, [])
            if sorted(expected) != sorted(detected):
                passed = False
                errors.append(f"Q{q}: expected {expected}, got {detected}")
        
        # Check no extra answers detected
        for q in detected_answers:
            if q not in expected_answers:
                passed = False
                errors.append(f"Q{q}: unexpected detection")
        
        if passed:
            print(f"  ✅ Passed - {len(detected_answers)} questions detected correctly")
            if num_questions <= 8:
                print(f"     Layout: Single column")
            else:
                print(f"     Layout: Two columns")
        else:
            print(f"  ❌ Failed - Detection errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"     {error}")
        
        # Clean up temp files
        if not verbose:
            typst_file.unlink()
            pdf_file.unlink()
        
        return {
            'passed': passed,
            'num_questions': num_questions,
            'expected': expected_answers,
            'detected': detected_answers,
            'errors': errors
        }
        
    except Exception as e:
        print(f"  ❌ OMR detection failed: {str(e)}")
        return {'passed': False, 'error': str(e)}

def main():
    """Run comprehensive tests for 1-16 questions."""
    print("=" * 60)
    print("Comprehensive OMR Test: 1-16 Questions")
    print("=" * 60)
    
    results = []
    passed_count = 0
    
    # Test each number of questions from 1 to 16
    for n in range(1, 17):
        result = run_test(n)
        results.append(result)
        if result.get('passed', False):
            passed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Total tests: 16")
    print(f"  Passed: {passed_count}/16 ({100*passed_count/16:.1f}%)")
    
    # Show which ones failed
    failed = [r for r in results if not r.get('passed', False)]
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            n = r.get('num_questions', '?')
            error = r.get('error', 'Detection mismatch')
            print(f"  {n} questions: {error}")
    
    # Analysis by layout type
    single_column = [r for r in results if r.get('num_questions', 0) <= 8]
    double_column = [r for r in results if r.get('num_questions', 0) > 8]
    
    single_passed = sum(1 for r in single_column if r.get('passed', False))
    double_passed = sum(1 for r in double_column if r.get('passed', False))
    
    print(f"\nBy layout type:")
    print(f"  Single column (1-8): {single_passed}/{len(single_column)} passed")
    print(f"  Double column (9-16): {double_passed}/{len(double_column)} passed")
    
    # Save results for analysis
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to comprehensive_test_results.json")
    
    return passed_count == 16

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)