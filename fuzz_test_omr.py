#!/usr/bin/env python3
"""Fuzz testing for OMR detection system."""

import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, asdict
import argparse
import sys
import shutil

# Add parent directory to path to import omr_detector
sys.path.insert(0, str(Path(__file__).parent))
from omr_detector import detect_answers

@dataclass
class TestCase:
    """A single test case configuration."""
    test_id: int
    num_questions: int
    answers: Dict[int, List[str]]  # Question number to selected options
    options: str = "ABCD"
    
    def to_typst(self) -> str:
        """Generate Typst source for this test case."""
        lines = [
            '#import "quiz_template.typ": quiz',
            '',
            '#show: quiz.with(',
            f'  title: "Fuzz Test #{self.test_id}",',
            f'  options: "{self.options}",',
            '  show_filled: true',
            ')',
            ''
        ]
        
        for q_num in range(1, self.num_questions + 1):
            lines.append(f'{q_num}. Question {q_num}')
            selected = self.answers.get(q_num, [])
            
            for i, option in enumerate(self.options):
                mark = '[x]' if option in selected else '[ ]'
                lines.append(f'  - {mark} Option {option}')
            lines.append('')
        
        return '\n'.join(lines)

@dataclass
class TestResult:
    """Result of running a test case."""
    test_case: TestCase
    detected_answers: Dict[int, List[str]]
    passed: bool
    error_message: str = ""
    
    def matches(self) -> bool:
        """Check if detected answers match expected."""
        if set(self.test_case.answers.keys()) != set(self.detected_answers.keys()):
            return False
        
        for q_num, expected in self.test_case.answers.items():
            detected = self.detected_answers.get(q_num, [])
            if sorted(expected) != sorted(detected):
                return False
        
        return True

def generate_random_test_case(test_id: int, 
                             min_questions: int = 1,
                             max_questions: int = 20,
                             options: str = "ABCD",
                             multi_select_prob: float = 0.2) -> TestCase:
    """Generate a random test case."""
    num_questions = random.randint(min_questions, max_questions)
    answers = {}
    
    for q_num in range(1, num_questions + 1):
        # Randomly decide if this question has any answer
        if random.random() < 0.1:  # 10% chance of no answer
            continue
            
        # Decide if multi-select
        if random.random() < multi_select_prob:
            # Multi-select: choose 2-3 options
            num_selected = random.randint(2, min(3, len(options)))
            selected = random.sample(list(options), num_selected)
        else:
            # Single select
            selected = [random.choice(options)]
        
        answers[q_num] = selected
    
    return TestCase(
        test_id=test_id,
        num_questions=num_questions,
        answers=answers,
        options=options
    )

def run_test_case(test_case: TestCase, temp_dir: Path, verbose: bool = False) -> TestResult:
    """Run a single test case."""
    # Copy necessary files to temp directory
    quiz_template = Path("quiz_template.typ")
    markers_dir = Path("markers")
    
    if quiz_template.exists():
        shutil.copy(quiz_template, temp_dir)
    if markers_dir.exists():
        shutil.copytree(markers_dir, temp_dir / "markers", dirs_exist_ok=True)
    
    # Generate Typst file
    typst_file = temp_dir / f"test_{test_case.test_id}.typ"
    typst_file.write_text(test_case.to_typst())
    
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
        return TestResult(
            test_case=test_case,
            detected_answers={},
            passed=False,
            error_message=f"Typst compilation failed: {e.stderr}"
        )
    
    # Run OMR detection
    try:
        result = detect_answers(pdf_file, visualize=verbose)
        detected_answers = result.answers
        
        # Check if detection matches expected
        test_result = TestResult(
            test_case=test_case,
            detected_answers=detected_answers,
            passed=True
        )
        
        if not test_result.matches():
            test_result.passed = False
            test_result.error_message = "Detection mismatch"
        
        return test_result
        
    except Exception as e:
        return TestResult(
            test_case=test_case,
            detected_answers={},
            passed=False,
            error_message=f"OMR detection failed: {str(e)}"
        )

def run_fuzz_tests(num_tests: int = 100,
                  min_questions: int = 1,
                  max_questions: int = 20,
                  options: str = "ABCD",
                  multi_select_prob: float = 0.2,
                  verbose: bool = False,
                  keep_failures: bool = False) -> List[TestResult]:
    """Run multiple fuzz tests."""
    results = []
    temp_dir = Path(tempfile.mkdtemp(prefix="omr_fuzz_"))
    failures_dir = Path("fuzz_failures")
    
    if keep_failures:
        failures_dir.mkdir(exist_ok=True)
    
    print(f"Running {num_tests} fuzz tests...")
    print(f"Temporary files in: {temp_dir}")
    
    for i in range(num_tests):
        # Generate test case
        test_case = generate_random_test_case(
            test_id=i,
            min_questions=min_questions,
            max_questions=max_questions,
            options=options,
            multi_select_prob=multi_select_prob
        )
        
        # Run test
        result = run_test_case(test_case, temp_dir, verbose=verbose)
        results.append(result)
        
        # Progress indicator
        status = "✓" if result.passed else "✗"
        print(f"Test {i:3d}: {test_case.num_questions:2d} questions {status}", end="")
        
        if not result.passed:
            print(f" - {result.error_message}")
            
            # Save failure for analysis
            if keep_failures:
                failure_dir = failures_dir / f"test_{i}"
                failure_dir.mkdir(exist_ok=True)
                
                # Copy files
                typst_file = temp_dir / f"test_{i}.typ"
                pdf_file = temp_dir / f"test_{i}.pdf"
                
                if typst_file.exists():
                    shutil.copy(typst_file, failure_dir)
                if pdf_file.exists():
                    shutil.copy(pdf_file, failure_dir)
                
                # Save test case and result
                with open(failure_dir / "test_case.json", "w") as f:
                    json.dump({
                        "test_case": asdict(test_case),
                        "detected": result.detected_answers,
                        "error": result.error_message
                    }, f, indent=2)
        else:
            print()
    
    # Clean up temp directory unless keeping for debugging
    if not verbose:
        shutil.rmtree(temp_dir)
    
    return results

def print_summary(results: List[TestResult]):
    """Print summary of test results."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"  Total tests: {len(results)}")
    print(f"  Passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"  Failed: {failed} ({100*failed/len(results):.1f}%)")
    
    if failed > 0:
        print(f"\nFailure Analysis:")
        
        # Group failures by error type
        error_types = {}
        for r in results:
            if not r.passed:
                error_type = r.error_message.split(':')[0] if ':' in r.error_message else r.error_message
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        # Show examples of failures
        print(f"\nExample failures:")
        for r in results[:5]:  # Show first 5 failures
            if not r.passed:
                print(f"  Test {r.test_case.test_id}:")
                print(f"    Questions: {r.test_case.num_questions}")
                print(f"    Expected: {r.test_case.answers}")
                print(f"    Detected: {r.detected_answers}")
                print(f"    Error: {r.error_message}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fuzz testing for OMR detection")
    parser.add_argument("-n", "--num-tests", type=int, default=50,
                       help="Number of tests to run (default: 50)")
    parser.add_argument("--min-questions", type=int, default=1,
                       help="Minimum number of questions (default: 1)")
    parser.add_argument("--max-questions", type=int, default=15,
                       help="Maximum number of questions (default: 15)")
    parser.add_argument("--options", type=str, default="ABCD",
                       help="Answer options (default: ABCD)")
    parser.add_argument("--multi-select-prob", type=float, default=0.2,
                       help="Probability of multi-select questions (default: 0.2)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output and save visualizations")
    parser.add_argument("--keep-failures", action="store_true",
                       help="Keep failed test cases for analysis")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Run tests
    results = run_fuzz_tests(
        num_tests=args.num_tests,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        options=args.options,
        multi_select_prob=args.multi_select_prob,
        verbose=args.verbose,
        keep_failures=args.keep_failures
    )
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on results
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()