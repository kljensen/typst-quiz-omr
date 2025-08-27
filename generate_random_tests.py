#!/usr/bin/env python3
"""
Generate random test exams with varying answer patterns for comprehensive testing
"""

import random
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def generate_random_answer_pattern(num_questions: int, 
                                  num_options: int = 5,
                                  multi_select_prob: float = 0.1,
                                  blank_prob: float = 0.05) -> List[List[int]]:
    """
    Generate random answer pattern for a test.
    
    Args:
        num_questions: Number of questions
        num_options: Number of options per question (default 5 for A-E)
        multi_select_prob: Probability of multiple selections per question
        blank_prob: Probability of leaving a question blank
    
    Returns:
        List of answer indices for each question
    """
    answers = []
    
    for _ in range(num_questions):
        # Decide if this question is blank
        if random.random() < blank_prob:
            answers.append([])
            continue
        
        # Decide if this is multi-select
        if random.random() < multi_select_prob:
            # Choose 2-3 answers
            num_selections = random.randint(2, min(3, num_options))
            selected = random.sample(range(num_options), num_selections)
            answers.append(sorted(selected))
        else:
            # Single answer
            answers.append([random.randint(0, num_options - 1)])
    
    return answers


def answers_to_typst(answers: List[List[int]], num_options: int = 5) -> str:
    """Convert answer pattern to Typst quiz syntax"""
    typst_code = []
    
    for q_num, answer_indices in enumerate(answers, 1):
        typst_code.append(f"  // Question {q_num}")
        
        for opt in range(num_options):
            letter = chr(65 + opt)  # A, B, C, etc.
            mark = " [x]" if opt in answer_indices else ""
            typst_code.append(f"  + Option {letter}{mark}")
        
        if q_num < len(answers):
            typst_code.append("")  # Empty line between questions
    
    return "\n".join(typst_code)


def generate_random_test_file(test_id: str, 
                             num_questions: int = 10,
                             num_options: int = 5,
                             seed: int = None) -> Dict:
    """Generate a single random test file"""
    
    if seed is not None:
        random.seed(seed)
    
    # Generate random answers
    answers = generate_random_answer_pattern(
        num_questions, 
        num_options,
        multi_select_prob=0.15,  # 15% chance of multiple answers
        blank_prob=0.05  # 5% chance of blank
    )
    
    # Convert to letters for answer key
    answer_key = {}
    for q_num, indices in enumerate(answers, 1):
        if indices:
            letters = [chr(65 + idx) for idx in indices]
            answer_key[str(q_num)] = letters
        else:
            answer_key[str(q_num)] = []
    
    # Generate Typst file
    typst_content = f'''#import "quiz_template.typ": quiz

// Random Test {test_id}
// Generated: {datetime.now().isoformat()}
// Seed: {seed if seed else "random"}

#quiz(
  title: "Random Test {test_id}",
)[
  #set text(size: 11pt)
  #heading(level: 1)[Random Test {test_id}]
  
  Instructions: Fill in the bubble(s) for each question.
  
  #v(1em)
  
{answers_to_typst(answers, num_options)}
]'''
    
    # Save Typst file
    filename = f"random_test_{test_id}.typ"
    filepath = Path(filename)
    filepath.write_text(typst_content)
    
    # Generate PDF
    pdf_file = f"random_test_{test_id}.pdf"
    result = subprocess.run(
        ["typst", "compile", filename, pdf_file],
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"Error generating PDF for test {test_id}: {result.stderr}")
        return None
    
    # Convert to PNG
    png_file = f"tests/generated/random_test_{test_id}.png"
    Path("tests/generated").mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        ["magick", "-density", "300", f"{pdf_file}[0]", png_file],
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"Error converting to PNG for test {test_id}: {result.stderr}")
        return None
    
    return {
        "test_id": test_id,
        "seed": seed,
        "num_questions": num_questions,
        "num_options": num_options,
        "answers": answer_key,
        "typst_file": filename,
        "pdf_file": pdf_file,
        "png_file": png_file,
        "pattern_stats": analyze_pattern(answers)
    }


def analyze_pattern(answers: List[List[int]]) -> Dict:
    """Analyze answer pattern statistics"""
    stats = {
        "total_questions": len(answers),
        "blank_questions": sum(1 for a in answers if len(a) == 0),
        "multi_select_questions": sum(1 for a in answers if len(a) > 1),
        "single_select_questions": sum(1 for a in answers if len(a) == 1),
        "option_distribution": {}
    }
    
    # Count how often each option is used
    option_counts = {}
    for answer_indices in answers:
        for idx in answer_indices:
            letter = chr(65 + idx)
            option_counts[letter] = option_counts.get(letter, 0) + 1
    
    stats["option_distribution"] = option_counts
    
    # Check for patterns
    consecutive_same = 0
    max_consecutive = 0
    prev_answer = None
    
    for answer_indices in answers:
        if len(answer_indices) == 1:  # Only check single answers
            if prev_answer == answer_indices[0]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
            prev_answer = answer_indices[0] if answer_indices else None
    
    stats["max_consecutive_same"] = max_consecutive + 1  # +1 because we count transitions
    
    return stats


def generate_test_batch(num_tests: int = 10, 
                       questions_per_test: int = 10,
                       start_seed: int = None) -> Dict:
    """Generate a batch of random tests"""
    
    print(f"Generating {num_tests} random tests...")
    print(f"Each test has {questions_per_test} questions")
    print("-" * 50)
    
    if start_seed is None:
        start_seed = random.randint(1000, 9999)
    
    test_batch = {
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_tests": num_tests,
        "questions_per_test": questions_per_test,
        "start_seed": start_seed,
        "tests": []
    }
    
    for i in range(num_tests):
        test_id = f"{test_batch['batch_id']}_{i+1:03d}"
        seed = start_seed + i
        
        print(f"Generating test {i+1}/{num_tests} (ID: {test_id}, Seed: {seed})")
        
        test_data = generate_random_test_file(
            test_id=test_id,
            num_questions=questions_per_test,
            seed=seed
        )
        
        if test_data:
            test_batch["tests"].append(test_data)
            
            # Print pattern summary
            stats = test_data["pattern_stats"]
            print(f"  ✓ Generated: {stats['single_select_questions']} single, "
                  f"{stats['multi_select_questions']} multi, "
                  f"{stats['blank_questions']} blank")
            print(f"  ✓ Distribution: {stats['option_distribution']}")
            
            # Warn about potential issues
            if stats["max_consecutive_same"] >= 4:
                print(f"  ⚠ Warning: {stats['max_consecutive_same']} consecutive same answers")
    
    # Save batch metadata
    batch_file = f"test_batch_{test_batch['batch_id']}.json"
    with open(batch_file, 'w') as f:
        json.dump(test_batch, f, indent=2)
    
    print("-" * 50)
    print(f"✓ Generated {len(test_batch['tests'])} tests successfully")
    print(f"✓ Batch metadata saved to: {batch_file}")
    
    return test_batch


def generate_edge_case_tests() -> Dict:
    """Generate specific edge case tests"""
    
    edge_cases = []
    
    # Test 1: All A's
    edge_cases.append({
        "name": "all_a",
        "pattern": [[0] for _ in range(10)]
    })
    
    # Test 2: Sequential (A, B, C, D, E, A, B, C, D, E)
    edge_cases.append({
        "name": "sequential",
        "pattern": [[i % 5] for i in range(10)]
    })
    
    # Test 3: Reverse sequential (E, D, C, B, A, ...)
    edge_cases.append({
        "name": "reverse",
        "pattern": [[(4 - i) % 5] for i in range(10)]
    })
    
    # Test 4: All blanks
    edge_cases.append({
        "name": "all_blank",
        "pattern": [[] for _ in range(10)]
    })
    
    # Test 5: All multi-select
    edge_cases.append({
        "name": "all_multi",
        "pattern": [[i, (i+1) % 5] for i in range(10)]
    })
    
    # Test 6: Alternating pattern (A, E, A, E, ...)
    edge_cases.append({
        "name": "alternating",
        "pattern": [[0] if i % 2 == 0 else [4] for i in range(10)]
    })
    
    # Test 7: Middle option only (all C's)
    edge_cases.append({
        "name": "all_middle",
        "pattern": [[2] for _ in range(10)]
    })
    
    # Test 8: Random but with high clustering
    random.seed(42)  # Fixed seed for reproducibility
    clustered = []
    current = random.randint(0, 4)
    for _ in range(10):
        clustered.append([current])
        if random.random() > 0.7:  # 30% chance to change
            current = random.randint(0, 4)
    edge_cases.append({
        "name": "clustered",
        "pattern": clustered
    })
    
    print("Generating edge case tests...")
    print("-" * 50)
    
    batch = {
        "batch_id": "edge_cases",
        "tests": []
    }
    
    for i, case in enumerate(edge_cases):
        test_id = f"edge_{case['name']}"
        print(f"Generating edge case: {case['name']}")
        
        # Convert pattern to answer key
        answer_key = {}
        for q_num, indices in enumerate(case['pattern'], 1):
            if indices:
                letters = [chr(65 + idx) for idx in indices]
                answer_key[str(q_num)] = letters
            else:
                answer_key[str(q_num)] = []
        
        # Generate Typst file
        typst_content = f'''#import "quiz_template.typ": quiz

// Edge Case Test: {case['name']}
// Generated: {datetime.now().isoformat()}

#quiz(
  title: "Edge Case: {case['name'].replace('_', ' ').title()}",
)[
  #set text(size: 11pt)
  #heading(level: 1)[Edge Case: {case['name'].replace('_', ' ').title()}]
  
  This is an edge case test pattern.
  
  #v(1em)
  
{answers_to_typst(case['pattern'], 5)}
]'''
        
        # Save files
        filename = f"edge_test_{case['name']}.typ"
        Path(filename).write_text(typst_content)
        
        # Generate PDF and PNG
        pdf_file = f"edge_test_{case['name']}.pdf"
        subprocess.run(["typst", "compile", filename, pdf_file], capture_output=True)
        
        png_file = f"tests/generated/edge_test_{case['name']}.png"
        Path("tests/generated").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["magick", "-density", "300", f"{pdf_file}[0]", png_file],
            capture_output=True
        )
        
        batch["tests"].append({
            "test_id": test_id,
            "name": case['name'],
            "answers": answer_key,
            "pattern_stats": analyze_pattern(case['pattern'])
        })
        
        print(f"  ✓ Generated: {filename}")
    
    # Save edge case metadata
    with open("test_batch_edge_cases.json", 'w') as f:
        json.dump(batch, f, indent=2)
    
    print("-" * 50)
    print(f"✓ Generated {len(batch['tests'])} edge case tests")
    
    return batch


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random OMR test exams")
    parser.add_argument("--num-tests", type=int, default=10,
                       help="Number of random tests to generate (default: 10)")
    parser.add_argument("--questions", type=int, default=10,
                       help="Number of questions per test (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Starting seed for reproducibility")
    parser.add_argument("--edge-cases", action="store_true",
                       help="Generate edge case tests")
    
    args = parser.parse_args()
    
    if args.edge_cases:
        # Generate edge case tests
        edge_batch = generate_edge_case_tests()
    else:
        # Generate random tests
        batch = generate_test_batch(
            num_tests=args.num_tests,
            questions_per_test=args.questions,
            start_seed=args.seed
        )
        
        # Print summary statistics
        print("\nBatch Summary:")
        print("=" * 50)
        
        total_questions = sum(t["pattern_stats"]["total_questions"] for t in batch["tests"])
        total_blank = sum(t["pattern_stats"]["blank_questions"] for t in batch["tests"])
        total_multi = sum(t["pattern_stats"]["multi_select_questions"] for t in batch["tests"])
        
        print(f"Total questions: {total_questions}")
        print(f"Blank questions: {total_blank} ({total_blank/total_questions*100:.1f}%)")
        print(f"Multi-select: {total_multi} ({total_multi/total_questions*100:.1f}%)")
        
        # Aggregate option distribution
        all_options = {}
        for test in batch["tests"]:
            for opt, count in test["pattern_stats"]["option_distribution"].items():
                all_options[opt] = all_options.get(opt, 0) + count
        
        print(f"Option distribution: {all_options}")
        
        # Check for interesting patterns
        max_consecutive = max(t["pattern_stats"]["max_consecutive_same"] for t in batch["tests"])
        if max_consecutive >= 4:
            print(f"⚠ Maximum consecutive same answers: {max_consecutive}")