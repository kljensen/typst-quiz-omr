#import "quiz_template.typ": quiz

// Generate a test quiz with specific answers filled
#let generate_test_quiz(
  num_questions: 10,
  filled_answers: (:),  // Dictionary like ("1": "A", "2": "B")
  title: "Test Quiz",
) = {
  show: quiz.with(
    title: title,
    date: "Test Data",
  )
  
  // Generate each question
  for i in range(1, num_questions + 1) {
    let q_num = str(i)
    let correct = filled_answers.at(q_num, default: none)
    
    // Build the question with proper markers
    [+ Test Question #i
      - #if correct == "A" { "[x]" } else { "[ ]" } Option A
      - #if correct == "B" { "[x]" } else { "[ ]" } Option B  
      - #if correct == "C" { "[x]" } else { "[ ]" } Option C
      - #if correct == "D" { "[x]" } else { "[ ]" } Option D
      - #if correct == "E" { "[x]" } else { "[ ]" } Option E
    ]
  }
}

// Generate multiple test cases
#pagebreak()
#generate_test_quiz(
  num_questions: 5,
  filled_answers: (
    "1": "A",
    "2": "B",
    "3": "C", 
    "4": "D",
    "5": "E"
  ),
  title: "Test 1: 5 Questions Sequential"
)

#pagebreak()
#generate_test_quiz(
  num_questions: 10,
  filled_answers: (
    "1": "C",
    "2": "C",
    "3": "A",
    "4": "B",
    "5": "D",
    "6": "E",
    "7": "A",
    "8": "B",
    "9": "C",
    "10": "D"
  ),
  title: "Test 2: 10 Questions Mixed"
)

#pagebreak()
#generate_test_quiz(
  num_questions: 15,
  filled_answers: (
    "1": "A",
    "3": "B",
    "5": "C",
    "7": "D",
    "9": "E",
    "11": "A",
    "13": "B",
    "15": "C"
  ),
  title: "Test 3: 15 Questions Partial"
)

#pagebreak()
#generate_test_quiz(
  num_questions: 20,
  filled_answers: (:),  // All blank
  title: "Test 4: 20 Questions Blank"
)

#pagebreak()
#generate_test_quiz(
  num_questions: 7,
  filled_answers: (
    "1": "A",
    "2": "A", 
    "3": "A",
    "4": "A",
    "5": "A",
    "6": "A",
    "7": "A"
  ),
  title: "Test 5: 7 Questions All A"
)