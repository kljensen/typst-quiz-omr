#import "quiz_template.typ": quiz

// Generate comprehensive test suite with known answer patterns
// This creates multiple test scenarios for validation

// Test 1: Sequential answers (A, B, C, D, E pattern)
#quiz(
  title: "Test 1: Sequential Pattern",
)[
  #set text(size: 11pt)
  = Test 1: Sequential Pattern
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
]
#pagebreak()

// Test 2: All same answer (stress test)
#quiz(
  title: "Test 2: All A's",
)[
  #set text(size: 11pt)
  = Test 2: All A's
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
]
#pagebreak()

// Test 3: Multiple answers per question
#quiz(
  title: "Test 3: Multiple Selections",
)[
  #set text(size: 11pt)
  = Test 3: Multiple Selections
  
  + Option A [x]
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B [x]
  + Option C
  + Option D [x]
  + Option E
  
  + Option A [x]
  + Option B [x]
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E [x]
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E [x]
]
#pagebreak()

// Test 4: No answers filled (blank test)
#quiz(
  title: "Test 4: Blank/No Answers",
)[
  #set text(size: 11pt)
  = Test 4: Blank/No Answers
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E
]
#pagebreak()

// Test 5: Variable number of options
#quiz(
  title: "Test 5: Variable Options",
)[
  #set text(size: 11pt)
  = Test 5: Variable Options
  
  + True [x]
  + False
  
  + Option A
  + Option B [x]
  + Option C
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
  + Option F
]
#pagebreak()

// Test 6: Edge pattern (first and last options)
#quiz(
  title: "Test 6: Edge Options",
)[
  #set text(size: 11pt)
  = Test 6: Edge Options
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
]
#pagebreak()

// Test 7: Random realistic pattern
#quiz(
  title: "Test 7: Realistic Pattern",
)[
  #set text(size: 11pt)
  = Test 7: Realistic Pattern
  
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
]
#pagebreak()

// Answer Keys Page
#set text(size: 14pt)
= Test Suite Answer Keys

#table(
  columns: (auto, auto),
  align: (left, left),
  [*Test*], [*Answers*],
  [Test 1: Sequential], [1:A, 2:B, 3:C, 4:D, 5:E, 6:A, 7:B, 8:C, 9:D, 10:E],
  [Test 2: All A's], [1:A, 2:A, 3:A, 4:A, 5:A, 6:A, 7:A, 8:A],
  [Test 3: Multiple], [1:A,C | 2:B,D | 3:A,B,C | 4:D,E | 5:C | 6:A,E],
  [Test 4: Blank], [All blank/no answers],
  [Test 5: Variable], [1:T, 2:B, 3:C, 4:D, 5:E],
  [Test 6: Edge], [1:A, 2:E, 3:A, 4:E, 5:A, 6:E],
  [Test 7: Realistic], [1:B, 2:D, 3:A, 4:C, 5:B, 6:E, 7:C, 8:A, 9:D, 10:B, 11:C, 12:D],
)