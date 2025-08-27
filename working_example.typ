#import "quiz_template.typ": quiz

// Simple working example with clear bubbles
#quiz(
  title: "Working Example - Simple Test",
)[
  #set text(size: 12pt)
  #heading(level: 1)[Working Example Quiz]
  
  Instructions: Fill in bubbles completely with dark ink.
  
  #v(1em)
  
  // Question 1 - Answer: B
  + Option A
  + Option B [x]
  + Option C
  + Option D
  + Option E
  
  #v(0.5em)
  
  // Question 2 - Answer: D  
  + Option A
  + Option B
  + Option C
  + Option D [x]
  + Option E
  
  #v(0.5em)
  
  // Question 3 - Answer: A
  + Option A [x]
  + Option B
  + Option C
  + Option D
  + Option E
  
  #v(0.5em)
  
  // Question 4 - Answer: C
  + Option A
  + Option B
  + Option C [x]
  + Option D
  + Option E
  
  #v(0.5em)
  
  // Question 5 - Answer: E
  + Option A
  + Option B
  + Option C
  + Option D
  + Option E [x]
]