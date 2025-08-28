#import "quiz_template.typ": quiz

// Experiment 1: Vertical bar marker
#page[
  = Option 1: Vertical Bar
  
  #grid(
    columns: (30pt, auto),
    rows: auto,
    gutter: 10pt,
    
    // Black bar on left
    rect(width: 100%, height: 200pt, fill: black),
    
    // Bubble grid
    [
      1. ○ ○ ○ ○ ○\
      2. ○ ○ ○ ○ ○\
      3. ○ ○ ○ ○ ○\
      4. ○ ○ ○ ○ ○\
      5. ○ ○ ○ ○ ○
    ]
  )
]

#pagebreak()

// Experiment 2: Registration squares
#page[
  = Option 2: Registration Squares
  
  #box(width: 300pt)[
    // Corner squares
    #place(top + left, rect(width: 10pt, height: 10pt, fill: black))
    #place(top + right, rect(width: 10pt, height: 10pt, fill: black))
    #place(bottom + left, rect(width: 10pt, height: 10pt, fill: black))
    #place(bottom + right, rect(width: 10pt, height: 10pt, fill: black))
    
    #pad(top: 20pt, bottom: 20pt, left: 20pt, right: 20pt)[
      1. ○ ○ ○ ○ ○\
      2. ○ ○ ○ ○ ○\
      3. ○ ○ ○ ○ ○\
      4. ○ ○ ○ ○ ○\
      5. ○ ○ ○ ○ ○
    ]
  ]
]

#pagebreak()

// Experiment 3: Row markers (triangles pointing to each row)
#page[
  = Option 3: Row Markers
  
  #grid(
    columns: (20pt, auto),
    rows: auto,
    gutter: 5pt,
    align: (center, left),
    
    // Row 1 marker
    [▶], [1. ○ ○ ○ ○ ○],
    [▶], [2. ○ ○ ○ ○ ○],
    [▶], [3. ○ ○ ○ ○ ○],
    [▶], [4. ○ ○ ○ ○ ○],
    [▶], [5. ○ ○ ○ ○ ○],
  )
]

#pagebreak()

// Experiment 4: Horizontal delimiter lines with pattern
#page[
  = Option 4: Delimiter Lines
  
  #block[
    // Top delimiter with pattern
    #rect(width: 300pt, height: 3pt, fill: black)
    #v(2pt)
    #rect(width: 300pt, height: 1pt, fill: black)
    
    #pad(top: 10pt, bottom: 10pt)[
      1. ○ ○ ○ ○ ○\
      2. ○ ○ ○ ○ ○\
      3. ○ ○ ○ ○ ○\
      4. ○ ○ ○ ○ ○\
      5. ○ ○ ○ ○ ○
    ]
    
    // Bottom delimiter with pattern  
    #rect(width: 300pt, height: 1pt, fill: black)
    #v(2pt)
    #rect(width: 300pt, height: 3pt, fill: black)
  ]
]