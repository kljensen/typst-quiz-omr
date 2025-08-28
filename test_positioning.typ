#set page(width: 8.5in, height: 11in, margin: 0.75in)

// Method 1: Using place in the main document flow
= Test Positioning Methods

#place(top + left, dx: 0pt, dy: 0pt)[
  #rect(width: 20pt, height: 20pt, fill: red)
]

#place(top + right, dx: -20pt, dy: 0pt)[
  #rect(width: 20pt, height: 20pt, fill: blue)
]

#place(bottom + left, dx: 0pt, dy: -20pt)[
  #rect(width: 20pt, height: 20pt, fill: green)
]

#place(bottom + right, dx: -20pt, dy: -20pt)[
  #rect(width: 20pt, height: 20pt, fill: yellow)
]

Main content goes here. This is the regular flow of the document.

#pagebreak()

// Method 2: Using page.background
#set page(background: {
  // Top-left marker
  place(top + left, dx: 10pt, dy: 10pt)[
    #rect(width: 30pt, height: 30pt, fill: red.lighten(50%))
  ]
  
  // Top-right marker
  place(top + right, dx: -40pt, dy: 10pt)[
    #rect(width: 30pt, height: 30pt, fill: blue.lighten(50%))
  ]
  
  // Bottom-left marker
  place(bottom + left, dx: 10pt, dy: -40pt)[
    #rect(width: 30pt, height: 30pt, fill: green.lighten(50%))
  ]
  
  // Bottom-right marker
  place(bottom + right, dx: -40pt, dy: -40pt)[
    #rect(width: 30pt, height: 30pt, fill: yellow.lighten(50%))
  ]
})

= Page with Background Markers

This page has markers placed in the background layer, which means they appear behind all content.

#pagebreak()

// Method 3: Using page.foreground
#set page(foreground: {
  // Top-left marker
  place(top + left, dx: 10pt, dy: 10pt)[
    #rect(width: 30pt, height: 30pt, fill: red.lighten(70%), stroke: black)
  ]
  
  // Top-right marker
  place(top + right, dx: -40pt, dy: 10pt)[
    #rect(width: 30pt, height: 30pt, fill: blue.lighten(70%), stroke: black)
  ]
  
  // Bottom-left marker
  place(bottom + left, dx: 10pt, dy: -40pt)[
    #rect(width: 30pt, height: 30pt, fill: green.lighten(70%), stroke: black)
  ]
  
  // Bottom-right marker
  place(bottom + right, dx: -40pt, dy: -40pt)[
    #rect(width: 30pt, height: 30pt, fill: yellow.lighten(70%), stroke: black)
  ]
})

= Page with Foreground Markers

This page has markers placed in the foreground layer, which means they appear above all content.

#pagebreak()

// Method 4: Testing with actual ArUco markers
#set page(foreground: {
  // NetID region markers
  place(top + right, dx: -200pt, dy: 40pt)[
    #image("markers/aruco_0.png", width: 15pt)
  ]
  
  place(top + right, dx: -50pt, dy: 40pt)[
    #image("markers/aruco_1.png", width: 15pt)
  ]
  
  // Bubble region markers (positioned lower on page)
  place(top + left, dx: 40pt, dy: 120pt)[
    #image("markers/aruco_2.png", width: 15pt)
  ]
  
  place(top + right, dx: -55pt, dy: 120pt)[
    #image("markers/aruco_3.png", width: 15pt)
  ]
})

= Page with ArUco Markers

This page demonstrates positioning actual ArUco markers around specific regions.

NetID box would go here (top right):
#place(top + right, dx: -185pt, dy: 55pt)[
  #rect(width: 120pt, height: 20pt, stroke: gray, fill: none)
]

Answer bubbles would start here:
#place(top + left, dx: 55pt, dy: 135pt)[
  #rect(width: 400pt, height: 300pt, stroke: gray, fill: none)
]