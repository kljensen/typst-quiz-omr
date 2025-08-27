#let quiz(
  title: none,
  margin: 1cm,
  date: none,
  options: "ABCDE",
  text_size: 10pt,
  doc,
) = {

   set page(paper: "us-letter")


  // As we process questions, we will increment
  // this counter. We'll need this to assemble
  // an array of correct answers and also to print
  // an answer key with the correct number of rows.
  let question_number = counter("question_number")
  question_number.update(0)

  // This is an array of arrays. Each sub-array has
  // the correct answers for a question.
  let correct_answers = state("correct_answers", ())

  // Function for printing the answer key
  // at the end of the quiz.
  let answer_key() = {
    let data = correct_answers.get().enumerate().map(((i, answer)) => {
      let question = i + 1
      let options = answer.map(o => {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ".at(o)
      }).join(", ")
      return ([#question], [#options])
    }).flatten()
    table(columns: 2, ..data)
  }

  let question_number = counter("question_number")
  let option_number = counter("option_number")

  show list.where(indent: 0pt): it => {
    option_number.update(0)
    question_number.step()
    let bodies = it.children.map( c => {
      c.body
    })
    enum(numbering: "A.", ..bodies)
  }

  show "[x] ": it => {
    context {
      let qn = question_number.get().first()
      let on = option_number.get().first()
      option_number.step()
      correct_answers.update( ca => {
        // Ensure array is large enough
        while ca.len() <= qn - 1 {
          ca.push(())
        }
        ca.at(qn - 1).push(on)
        return ca
      })
      // [#qn #on x]
      [#sym.zws#label(str(qn) + "-" + str(on))]
    }
  }
  show "[ ] ": it => {
    context {
      let qn = question_number.get().first()
      let on = option_number.get().first()
      // [#qn #on]
      [#sym.zws#label(str(qn) + "-" + str(on))]
      option_number.step()
    }
  }


  // --------------------------------------------
  // First page: instructions and answer bubbles
  // --------------------------------------------

  // Title and date
  {
    set page(
      margin: 1.5in,

    )
    set page(header: [
      #set align(right)
      #overline(offset: -1.5em)[~~ \u{2191} Your Yale netid \u{2191} ~~]
    ])
    set align(center)
    text(17pt, title)
    if date == none {
      date = datetime.today().display("[month repr:long] [day], [year]")
    }
    parbreak()
    date
    v(1cm)

    // Instructions
    [== Instructions]
    {

    rect(inset: 1em)[
      #set list(indent: 0.1pt)
      #set align(left)
      - Do not flip over this page or read the questions on the other side of this page until the start of class.
      - Your quiz will be collected after ten minutes.
      - Write your netid _clearly_ at the top right.
      - _Unless otherwise stated_, each question has one correct answer.
      - Fill in the appropriate bubble below. I will grade nothing but these bubbles. If you need to change an answer please indicate your final answer clearly.
      - If a question stinks, Kyle will fix it later. I will not answer questions during the quiz.
      - This quiz is closed book, closed device. You can only use your own meat computer.
      - When you're done, _raise your hand_ and an instructor will collect it.
    ]
    }

      v(1cm)
    columns(2,
    // Answer bubbles
    context {
      set align(left)
      box[
        == Multiple Choice Answers
      ]
      v(0.25cm)

      let row_of_bubbles = options.clusters().map(c => {
        [#box(circle(inset: 1pt)[
          #set align(center)
          #set text(size: 6pt)
          #c
        ])]
      }).join(" ")

      let elements = ()
      let qn = 1
      let num_questions = question_number.final().first()
      while qn < num_questions + 1 {

        // Write the question number. We
        // shift this to align with bubbles.
        let label = block[
          #set text(baseline: 0.05em)
          #qn.
        ]
        elements.push(label)
        elements.push(row_of_bubbles)
        qn += 1
      }
      
      // Add corner brackets around the bubble grid for OMR detection
      let bracket_size = 20pt  // Increased from 12pt
      let bracket_thickness = 4pt + black  // Increased from 2pt
      let grid_content = grid(
        columns: 2,
        align: (right, center),
        gutter: .75em,
        ..elements
      )
      
      // Wrap grid with brackets positioned at corners
      box(width: 100%)[
        #place(
          top + left,
          dx: -5pt, dy: -5pt,
          {
            line(start: (bracket_size, 0pt), end: (0pt, 0pt), stroke: bracket_thickness)
            line(start: (0pt, 0pt), end: (0pt, bracket_size), stroke: bracket_thickness)
          }
        )
        #place(
          top + right,
          dx: 5pt, dy: -5pt,
          {
            line(start: (-bracket_size, 0pt), end: (0pt, 0pt), stroke: bracket_thickness)
            line(start: (0pt, 0pt), end: (0pt, bracket_size), stroke: bracket_thickness)
          }
        )
        #place(
          bottom + left,
          dx: -5pt, dy: 5pt,
          {
            line(start: (0pt, -bracket_size), end: (0pt, 0pt), stroke: bracket_thickness)
            line(start: (0pt, 0pt), end: (bracket_size, 0pt), stroke: bracket_thickness)
          }
        )
        #place(
          bottom + right,
          dx: 5pt, dy: 5pt,
          {
            line(start: (0pt, -bracket_size), end: (0pt, 0pt), stroke: bracket_thickness)
            line(start: (0pt, 0pt), end: (-bracket_size, 0pt), stroke: bracket_thickness)
          }
        )
        #grid_content
      ]
      colbreak()
      box[
        == Handwritten SQL Query
      ]
      v(0.25cm)
    } 
    )
  }
  pagebreak()


  // --------------------------------------------
  // Second page: quiz questions
  // --------------------------------------------
  {
    set page(
      columns: 2,
      margin: margin,
    )
    set text(size: text_size)

    // This is the document body that is received
    // as the final argument to the quiz template.
    doc
    pagebreak()
  }

  // --------------------------------------------
  // Third page: answer key
  // --------------------------------------------
  {
    set align(center)
    [== Answer Key]
    context[
      #answer_key()
    ]
    // context[
    //   #correct_answers.get()
    // ]
  }

}
