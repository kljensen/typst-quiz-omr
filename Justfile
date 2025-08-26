# Build quiz PDF from markdown through typst
default: compile

compile:
    typst compile quiz.typ

watch: view
    typst watch quiz.typ

view:
    open -a Skim quiz.pdf

typeset-in-class-question:
    pandoc --standalone in-class-question.md -o in-class-question.html
