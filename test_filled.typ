#import "quiz_template.typ": quiz

#show: quiz.with(
  title: "Filled Bubble Test",
  show_filled: true  // Enable visual bubble filling for testing
)

1. First question (select B)
  - [ ] Option A
  - [x] Option B  
  - [ ] Option C
  - [ ] Option D

2. Second question (select A and C)
  - [x] Option A
  - [ ] Option B
  - [x] Option C
  - [ ] Option D

3. Third question (select D)
  - [ ] Option A
  - [ ] Option B
  - [ ] Option C
  - [x] Option D