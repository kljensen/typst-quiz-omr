---
title: In-class Handwritten SQL question
header-includes: |
  <style>
  body {
    max-width: 50em;
  }
  </style>
---

Imagine a Tinder-like app in which people swipe right (a "match")
or left on each other.

```sql
CREATE TABLE user (
    user_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE swipe (
    user_1 INTEGER NOT NULL REFERENCES user(user_id),
    user_2 INTEGER NOT NULL REFERENCES user(user_id),
    swiped_right BOOLEAN NOT NULL,
    CHECK (user_1 <> user_2),
    PRIMARY KEY (user_1, user_2)
);
```

Write a single SQL query to show which users jointly match. Only
show one row per match (Kyle and Jeff just once: not Kyle
and Jeff then Jeff and Kyle.) You can use just the `swipe` table and
have output that looks like

```txt
| user_1   | user_2   |
|----------|----------|
| 39       | 89       |
| 43       | 49       |
```
