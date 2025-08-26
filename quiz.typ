#import "quiz_template.typ": quiz

#show: quiz.with(
  title: "Columnar Databases",
  text_size: 12pt, // Assuming default size
  margin: 2cm,     // Using the left/right margin from old format
  date: "April 23, 2025"
)

+ What is the primary benefit of using a column-oriented DBMS for analytics?
  - [ ] Faster write speeds if columns are numeric and updated frequently
  - [ ] Improved data normalization for columns that depend on each other
  - [x] Faster read speeds for queries involving a subset of columns
  - [ ] Enhanced transaction concurrency, regardless of column types

+ Why might someone choose `DuckDB` for their project?
  - [ ] It is primarily designed for OLTP systems
  - [ ] It supports real-time data updating
  - [x] It operates as an embedded SQL database system with zero configuration
  - [ ] It uses a row-oriented storage approach that helps with write-heavy workloads

+ What is a significant advantage of the `Parquet` file format over `CSV`?
  - [ ] Better support for text data
  - [ ] Faster write speeds for numeric data
  - [x] More efficient storage and faster data retrieval
  - [ ] Higher data integrity using cryptographic hashing

+ Which of the following pairs describes `MongoDB` and `DuckDB` accurately?
  - [ ] columnar database, document-oriented database
  - [x] document-oriented database, columnar database
  - [ ] both are columnar databases
  - [ ] both are document-oriented databases

+ What does `Parquet` use to reduce the amount of data scanned during queries?
  - [ ] Partitioning by hash of primary key
  - [x] Metadata to skip irrelevant data chunks
  - [ ] Foreign key constraints, particularly for joins
  - [ ] Implicit `WHERE` clauses that filter out unnecessary data

+ What technique does `Parquet` use to handle columns with many repeating values?
  - [ ] B-tree indexing
  - [ ] Bitmap indexing
  - [x] Run-length encoding
  - [ ] Hash mapping

+ Which of the following is a common use case for `DuckDB`?
  - [ ] ​`OLTP` workloads
  - [x] ​`OLAP` workloads
  - [ ] Session storage for web applications
  - [ ] Social network/graph storage

+ Which of the following `JOIN` operations would likely produce the most rows?
  - [ ] ​`INNER JOIN`
  - [ ] ​`LEFT JOIN`
  - [x] ​`CROSS JOIN`
  - [ ] ​`FULL OUTER JOIN`

+ The Bitcoin blockchain is most similar to which of the following systems?
  - [ ] ​`DuckDB`
  - [ ] ​`MongoDB`
  - [ ] ​`PostgreSQL`
  - [x] ​`Git`

+ You export your data to a file for analysis. Which format is likely smallest on disk?
  - [ ] ​`CSV`
  - [x] ​`Parquet`
  - [ ] ​`JSON`
