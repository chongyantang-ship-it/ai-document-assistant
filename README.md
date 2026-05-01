# AI-Powered Academic Document Assistant

## Project Overview
This project is an AI-powered academic document assistant for Assessment 3 of 36121 Artificial Intelligence Principles and Applications.
The system helps students understand assignment documents, report templates, oral presentation requirements, submission rules, and project expectations.

The assistant is implemented as a hybrid AI system combining structured knowledge representation, rule-based reasoning, semantic embedding retrieval, retrieval-augmented generation principles, and evidence-grounded answer generation.

## Real-World Problem
Students often need to interpret long and complex assignment documents. Important details such as due dates, word limits, file formats, similarity limits, GitHub requirements, presentation rules, and report sections may be spread across multiple documents.
This project addresses this problem by answering assignment-related questions using evidence from the provided documents.

## AI Methods Used
- Structured Knowledge Representation: exact assessment facts are stored in data/processed/structured_facts.json.
- Rule-Based Reasoning: exact questions about deadlines, word limits, file formats and GitHub requirements are answered using deterministic rules.
- Semantic Embedding Retrieval: document chunks and user questions are encoded with sentence embeddings and compared using cosine similarity.
- RAG Principle: open-ended questions retrieve relevant document chunks before answer generation.
- Evidence-Grounded Answer Generation: answers include supporting evidence and confidence levels.
- Empirical Evaluation: the system is compared with a keyword search baseline.

## Repository Structure
README.md
requirements.txt
data/raw/
data/processed/
src/
evaluation/
report_figures/

## Installation
pip install -r requirements.txt

## How to Run
python src/app.py

## Example Questions
- When is the report due?
- What is the report word limit?
- What file format should the slides use?
- Is a GitHub repository required?
- What AI methods are used in this project?
- What is the system pipeline?
- How should the system be evaluated?
- Can we submit late without approval?

## Evaluation
Run the evaluation with:
python evaluation/evaluate.py

The evaluation compares Keyword Search and the Proposed Hybrid System.

## Main Evaluation Results
| Method | Answer Accuracy | Citation Support Rate | Hallucination Rate | Unsupported Handling Accuracy |
|---|---:|---:|---:|---:|
| Keyword Search | 0.333 | 0.333 | 0.667 | 0.867 |
| Proposed Hybrid System | 1.000 | 0.867 | 0.000 | 1.000 |

## Limitations
- The dataset is small and based on limited assignment documents.
- The answer generator is template-based rather than a full LLM.
- Retrieval quality depends on chunking quality.
- The evaluation dataset is manually created and relatively small.
- The system should not replace official advice from lecturers or tutors.

## Ethical Considerations
The assistant is designed as a learning support tool. It should help students understand assignment requirements but should not be used to generate entire assignments dishonestly.
The system should provide evidence, show uncertainty when documents are insufficient, and encourage students to check official course instructions for high-stakes decisions.

## Future Work
- Add an LLM-based answer generator.
- Improve query classification.
- Expand the evaluation dataset.
- Support more course documents.
- Add a Streamlit web interface.