# Redrob Intel-Ranker

An intelligent, high-performance candidate discovery and ranking system designed for the **Redrob Talent Discovery & Ranking Challenge**. 

This system evaluates candidate profiles against a **Founding Senior AI Engineer** job description, filtering out honeypots and service-company profiles while ranking active, product-focused AI talent using a combination of semantic text similarity and behavioral features.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your computer.

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🏃 Reproduction Command

To reproduce the ranking shortlist, run the following command. The script processes the candidate database and outputs the top 100 candidates to a validated CSV file.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*Note: The script completes execution on the full 100,000 candidate dataset in **under 1 minute** on a standard CPU machine.*

---

## 🖥️ Interactive Dashboard

We have included a Streamlit web application to visually inspect the candidate pool, customize ranking weights dynamically, and explore candidate profile details (career timelines, education, skills, and scoring breakdowns).

To launch the dashboard, run:
```bash
streamlit run app.py
```
Open the provided URL (usually `http://localhost:8501`) in your web browser.

---

## 🛠️ Architecture & Methodology

Our system uses a hybrid filtering and scoring architecture to ensure high accuracy while avoiding the traps (honeypots and keyword stuffers) embedded in the dataset:

1. **Honeypot & Trap Screening:** Identifies logical inconsistencies in resumes (e.g., zero-duration skills, impossible dates at newly founded startups) and automatically blacklists them.
2. **Experience & Title Matching:** Uses customized score curves favoring 6–8 years of experience, AI/ML titles, and hybrid availability (Noida/Pune).
3. **NLP Keyword Extraction:** Matches candidates against core skill groups (vector databases, dense retrieval, RAG, ranking metrics) and nice-to-haves (fine-tuning, MLOps).
4. **Semantic Cosine Similarity:** Computes TF-IDF vectors over candidate headlines, summaries, skills, and work descriptions to match relevance against the job description text.
5. **Behavioral Modifiers:** Adjusts scores based on recruiter response rates, login recency, and GitHub contributions.
6. **Deterministic Tiebreaking:** Sorts candidates descending by composite score, using `candidate_id` ascending as a tiebreaker.
7. **Personalized Reasoning:** Automatically creates structured, non-templated reasoning statements summarizing why each candidate was placed in their respective rank tier.
