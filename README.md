# AI-Based Resume Screening and Job Recommendation System Using NLP

> **Academic Level:** 3rd-Year BTech Computer Science & Engineering (CSE) Final Year Project  
> **Core Stack:** Python, Scikit-Learn, PyMuPDF, Pandas, NumPy, Streamlit, Joblib  
> **Approach:** Vector Space Model (TF-IDF Vectorization) + Cosine Similarity  

---

## 📌 Project Overview
The **AI-Based Resume Screening and Job Recommendation System** is an end-to-end Machine Learning and Natural Language Processing project designed to automate the initial stage of candidate resume screening and match candidates with appropriate career pathways.

Using an approved dataset of **1,016 real occupational profiles** from the **O\*NET 31.0 Database** and **600 real candidate resumes** from the **Djinni Recruitment Corpus**, this system:
1. Extracts text from digital PDF resumes using **PyMuPDF (`fitz`)**.
2. Cleans and normalizes text while strictly preserving essential programming languages, technical terms, and frameworks (`C++`, `C#`, `.NET`, `Python`, `SQL`, `AWS`, `AI`, `ML`, `NLP`, etc.).
3. Converts text into high-dimensional TF-IDF vectors (30,000 features, unigrams + bigrams).
4. Computes **Cosine Similarity** against all 1,016 occupational records.
5. Recommends the **Top 5 best-aligned job profiles**.
6. Computes **Matching Skills** and **Potential Missing Skills** strictly derived from explicit O\*NET occupational data without fabricating requirements.
7. Delivers an interactive, professional **Streamlit Web Dashboard**.

---

## 📂 Project Directory Structure

```text
AI_Resume_Job_Matcher/
│
├── data/
│   ├── resumes_verified.csv       # Original verified 600 resumes (Djinni Corpus)
│   ├── jobs_verified.csv          # Original verified 1,016 jobs (O*NET 31.0)
│   ├── clean_resumes.csv          # Processed candidate resumes with clean text
│   ├── clean_jobs.csv             # Processed job profiles with clean combined text
│   ├── sample_resume_cs.pdf       # Test PDF resume (BTech CSE / ML Student)
│   └── sample_resume_artist.pdf   # Test PDF resume (2D/3D Animator)
│
├── model/
│   ├── tfidf_vectorizer.joblib    # Fitted TF-IDF Vectorizer (30,000 features)
│   └── job_vectors.joblib         # Sparse TF-IDF matrix for all 1,016 jobs (1016, 30000)
│
├── app/
│   └── app.py                     # Interactive Streamlit frontend dashboard
│
├── notebooks/
│   └── exploration_and_evaluation.ipynb  # Jupyter notebook for viva/presentation
│
├── preprocessing.py               # Data cleaning and field combination pipeline
├── train_model.py                 # TF-IDF model training & vector serialization
├── evaluate_model.py              # Objective IR evaluation & distribution metrics
├── utils.py                       # Reusable PDF parser, text cleaner & skill matcher
├── requirements.txt               # Complete Python package dependencies
├── README.md                      # Complete setup, usage, and project documentation
└── MODEL_REPORT.md                # In-depth technical report and BTech project defense guide
```

---

## 📊 Dataset Transparency & Provenance

In strict compliance with academic research guidelines:
- **Zero fake resumes were created.**
- **Zero fake job records were created.**
- **Zero records were duplicated to artificially inflate dataset size.**
- **Original CSV files remain strictly unchanged.**

### 1. Resumes Dataset (`resumes_verified.csv`)
* **Total Records:** 600 real candidate profiles.
* **Source:** **Djinni Recruitment Dataset** (*Drushchak & Romanyshyn, UNLP @ LREC-COLING 2024*).
* **Paper URL:** https://aclanthology.org/2024.unlp-1.2/
* **Repository URL:** https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english
* **License:** MIT License.
* **Data Type:** 100% Anonymized real candidate profiles. All Personally Identifiable Information (PII) removed.
* **Missing Column Audit:** The Djinni dataset publishes candidate skills, experience, and background embedded directly inside `resume_text`. Dedicated isolated columns for `skills` and `education` were not published in the source schema; per data integrity rules, they are preserved as `"Not Available"` and never guessed.

### 2. Jobs Dataset (`jobs_verified.csv`)
* **Total Records:** 1,016 official detailed occupations.
* **Source:** **O\*NET 31.0 Database** (August 2026 Release, U.S. Department of Labor / USDOL/ETA).
* **Source URL:** https://www.onetcenter.org/database.html
* **License:** Creative Commons Attribution 4.0 International License (CC BY 4.0).
* **Missing Column Audit:** Newly created codes and residual "All Other" occupational classifications (e.g. *Managers, All Other*) do not undergo direct incumbent survey collection for tasks and skills. Missing fields are preserved as `"Not Available"`.

---

## ⚙️ Installation & Setup Instructions

### Step 1: Clone or Navigate to Project Directory
```bash
cd AI_Resume_Job_Matcher
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Workflow

### 1. Data Preprocessing
Cleans raw text, handles missing values, preserves technical acronyms (`C++`, `C#`, `.NET`, `Python`, `SQL`, `AWS`), and generates `clean_resumes.csv` and `clean_jobs.csv`.
```bash
python preprocessing.py
```

### 2. Model Training & Vectorization
Fits `TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_features=30000)` on the corpus, generates vector representations for all 1,016 job profiles, and serializes artifacts into `model/`.
```bash
python train_model.py
```

### 3. Model Evaluation & System Sanity Check
Computes vector sparsity, similarity distribution across all 609,600 candidate-job pairs, and runs qualitative domain checks.
```bash
python evaluate_model.py
```

### 4. Launch Streamlit Web Dashboard
Launches the interactive web application in your browser.
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 How the ML Model Works

### 1. Text Extraction with PyMuPDF
PyMuPDF (`fitz`) parses the internal PDF document streams, extracts structured text blocks across all document pages, and handles formatting boundaries.

### 2. Domain-Aware Cleaning
Specialized regex rules preserve technical symbols:
- `C++` $\rightarrow$ `cpp cplusplus`
- `C#` $\rightarrow$ `csharp`
- `.NET` $\rightarrow$ `dotnet`
- `Node.js` $\rightarrow$ `nodejs`
- `Power BI` $\rightarrow$ `powerbi`
- Lowercase conversion, non-alphanumeric noise stripping, and whitespace collapse.

### 3. TF-IDF Vectorization
The vectorizer assigns weights according to:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \left( \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \right)$$

- **Term Frequency (TF):** Measures how frequently term $t$ appears in the resume or job.
- **Inverse Document Frequency (IDF):** Penalizes generic words across the 1,616 corpus documents.
- Produces an L2-normalized 30,000-dimensional numerical vector for every document.

### 4. Cosine Similarity Matching
Measures the directional cosine angle between the candidate vector $\vec{u}$ and every job vector $\vec{v}$:

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$

Because all job vectors are pre-computed and stored in `job_vectors.joblib`, calculating cosine similarity against 1,016 occupations executes in **less than 5 milliseconds**.

### 5. Skill Gap Analysis (Zero Hallucination)
- **Matching Skills:** Skills explicitly listed in the job's `technology_skills` and `required_skills` that are identified in the resume.
- **Potential Missing Skills:** Skills explicitly required by the job that are not found in the candidate resume.
- **Strict Rule:** Only explicit O\*NET dataset entries appear in the missing skills list. No requirements are invented or guessed.

---

## 📈 Evaluation & System Metrics

| Metric Category | Value | Explanation |
| :--- | :--- | :--- |
| **Resumes Count** | 600 | Real candidate profiles from Djinni dataset |
| **Occupations Count** | 1,016 | Official detailed occupations in O\*NET 31.0 |
| **Vocabulary Size** | 30,000 | 8,927 Unigrams + 21,073 Bigrams |
| **Job Matrix Shape** | $(1016, 30000)$ | Pre-computed knowledge base |
| **Matrix Sparsity** | 97.27% | Specialized vocabularies across diverse job roles |
| **Mean Pairwise Similarity** | 0.57% | Average similarity across all 609,600 pairs |
| **99th Percentile Similarity**| 2.74% | Only 1% of pairs exceed 2.74% |
| **Top-1 Mean Similarity** | 4.93% | Average score of best-fit occupation per resume |
| **Maximum Similarity** | 14.85% | Highest single match in the dataset |

> **Important Note on Scores:**  
> In a 30,000-dimensional sparse TF-IDF space, raw similarity scores between **6% and 15%** represent top 0.1% matches (99.9th percentile). In the Streamlit dashboard, users can view both the **Normalized Match Index (0-100%)** for student-friendly presentation and the **Raw Cosine Similarity (0-100%)** for scientific precision.

---

## 🛠️ Common Errors & Fixes

### 1. `ModuleNotFoundError: No module named 'fitz'` or `'pymupdf'`
* **Cause:** PyMuPDF library is not installed in the active environment.
* **Fix:** Run:
  ```bash
  pip install PyMuPDF
  ```

### 2. `FileNotFoundError: Model file missing: model/tfidf_vectorizer.joblib`
* **Cause:** Streamlit was started before training the model.
* **Fix:** Run the preprocessing and training scripts first:
  ```bash
  python preprocessing.py
  python train_model.py
  ```

### 3. "No selectable text could be extracted from this PDF"
* **Cause:** The uploaded PDF is a scanned image (photograph of paper without a digital text layer).
* **Fix:** Upload a standard text-based PDF (exported from Word, Google Docs, or LaTeX) or use the **"Paste Resume Text"** tab in the dashboard.

### 4. `ModuleNotFoundError: No module named 'streamlit'`
* **Cause:** Dependencies not installed.
* **Fix:** Run:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📜 Academic Integrity & Disclaimer
- **Text Similarity Disclaimer:** The match scores produced by this system represent lexical and n-gram vocabulary overlap using TF-IDF and Cosine Similarity. They do not represent a guaranteed qualification, automated hiring decision, or legal employment determination.
- **Dataset Attribution:**
  - O\*NET 31.0 Database is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0) by USDOL/ETA.
  - Djinni Recruitment Dataset is licensed under the MIT License by Drushchak & Romanyshyn (UNLP 2024).
- Built for academic demonstration in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science and Engineering.
