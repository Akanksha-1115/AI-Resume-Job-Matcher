# Model & Technical Report
## AI-Based Resume Screening and Job Recommendation System Using NLP
**Project Level:** 3rd-Year BTech Computer Science & Engineering (CSE)  
**Domain:** Natural Language Processing (NLP) / Information Retrieval (IR)  
**Methodology:** Vector Space Model with TF-IDF Vectorization and Cosine Similarity  

---

## 1. Problem Statement
In modern recruitment processes, technical recruiters and automated talent portals receive hundreds to thousands of candidate resumes for diverse technical, managerial, and creative openings. Manual screening is time-consuming, prone to human fatigue, and often inconsistent. 

The objective of this project is to build an explainable, automated Natural Language Processing (NLP) system that:
1. Extracts text from candidate resumes in PDF format.
2. Cleans and normalizes freeform text while strictly preserving essential technical terms (e.g., C++, C#, .NET, Python, SQL, AWS, Docker).
3. Transforms candidate profiles and job occupation descriptions into high-dimensional numerical vectors using TF-IDF (Term Frequency - Inverse Document Frequency).
4. Matches the candidate's profile against an approved catalog of **1,016 real occupational profiles** using Cosine Similarity.
5. Recommends the **Top 5 most relevant occupations**.
6. Identifies **Matching Skills** and **Potential Missing Skills** based strictly on the job's explicit requirements without hallucinating or generating synthetic data.

---

## 2. Dataset Information & Provenance
In strict compliance with academic research integrity, **no fake, synthetic, or generated records were used**. The project runs on two verified real-world datasets:

### A. Candidate Resumes Dataset (`resumes_verified.csv`)
* **Total Records:** 600 verified real candidate profiles.
* **Source:** **Djinni Recruitment Dataset** (*Drushchak & Romanyshyn, UNLP @ LREC-COLING 2024*).
* **Source URL:** [Hugging Face Recruitment Dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)
* **License:** MIT License (Permissive open research and commercial use).
* **Fields:** `resume_id`, `resume_text`, `category`, `skills`, `education`, `experience`, `job_title`, `source_dataset`, `source_url`, `original_record_id`, `data_type`, `license`.
* **Missing Value Audit:** In the Djinni dataset, candidate skills, projects, and educational backgrounds are embedded directly within the freeform profile text (`resume_text`). As separate tabular columns, `skills` and `education` were not published by the authors; following data integrity rules, they are preserved as `"Not Available"` and never fabricated.

### B. Occupational Profiles Dataset (`jobs_verified.csv`)
* **Total Records:** 1,016 unique, detailed occupational records.
* **Source:** **O\*NET 31.0 Database** (Official release by the U.S. Department of Labor / Employment and Training Administration - USDOL/ETA).
* **Source URL:** [O\*NET Resource Center](https://www.onetcenter.org/database.html)
* **License:** Creative Commons Attribution 4.0 International License (CC BY 4.0).
* **Fields:** `job_id`, `job_title`, `occupation_code`, `alternate_titles`, `job_description`, `required_skills`, `knowledge`, `abilities`, `tasks`, `education`, `experience`, `technology_skills`, `source_dataset`, `source_url`, `original_record_id`, `license`.
* **Missing Value Audit:** Residual "All Other" occupational classifications (e.g., *11-9199.00 - Managers, All Other*) and military-specific codes do not undergo direct survey data collection for tasks and skills. These 106 records preserve `"Not Available"` in accordance with official O\*NET metadata.

---

## 3. Data Preprocessing Pipeline
Raw textual data in resumes and occupational documents contains formatting irregularities, special characters, URLs, and punctuation. The preprocessing pipeline implemented in `preprocessing.py` performs systematic cleaning:

1. **Missing Value & Placeholder Isolation:**
   - Any column containing `"Not Available"`, `"nan"`, or empty strings is omitted from text concatenation so that non-informative placeholder tokens do not contaminate the vocabulary.
2. **Technical Term Preservation:**
   - Standard regex punctuation removal destroys critical technical symbols (e.g., `C++` becomes `C`, `C#` becomes `C`, `.NET` becomes `NET`).
   - We explicitly preserve and alias these terms prior to punctuation stripping:
     - `C++` $\rightarrow$ `cpp cplusplus`
     - `C#` $\rightarrow$ `csharp`
     - `.NET` / `DOTNET` $\rightarrow$ `dotnet`
     - `Node.js` $\rightarrow$ `nodejs`, `React.js` $\rightarrow$ `reactjs`, `Vue.js` $\rightarrow$ `vuejs`
     - `CI/CD` $\rightarrow$ `cicd`, `TCP/IP` $\rightarrow$ `tcpip`, `Power BI` $\rightarrow$ `powerbi`
3. **Case Normalization:** Converts all text to lowercase to ensure case-insensitive matching.
4. **Noise Removal:** Strips URL hyperlinks (`https?://\S+`) and candidate email addresses.
5. **Punctuation Cleaning:** Non-alphanumeric characters (hyphens, commas, colons, brackets) are replaced with whitespace while preserving alphanumeric tokens and underscores.
6. **Whitespace Collapsing:** Eliminates extra spaces, line breaks, and tabs.
7. **Combined Text Generation:**
   - **Resumes:** Combines `job_title`, `resume_text`, `skills`, `education`, and `experience`.
   - **Jobs:** Combines `job_title`, `job_description`, `required_skills`, `knowledge`, `abilities`, `tasks`, `education`, `experience`, and `technology_skills`.

---

## 4. Feature Extraction
To compare unstructured text computationally, textual documents must be converted into numerical vector spaces.

We use **N-gram Modeling**:
- **Unigrams ($n=1$):** Captures individual technical tokens like `python`, `sql`, `docker`, `aws`, `java`.
- **Bigrams ($n=2$):** Captures multi-word phrases and compound concepts like `machine learning`, `data science`, `project management`, `deep learning`, `database architects`.

The feature vocabulary is bounded to a maximum of **30,000 features** with a minimum document frequency ($df \ge 2$) to eliminate one-off typos and retain meaningful occupational terminology.

---

## 5. TF-IDF Algorithm (Mathematical Details)
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It balances how often a word appears in a specific document against how common that word is across the entire corpus.

### A. Term Frequency (TF)
Term frequency measures how frequently term $t$ appears in document $d$:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

Where $f_{t, d}$ is the count of term $t$ in document $d$. A higher count indicates greater topical relevance to that document.

### B. Inverse Document Frequency (IDF)
Words that appear in virtually every document (e.g., "work", "responsibilities", "skills", "experience") provide little discriminative power. IDF penalizes universally frequent words:

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Where:
- $|D|$ is the total number of documents in the corpus (1,616 total documents = 1,016 jobs + 600 resumes).
- $|\{d \in D : t \in d\}|$ is the number of documents containing term $t$.
- Adding $1$ to numerator and denominator provides Laplacian smoothing to avoid division by zero.

### C. Combined TF-IDF Weight
The final weight assigned to term $t$ in document $d$ is:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

The resulting vector for document $d$ is L2-normalized:

$$\|\vec{v}\|_2 = \sqrt{\sum_{i=1}^M v_i^2} = 1.0$$

---

## 6. Cosine Similarity
To measure the degree of alignment between a candidate's resume vector $\vec{u}$ and an occupation profile vector $\vec{v}$, we compute the **Cosine Similarity**:

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^M u_i v_i}{\sqrt{\sum_{i=1}^M u_i^2} \sqrt{\sum_{i=1}^M v_i^2}}$$

### Why Cosine Similarity?
1. **Length Invariance:** A 4-page resume and a 1-page resume discussing the exact same skills will produce almost identical directional vectors. Cosine similarity measures the angle between vectors, not Euclidean distance (vector magnitude).
2. **Bounded Output:** For non-negative TF-IDF representations, cosine similarity yields a score strictly in the range $[0.0, 1.0]$ ($0\%$ to $100\%$).
3. **High Efficiency:** Computing cosine similarity between a $1 \times 30000$ sparse vector and a $1016 \times 30000$ sparse matrix takes under **5 milliseconds** in Scikit-Learn.

---

## 7. Recommendation Process
The end-to-end recommendation workflow operates as follows:

```
[Candidate Resume PDF]
         │
         ▼
[PyMuPDF / fitz Stream Parser] ──► Extracts clean raw text across all pages
         │
         ▼
[Text Cleaning & Preservation] ──► Aliases technical terms, lowercases, cleans punctuation
         │
         ▼
[TF-IDF Vectorizer Transform] ──► Converts text into 1 x 30,000 sparse vector
         │
         ▼
[Cosine Similarity Matrix] ──► Dot product with pre-computed 1,016 x 30,000 job vectors
         │
         ▼
[Descending Sort & Top 5 Selection]
         │
         ▼
[Explicit Skill Overlap Engine]
   ├─► Matching Skills (in resume & in job's explicit skills)
   └─► Missing Skills (in job's explicit skills, NOT in resume)
         │
         ▼
[Interactive Streamlit Dashboard Display]
```

---

## 8. Evaluation & Validation
### A. Why Supervised Metrics (Accuracy / Precision / Recall) Are Inapplicable
In academic research, supervised classification metrics require a labeled ground-truth matrix where human recruiters have evaluated and tagged every candidate against all 1,016 occupations with a binary target ($y \in \{0, 1\}$). 

Because this is an **unsupervised information retrieval / recommendation system**, we report objective system metrics, vector sparsity, similarity distribution statistics, and domain sanity checks.

### B. Quantitative Corpus & Vector Metrics
| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Resumes Processed** | 600 | Real candidate profiles from Djinni dataset |
| **Job Profiles Vectorized** | 1,016 | Official detailed occupations in O\*NET 31.0 |
| **Vocabulary Size** | 30,000 | 8,927 Unigrams + 21,073 Bigrams |
| **Job Vector Matrix Shape** | $(1016, 30000)$ | Full occupational knowledge base |
| **Matrix Non-Zero Elements** | 833,271 | Actual term occurrences |
| **Matrix Sparsity** | **97.27%** | Sparse representation reflects specialized domain vocabularies |

### C. Similarity Distribution Across 609,600 Candidate-Job Pairs
Evaluating all $600 \times 1,016 = 609,600$ pairwise cosine similarity scores reveals:

| Statistic | Value (%) | Interpretation |
| :--- | :--- | :--- |
| **Corpus Mean Similarity** | **0.57%** | Most job-resume pairs share minimal vocabulary |
| **Corpus Median Similarity**| **0.44%** | Unrelated occupations have near-zero overlap |
| **95th Percentile** | **1.56%** | Only top 5% of pairs score above 1.56% |
| **99th Percentile** | **2.74%** | Extreme top 1% cutoff |
| **Top-1 Best Match Mean** | **4.93%** | Average score of the highest-ranked job per candidate |
| **Maximum Observed** | **14.85%** | Highest single pair alignment in the entire dataset |

> **Key Takeaway for College Defense:**  
> In a 30,000-dimensional sparse TF-IDF space, raw similarity scores between **6% and 15%** represent top-tier alignment (the 99.9th percentile). This mathematically explains why raw cosine scores in high-dimensional IR are not 90% and why relative ranking is the primary criterion.

### D. Qualitative Domain Sanity Checks
Testing diverse candidates against the 1,016 occupations demonstrates consistent domain-congruent recommendations:

1. **Test Profile 1: Full-Stack / Blockchain Lead**
   - *Top Recommendations:* 
     1. Blockchain Engineers (11.52%)
     2. Information Technology Project Managers (7.18%)
     3. Web Developers (6.98%)
   - *Outcome:* Spot-on alignment with candidate's Solidity/Web3 and engineering management background.

2. **Test Profile 2: 2D/3D Animator**
   - *Top Recommendations:*
     1. Special Effects Artists and Animators (3.37%)
     2. Video Game Designers (2.89%)
   - *Outcome:* Retains artistic and animation taxonomy without matching unrelated engineering jobs.

3. **Test Profile 3: B.Tech CSE / Machine Learning Student**
   - *Top Recommendations:*
     1. Data Scientists (8.85%)
     2. Database Architects (7.13%)
     3. Validation Engineers (7.07%)
   - *Outcome:* Accurately matches data engineering, scientific modeling, and software infrastructure roles.

---

## 9. Limitations
1. **Lexical / Bag-of-Words Constraint:** TF-IDF relies on exact word and n-gram overlap. It cannot recognize semantic synonyms that are not present in n-grams (e.g., "Kubernetes administrator" vs "container orchestration engineer") unless both terms appear in the corpus.
2. **Context Blindness:** A candidate stating "interested in learning Python" receives the same TF-IDF credit for "Python" as a senior engineer with "10 years leading Python development".
3. **Unsupervised Recommendation:** Without ground-truth hiring data, the system evaluates lexical alignment, not verified candidate capability.
4. **Scanned PDF Limitation:** PyMuPDF extracts text streams from digital PDFs. Scanned image-based resumes without OCR layers cannot be parsed without dedicated OCR tools (e.g., Tesseract).

---

## 10. Future Improvements
1. **Dense Semantic Embeddings:** Upgrade from sparse TF-IDF to transformer-based dense bi-encoders (e.g., `all-MiniLM-L6-v2` or `Sentence-BERT`) to capture deep semantic similarity beyond exact keywords.
2. **Experience-Weighted Matching:** Extract numerical years of experience using Named Entity Recognition (NER) to weight seniority against job requirements.
3. **Skill Graph Knowledge Base:** Build an explicit graph (e.g., Neo4j) linking technologies (e.g., PyTorch $\rightarrow$ Deep Learning $\rightarrow$ Artificial Intelligence) to enable hierarchical skill matching.
4. **Two-Stage Hybrid Retrieval:** Use TF-IDF for rapid candidate filtering (Candidate Retrieval Stage) followed by a Cross-Encoder Transformer for fine-grained ranking (Re-ranking Stage).
