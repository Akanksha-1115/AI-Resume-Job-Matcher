"""
================================================================================
AI-Based Resume Screening and Job Recommendation System Using NLP
File: train_model.py
Description: Trains TF-IDF Vectorizer and generates job profile vectors.
Author: 3rd-Year BTech CSE Student
================================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def train_and_save_model(
    data_dir: str = "data",
    model_dir: str = "model"
):
    """
    Fits the TF-IDF Vectorizer on the cleaned job and resume text corpus,
    transforms all 1,016 job profiles into vector representations, and
    saves the vectorizer and job vectors to disk using Joblib.
    """
    print("=" * 80)
    print("AI_Resume_Job_Matcher: MODEL TRAINING & VECTORIZATION")
    print("=" * 80)

    clean_jobs_path = os.path.join(data_dir, "clean_jobs.csv")
    clean_resumes_path = os.path.join(data_dir, "clean_resumes.csv")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    job_vectors_path = os.path.join(model_dir, "job_vectors.joblib")

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Cleaned Datasets
    if not os.path.exists(clean_jobs_path):
        raise FileNotFoundError(f"Missing {clean_jobs_path}. Run preprocessing.py first!")
    if not os.path.exists(clean_resumes_path):
        raise FileNotFoundError(f"Missing {clean_resumes_path}. Run preprocessing.py first!")

    print(f"\n[1/5] Loading preprocessed datasets from '{data_dir}'...")
    df_jobs = pd.read_csv(clean_jobs_path)
    df_resumes = pd.read_csv(clean_resumes_path)

    print(f"      - Jobs loaded    : {len(df_jobs)} records")
    print(f"      - Resumes loaded : {len(df_resumes)} records")

    # Combine text series for comprehensive vocabulary representation
    # Fill any potential NaN with empty string
    jobs_text = df_jobs['clean_combined_text'].fillna("")
    resumes_text = df_resumes['clean_combined_text'].fillna("")
    full_corpus = pd.concat([jobs_text, resumes_text], ignore_index=True)

    # 2. Configure TF-IDF Vectorizer strictly as specified
    print("\n[2/5] Configuring TF-IDF Vectorizer...")
    print("      - stop_words   : 'english'")
    print("      - ngram_range  : (1, 2) [unigrams and bigrams]")
    print("      - min_df       : 2")
    print("      - max_features : 30000")

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000
    )

    # 3. Fit Vectorizer on Corpus
    print("\n[3/5] Fitting TF-IDF Vectorizer on full training corpus...")
    tfidf_vectorizer.fit(full_corpus)
    vocab_size = len(tfidf_vectorizer.vocabulary_)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    unigram_count = sum(1 for f in feature_names if " " not in f)
    bigram_count = sum(1 for f in feature_names if " " in f)

    print(f"      -> Vocabulary fitted successfully!")
    print(f"      - Total features learned : {vocab_size:,}")
    print(f"      - Unigrams (single words): {unigram_count:,}")
    print(f"      - Bigrams (two words)    : {bigram_count:,}")

    # 4. Transform Job Profiles into Vectors
    print(f"\n[4/5] Vectorizing all {len(df_jobs)} job profiles...")
    job_vectors = tfidf_vectorizer.transform(jobs_text)
    
    # Calculate matrix density / sparsity
    n_nonzero = job_vectors.nnz
    total_elements = job_vectors.shape[0] * job_vectors.shape[1]
    sparsity = (1.0 - (n_nonzero / total_elements)) * 100.0

    print(f"      - Job vectors matrix shape: {job_vectors.shape}")
    print(f"      - Non-zero entries        : {n_nonzero:,}")
    print(f"      - Matrix sparsity         : {sparsity:.2f}%")

    # 5. Persist Model Artifacts
    print("\n[5/5] Saving model artifacts via Joblib...")
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(job_vectors, job_vectors_path)
    print(f"      -> Saved vectorizer : {vectorizer_path} ({os.path.getsize(vectorizer_path)/1024:.1f} KB)")
    print(f"      -> Saved job vectors: {job_vectors_path} ({os.path.getsize(job_vectors_path)/1024:.1f} KB)")

    # ------------------------------------------------------------------
    # SANITY CHECKS & VALIDATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL VALIDATION & SANITY CHECKS")
    print("=" * 80)

    # Sanity Check 1: Software / Machine Learning Profile
    sample_cs = (
        "btech cse computer science engineering student python sql machine learning "
        "scikit learn pandas numpy deep learning nlp git docker fastapi model training "
        "predictive modeling data pipelines"
    )
    vec_cs = tfidf_vectorizer.transform([sample_cs])
    sims_cs = cosine_similarity(vec_cs, job_vectors).flatten()
    top5_cs = np.argsort(sims_cs)[::-1][:5]

    print("\n[Sanity Check 1] Test Resume: Computer Science / ML Profile")
    for rank, idx in enumerate(top5_cs, 1):
        score_pct = sims_cs[idx] * 100
        title = df_jobs.iloc[idx]['job_title']
        print(f"  {rank}. {title:50s} | Similarity: {score_pct:5.2f}%")

    # Sanity Check 2: 2D/3D Animation Profile
    sample_art = (
        "2d artist 3d animator animation spine2d 3dmax character design concept art "
        "photoshop illustration motion design storyboard digital painting"
    )
    vec_art = tfidf_vectorizer.transform([sample_art])
    sims_art = cosine_similarity(vec_art, job_vectors).flatten()
    top5_art = np.argsort(sims_art)[::-1][:5]

    print("\n[Sanity Check 2] Test Resume: 2D/3D Animator Profile")
    for rank, idx in enumerate(top5_art, 1):
        score_pct = sims_art[idx] * 100
        title = df_jobs.iloc[idx]['job_title']
        print(f"  {rank}. {title:50s} | Similarity: {score_pct:5.2f}%")

    print("\n" + "=" * 80)
    print("Training and validation completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    model_dir = os.path.join(script_dir, "model")
    if not os.path.exists(data_dir):
        data_dir = os.path.join("AI_Resume_Job_Matcher", "data")
        model_dir = os.path.join("AI_Resume_Job_Matcher", "model")
    train_and_save_model(data_dir=data_dir, model_dir=model_dir)
