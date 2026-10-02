import os
import sys
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import extract_text_from_pdf, clean_text, compute_job_recommendations

def test_full_pipeline():
    print("=" * 80)
    print("TESTING COMPLETE RESUME SCREENING & JOB RECOMMENDATION WORKFLOW")
    print("=" * 80)

    # Step 1: Check artifact existence
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resumes_path = os.path.join(base_dir, "data", "clean_resumes.csv")
    jobs_path = os.path.join(base_dir, "data", "clean_jobs.csv")
    vec_path = os.path.join(base_dir, "model", "tfidf_vectorizer.joblib")
    job_vec_path = os.path.join(base_dir, "model", "job_vectors.joblib")
    pdf_path = os.path.join(base_dir, "data", "sample_resume_cs.pdf")

    print("\n[Step 1] Verifying and Loading Artifacts:")
    df_resumes = pd.read_csv(resumes_path)
    df_jobs = pd.read_csv(jobs_path)
    vectorizer = joblib.load(vec_path)
    job_vectors = joblib.load(job_vec_path)

    print(f"  - clean_resumes.csv loaded : {len(df_resumes)} rows (Expected 600)")
    print(f"  - clean_jobs.csv loaded    : {len(df_jobs)} rows (Expected 1016)")
    print(f"  - tfidf_vectorizer.joblib  : {len(vectorizer.vocabulary_)} features")
    print(f"  - job_vectors.joblib       : Shape {job_vectors.shape}")

    assert len(df_resumes) == 600
    assert len(df_jobs) == 1016
    assert len(vectorizer.vocabulary_) == 30000
    assert job_vectors.shape == (1016, 30000)

    # Step 2: Read PDF bytes (exactly what st.file_uploader supplies)
    print("\n[Step 2] Testing PDF File Upload & Extraction:")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    print(f"  - Read {len(pdf_bytes)} bytes from '{pdf_path}'")

    extracted_text, extract_err = extract_text_from_pdf(pdf_bytes)
    assert extract_err is None, f"PDF extraction failed: {extract_err}"
    assert len(extracted_text) > 0, "PDF extraction produced 0 characters"
    print(f"  - Text extracted successfully: {len(extracted_text)} characters")
    print(f"  - First 120 chars: {extracted_text[:120].strip()}...")

    # Step 3: Text Preprocessing & Cleaning
    print("\n[Step 3] Text Preprocessing & Technical Term Preservation:")
    cleaned_text = clean_text(extracted_text)
    assert len(cleaned_text) > 0
    # Check that technical terms like python, sql, c++, etc. are preserved
    print(f"  - Cleaned text word count: {len(cleaned_text.split())} words")
    print(f"  - Contains 'python': {'python' in cleaned_text}")
    print(f"  - Contains 'sql': {'sql' in cleaned_text}")
    print(f"  - Contains 'cpp/cplusplus': {'cpp' in cleaned_text or 'cplusplus' in cleaned_text}")

    # Step 4: Resume Vectorization & Cosine Similarity
    print("\n[Step 4] Resume Vectorization & Cosine Similarity against 1,016 jobs:")
    recs, _ = compute_job_recommendations(
        raw_resume_text=extracted_text,
        vectorizer=vectorizer,
        job_vectors=job_vectors,
        df_jobs=df_jobs,
        top_n=5
    )
    assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
    print(f"  - Computed Top {len(recs)} recommendations successfully.")

    # Step 5: Verification of recommendation output fields
    print("\n[Step 5] Detailed Recommendation Results Inspection:")
    for r in recs:
        print(f"\n  Rank #{r['rank']}: {r['job_title']}")
        print(f"    - Match Score           : {r['similarity_percentage']}% (Raw Cosine Similarity)")
        print(f"    - O*NET-SOC Code        : {r['occupation_code']}")
        print(f"    - Description Snippet   : {r['job_description'][:100]}...")
        print(f"    - Matching Skills ({len(r['matching_skills'])}): {', '.join(r['matching_skills'][:5])}...")
        print(f"    - Potential Missing ({len(r['missing_skills'])}): {', '.join(r['missing_skills'][:5])}...")
        print(f"    - Education             : {r['education'][:80]}...")
        print(f"    - Experience            : {r['experience'][:80]}...")

        # Ensure all required fields exist
        assert r['job_title'] != "Not Available"
        assert r['similarity_percentage'] > 0
        assert isinstance(r['matching_skills'], list)
        assert isinstance(r['missing_skills'], list)
        assert len(r['matching_skills']) > 0

    # Step 6: Test Error Handling for Invalid / Empty PDFs
    print("\n[Step 6] Testing Error Handling for Edge Cases:")
    # Empty bytes
    _, err_empty = extract_text_from_pdf(b"")
    print(f"  - Empty bytes test error: {err_empty}")
    assert err_empty is not None

    # Corrupt PDF bytes
    _, err_corrupt = extract_text_from_pdf(b"%PDF-invalid-bytes-12345")
    print(f"  - Corrupt bytes test error: {err_corrupt}")
    assert err_corrupt is not None

    print("\n" + "=" * 80)
    print("ALL WORKFLOW STEPS VALIDATED AND PASSING PERFECTLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_full_pipeline()
