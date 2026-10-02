import os
import joblib
import pandas as pd
from utils import extract_text_from_pdf, compute_job_recommendations

pdf_path = os.path.join("data", "sample_resume_cs.pdf")
text, err = extract_text_from_pdf(pdf_path)
print("PDF extraction error:", err)
print("PDF text length:", len(text) if text else 0)

vec = joblib.load(os.path.join("model", "tfidf_vectorizer.joblib"))
job_vectors = joblib.load(os.path.join("model", "job_vectors.joblib"))
df_jobs = pd.read_csv(os.path.join("data", "clean_jobs.csv"))

recs, cleaned = compute_job_recommendations(text, vec, job_vectors, df_jobs, top_n=5)
print(f"\nGenerated {len(recs)} recommendations:")
for r in recs:
    print(f"\n{r['rank']}. {r['job_title']} (Match Score: {r['similarity_percentage']}%)")
    print(f"   Matching Skills ({len(r['matching_skills'])}): {', '.join(r['matching_skills'][:6])}")
    print(f"   Missing Skills ({len(r['missing_skills'])}): {', '.join(r['missing_skills'][:6])}")
    print(f"   Education: {r['education'][:80]}...")
    print(f"   Experience: {r['experience'][:80]}...")
