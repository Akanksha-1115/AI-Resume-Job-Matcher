"""
================================================================================
AI-Based Resume Screening and Job Recommendation System Using NLP
File: utils.py
Description: Utility functions for text cleaning, PDF extraction, skill matching,
             and recommendation computation.
Author: 3rd-Year BTech CSE Student
================================================================================
"""

import re
import os
import io
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


# ----------------------------------------------------------------------
# 1. TEXT CLEANING WITH TECHNICAL TERM PRESERVATION
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans freeform text while strictly preserving essential technical acronyms,
    programming languages, and frameworks (C++, C#, .NET, Node.js, SQL, etc.).
    """
    if not isinstance(text, str):
        return ""
    
    stripped_lower = text.strip().lower()
    if stripped_lower in ["not available", "nan", "none", "null", "n/a", ""]:
        return ""
    
    # 1. Preserve technical terms with symbols before punctuation stripping
    text = re.sub(r'\bC\+\+', 'cpp cplusplus', text, flags=re.I)
    text = re.sub(r'\bC#', 'csharp', text, flags=re.I)
    text = re.sub(r'(?:\.NET|\bDOTNET\b)', 'dotnet', text, flags=re.I)
    text = re.sub(r'\bNode\.js\b', 'nodejs', text, flags=re.I)
    text = re.sub(r'\bVue\.js\b', 'vuejs', text, flags=re.I)
    text = re.sub(r'\bReact\.js\b', 'reactjs', text, flags=re.I)
    text = re.sub(r'\bAngular\.js\b', 'angularjs', text, flags=re.I)
    text = re.sub(r'\bNext\.js\b', 'nextjs', text, flags=re.I)
    text = re.sub(r'\bCI\/CD\b', 'cicd', text, flags=re.I)
    text = re.sub(r'\bTCP\/IP\b', 'tcpip', text, flags=re.I)
    text = re.sub(r'\bPower\s*BI\b', 'powerbi', text, flags=re.I)
    text = re.sub(r'\bC\s*\/\s*C\+\+', 'c_language cpp cplusplus', text, flags=re.I)
    text = re.sub(r'\bC\s+programming\b', 'c_language programming', text, flags=re.I)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Strip URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # 4. Remove punctuation, preserving alphanumeric, underscores, spaces
    text = re.sub(r'[^a-z0-9_]', ' ', text)
    
    # 5. Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------------------------------------------------
# 2. PDF TEXT EXTRACTION WITH ERROR HANDLING
# ----------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    
    Parameters:
        uploaded_file: Streamlit UploadedFile, file-like object, bytes, or file path.
        
    Returns:
        (extracted_text, error_message):
        - If successful, extracted_text is a string and error_message is None.
        - If error, extracted_text is None and error_message describes the issue.
    """
    if pymupdf is None:
        return None, (
            "PyMuPDF library is not installed in the current environment. "
            "Please install it via: pip install PyMuPDF"
        )
        
    if uploaded_file is None:
        return None, "No file was uploaded."

    try:
        # Determine whether input is file path, bytes, or buffer
        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                return None, f"File not found: {uploaded_file}"
            doc = pymupdf.open(uploaded_file)
        elif isinstance(uploaded_file, (bytes, bytearray)):
            if len(uploaded_file) == 0:
                return None, "The uploaded file is empty (0 bytes)."
            doc = pymupdf.open(stream=uploaded_file, filetype="pdf")
        elif hasattr(uploaded_file, "read"):
            # File-like object (e.g. Streamlit UploadedFile)
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            if len(content) == 0:
                return None, "The uploaded file is empty (0 bytes)."
            doc = pymupdf.open(stream=content, filetype="pdf")
        else:
            return None, "Unsupported file input type."

        # Check page count
        if len(doc) == 0:
            doc.close()
            return None, "The PDF document contains 0 pages."

        extracted_text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                extracted_text_parts.append(text)
        doc.close()

        full_text = "\n".join(extracted_text_parts).strip()
        
        # Check for scanned / image-only PDFs
        if not full_text:
            return None, (
                "No selectable text could be extracted from this PDF. "
                "It appears to be a scanned image or empty document. "
                "Please upload a text-based PDF or paste your resume text."
            )

        return full_text, None

    except getattr(pymupdf, "FileDataError", Exception) as e:
        return None, f"Corrupted or invalid PDF file format: {str(e)}"
    except Exception as e:
        return None, f"An unexpected error occurred while reading the PDF: {str(e)}"


# ----------------------------------------------------------------------
# 3. EXPLICIT SKILL EXTRACTION & OVERLAP MATCHING
# ----------------------------------------------------------------------
def parse_job_skills(tech_skills_str: Any, req_skills_str: Any) -> List[str]:
    """
    Extracts individual skills explicitly listed in the job record.
    Prioritizes technology skills (tools, programming languages, software),
    followed by foundational worker skills.
    
    IMPORTANT: Strictly uses only explicit dataset entries and never invents skills.
    """
    skills = []
    
    # 1. Parse Technology Skills (semicolon-separated)
    if isinstance(tech_skills_str, str) and tech_skills_str.strip().lower() not in ["not available", "nan", "none", ""]:
        for s in tech_skills_str.split(";"):
            clean_s = s.strip()
            if clean_s and clean_s not in skills:
                skills.append(clean_s)
                
    # 2. Parse Required Skills (foundational skills with importance ratings)
    if isinstance(req_skills_str, str) and req_skills_str.strip().lower() not in ["not available", "nan", "none", ""]:
        for s in req_skills_str.split(";"):
            # Strip '(Importance: X.XX)' to get base skill name
            clean_s = re.sub(r'\(Importance:.*?\)', '', s).strip()
            if clean_s and clean_s not in skills:
                skills.append(clean_s)
                
    return skills


def calculate_skill_overlap(
    job_skills: List[str],
    resume_text: str
) -> Tuple[List[str], List[str]]:
    """
    Compares explicit job skills against candidate resume text.
    
    Rules:
    - Matching Skills: explicit job skills present in the resume.
    - Potential Missing Skills: explicit job skills NOT found in the resume.
    - Strictly bounded to job_skills. Never invents skills.
    """
    if not job_skills or not resume_text:
        return [], job_skills

    resume_lower = resume_text.lower()
    # Padded string for word boundary matching
    resume_norm = " " + re.sub(r'[^a-z0-9+#.]', ' ', resume_lower) + " "

    matching_skills = []
    missing_skills = []

    # Stopwords to avoid matching generic single words in compound skill names
    generic_words = {
        "software", "and", "the", "for", "system", "systems", "language",
        "program", "programming", "applications", "application", "development",
        "tools", "tool", "skills", "suite", "technologies", "technology"
    }

    for skill in job_skills:
        skill_lower = skill.lower()
        found = False

        # 1. Exact or direct phrase match
        if skill_lower in resume_lower:
            found = True
        else:
            # 2. Extract distinctive tokens (e.g., 'Python' in 'Python', 'SQL' in 'Structured query language SQL')
            tokens = [
                tok for tok in re.split(r'[\s/(),]+', skill_lower)
                if len(tok) >= 2 and tok not in generic_words
            ]
            
            # Check for technical aliases:
            # C++ -> cpp, c# -> csharp, aws -> amazon web services
            for tok in tokens:
                if tok == "c++" and (" c++ " in resume_norm or " cpp " in resume_norm or " cplusplus " in resume_norm):
                    found = True
                    break
                elif tok == "c#" and (" c# " in resume_norm or " csharp " in resume_norm):
                    found = True
                    break
                elif tok == "aws" and (" aws " in resume_norm or "amazon web services" in resume_lower):
                    found = True
                    break
                elif f" {tok} " in resume_norm and len(tok) >= 3:
                    found = True
                    break

        if found:
            matching_skills.append(skill)
        else:
            missing_skills.append(skill)

    return matching_skills, missing_skills


# ----------------------------------------------------------------------
# 4. RECOMMENDATION COMPUTATION ENGINE
# ----------------------------------------------------------------------
def compute_job_recommendations(
    raw_resume_text: str,
    vectorizer,
    job_vectors,
    df_jobs: pd.DataFrame,
    top_n: int = 5
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Computes top N job recommendations for a resume using TF-IDF and Cosine Similarity.
    
    Returns:
        (recommendations_list, cleaned_resume_text)
    """
    cleaned_resume = clean_text(raw_resume_text)
    if not cleaned_resume:
        return [], ""

    # Vectorize candidate resume
    resume_vector = vectorizer.transform([cleaned_resume])

    # Compute Cosine Similarity against all job vectors (1,016 jobs)
    cosine_sims = cosine_similarity(resume_vector, job_vectors).flatten()

    # Sort descending
    top_indices = np.argsort(cosine_sims)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        job_row = df_jobs.iloc[idx]
        raw_score = float(cosine_sims[idx])

        # Extract explicit job skills
        job_skills = parse_job_skills(
            job_row.get('technology_skills', ''),
            job_row.get('required_skills', '')
        )

        # Match skills against resume
        matching, missing = calculate_skill_overlap(job_skills, raw_resume_text)

        results.append({
            "rank": rank,
            "job_id": job_row.get('job_id', f'JOB-{idx}'),
            "job_title": job_row.get('job_title', 'Not Available'),
            "occupation_code": job_row.get('occupation_code', 'Not Available'),
            "similarity_score": raw_score,
            "similarity_percentage": round(raw_score * 100, 2),
            "job_description": job_row.get('job_description', 'Not Available'),
            "required_skills": job_row.get('required_skills', 'Not Available'),
            "technology_skills": job_row.get('technology_skills', 'Not Available'),
            "education": job_row.get('education', 'Not Available'),
            "experience": job_row.get('experience', 'Not Available'),
            "matching_skills": matching,
            "missing_skills": missing,
            "source_dataset": job_row.get('source_dataset', 'O*NET 31.0 Database'),
            "source_url": job_row.get('source_url', 'https://www.onetcenter.org/database.html'),
            "license": job_row.get('license', 'CC BY 4.0')
        })

    return results, cleaned_resume
