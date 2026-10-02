"""
AI Resume Screening & Job Recommendation System
File: app.py
Description: Streamlit entrypoint for local execution and Streamlit Community Cloud deployment.
             Dynamic path resolution ensures it runs both locally and on Streamlit Cloud.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Ensure project root is in sys.path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "model")) and os.path.exists(os.path.join(current_dir, "data")):
    project_root = current_dir
elif os.path.exists(os.path.join(os.path.dirname(current_dir), "model")) and os.path.exists(os.path.join(os.path.dirname(current_dir), "data")):
    project_root = os.path.dirname(current_dir)
else:
    project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import (
    clean_text,
    extract_text_from_pdf,
    compute_job_recommendations,
    parse_job_skills,
    calculate_skill_overlap
)

# ----------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# SIMPLE, CLEAN, COLLEGE-PROJECT CSS (NO EMOJIS, NO GRADIENTS, NO GLASSERY)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Font and base layout */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1f2937;
        font-size: 15px;
    }

    /* Simple Header */
    .header-box {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 26px;
        font-weight: 600;
        color: #111827;
        margin: 0 0 6px 0;
    }
    .header-subtitle {
        font-size: 15px;
        color: #4b5563;
        margin: 0;
    }

    /* Section Containers */
    .section-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    .section-title {
        font-size: 19px;
        font-weight: 600;
        color: #111827;
        margin: 0 0 12px 0;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 6px;
    }

    /* Status indicator */
    .status-text {
        font-size: 14px;
        color: #047857;
        font-weight: 500;
        margin-top: 4px;
    }
    .file-name-text {
        font-size: 14px;
        color: #1f2937;
        font-weight: 600;
    }

    /* Skill Tags */
    .tag-matching {
        display: inline-block;
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
        border-radius: 3px;
        padding: 3px 8px;
        margin: 2px 4px 2px 0;
        font-size: 13px;
    }
    .tag-missing {
        display: inline-block;
        background-color: #fef7e0;
        color: #b06000;
        border: 1px solid #feefc3;
        border-radius: 3px;
        padding: 3px 8px;
        margin: 2px 4px 2px 0;
        font-size: 13px;
    }
    .tag-general {
        display: inline-block;
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-radius: 3px;
        padding: 3px 8px;
        margin: 2px 4px 2px 0;
        font-size: 13px;
    }

    /* Score display */
    .score-badge {
        font-size: 15px;
        font-weight: 600;
        color: #1d4ed8;
    }

    /* Footer */
    .footer-box {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 36px;
        padding-top: 14px;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED DATA & MODEL LOADER
# ----------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    vec_path = os.path.join(project_root, "model", "tfidf_vectorizer.joblib")
    vectors_path = os.path.join(project_root, "model", "job_vectors.joblib")
    jobs_csv_path = os.path.join(project_root, "data", "clean_jobs.csv")
    resumes_csv_path = os.path.join(project_root, "data", "clean_resumes.csv")

    errors = []
    if not os.path.exists(vec_path):
        errors.append(f"Model file missing: '{vec_path}'. Please run 'python train_model.py'.")
    if not os.path.exists(vectors_path):
        errors.append(f"Job vectors missing: '{vectors_path}'. Please run 'python train_model.py'.")
    if not os.path.exists(jobs_csv_path):
        errors.append(f"Clean jobs data missing: '{jobs_csv_path}'. Please run 'python preprocessing.py'.")

    if errors:
        return None, None, None, None, errors

    try:
        vectorizer = joblib.load(vec_path)
        job_vectors = joblib.load(vectors_path)
        df_jobs = pd.read_csv(jobs_csv_path)
        df_resumes = pd.read_csv(resumes_csv_path) if os.path.exists(resumes_csv_path) else None
        return vectorizer, job_vectors, df_jobs, df_resumes, []
    except Exception as e:
        return None, None, None, None, [f"Error loading files: {str(e)}"]


vectorizer, job_vectors, df_jobs, df_resumes, load_errors = load_project_artifacts()


# ----------------------------------------------------------------------
# HELPER: EXTRACT RESUME OVERVIEW (SKILLS, EDUCATION, EXPERIENCE)
# ----------------------------------------------------------------------
def extract_resume_details(raw_text: str, vectorizer):
    """
    Extracts summary information (Skills, Education, Experience) directly
    from candidate resume text.
    """
    if not raw_text:
        return [], "Not Available", "Not Available"

    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    text_lower = raw_text.lower()

    # 1. Extract Education
    edu_found = []
    edu_patterns = [
        r'(b\.?tech[^\n,]*)',
        r'(bachelor[^\n,]*)',
        r'(m\.?tech[^\n,]*)',
        r'(master[^\n,]*)',
        r'(b\.?e\.?[^\n,]*)',
        r'(b\.?sc[^\n,]*)',
        r'(diploma[^\n,]*)',
        r'(ph\.?d[^\n,]*)',
        r'(degree[^\n,]*)'
    ]
    for pat in edu_patterns:
        matches = re.findall(pat, text_lower, flags=re.I)
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 4 and clean_m.title() not in edu_found:
                edu_found.append(clean_m.title())

    # Fallback to search lines with 'education'
    if not edu_found:
        for idx, line in enumerate(lines):
            if 'education' in line.lower() and idx + 1 < len(lines):
                edu_found.append(lines[idx + 1])
                break

    edu_str = "; ".join(edu_found[:2]) if edu_found else "Not explicitly specified in text"

    # 2. Extract Experience
    exp_found = []
    exp_patterns = [
        r'(\d+\+?\s+years?(?:\s+of)?\s+experience[^\n,]*)',
        r'(intern(?:ship)?[^\n,]*)',
        r'((?:software|data|web|ml|developer|engineer|analyst|designer)\s+(?:intern|specialist|lead|architect)[^\n,]*)'
    ]
    for pat in exp_patterns:
        matches = re.findall(pat, text_lower, flags=re.I)
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 4 and clean_m.title() not in exp_found:
                exp_found.append(clean_m.title())

    if not exp_found:
        for idx, line in enumerate(lines):
            if 'experience' in line.lower() and idx + 1 < len(lines):
                exp_found.append(lines[idx + 1])
                break

    exp_str = "; ".join(exp_found[:2]) if exp_found else "Not explicitly specified in text"

    # 3. Extract Identified Skills
    # Match vocabulary unigrams from vectorizer that appear in the resume
    identified_skills = []
    common_tech_keywords = [
        "python", "sql", "c++", "c#", "java", "r", "javascript", "typescript",
        "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
        "fastapi", "spring", "pandas", "numpy", "scikit-learn", "tensorflow",
        "keras", "pytorch", "nlp", "machine learning", "deep learning", "docker",
        "kubernetes", "git", "aws", "azure", "gcp", "linux", "tableau", "power bi",
        "excel", "mongodb", "postgresql", "mysql", "redis", "spark", "hadoop"
    ]
    norm_text = " " + re.sub(r'[^a-z0-9+#.]', ' ', text_lower) + " "

    for kw in common_tech_keywords:
        kw_clean = kw.lower()
        if f" {kw_clean} " in norm_text or (kw_clean == "c++" and " c++ " in norm_text) or (kw_clean == "c#" and " c# " in norm_text):
            identified_skills.append(kw.upper() if len(kw) <= 4 else kw.title())

    return identified_skills, edu_str, exp_str


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.subheader("AI Resume Matcher")
    st.write("A machine learning project comparing resume text with verified job requirements using TF-IDF and Cosine Similarity.")
    
    st.markdown("---")
    st.markdown("**Navigation**")
    nav_selection = st.radio(
        "Go to section:",
        ["Home", "Resume Analysis", "Recommended Jobs", "Job Comparison", "About Project"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Settings**")
    top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5, step=1)
    
    score_mode = st.selectbox(
        "Score calculation:",
        ["Match Score (%)", "Raw Text Similarity Score"]
    )

    st.markdown("---")
    # Optional sample resume testing placed cleanly in sidebar per prompt instructions
    with st.expander("Test with sample resume (optional)"):
        sample_choice = st.selectbox(
            "Load sample:",
            [
                "None",
                "Sample 1: Software Engineer / ML Student",
                "Sample 2: 2D/3D Animator & Artist",
                "Sample 3: Full-Stack Developer",
                "Sample 4: Data Analyst"
            ]
        )

    st.markdown("---")
    st.markdown("**About Project**")
    st.caption(
        "This project uses NLP and machine learning techniques to compare resume "
        "information with job information and recommend potentially relevant roles.\n\n"
        "The recommendations are based on text similarity and available resume/job "
        "information. They should not be treated as a final hiring decision."
    )


# Sample resume text definitions
sample_texts = {
    "Sample 1: Software Engineer / ML Student": (
        "Rahul Verma | B.Tech Computer Science and Engineering (CSE)\n\n"
        "EDUCATION:\n"
        "Bachelor of Technology in Computer Science and Engineering\n"
        "Relevant Coursework: Data Structures, Algorithms, DBMS, Machine Learning, Deep Learning, NLP\n\n"
        "TECHNICAL SKILLS:\n"
        "Programming Languages: Python, SQL, C++, Java\n"
        "Frameworks & Libraries: Pandas, NumPy, Scikit-Learn, TensorFlow, FastAPI, Flask\n"
        "Tools & Cloud: Git, Docker, AWS, Linux, Postman\n"
        "Core Concepts: Natural Language Processing (NLP), TF-IDF, Vector Search, Supervised Learning\n\n"
        "PROJECTS:\n"
        "1. AI-Based Resume Screening and Job Recommendation System using TF-IDF and Cosine Similarity.\n"
        "2. Predictive Customer Analytics using Machine Learning models in Scikit-Learn with FastAPI deployment.\n\n"
        "EXPERIENCE:\n"
        "Machine Learning Intern (Summer 2025) - Built automated feature pipelines and evaluation benchmarks."
    ),
    "Sample 2: 2D/3D Animator & Artist": (
        "Ananya Sharma | 2D/3D Artist & Animator\n\n"
        "SUMMARY:\n"
        "Creative 2D Animator and Concept Artist with experience in character animation, spine 2D, "
        "3D modeling in 3ds Max, and digital illustration for mobile gaming and interactive media.\n\n"
        "TECHNICAL SKILLS:\n"
        "Animation & 3D: Spine2D, 3ds Max, Blender, Maya, Unity, Motion Design\n"
        "Illustration & Design: Adobe Photoshop, Adobe Illustrator, Procreate, Digital Painting\n"
        "Concepts: Storyboarding, Character Rigging, UI/UX Asset Design, Concept Art\n\n"
        "EXPERIENCE:\n"
        "Lead 2D Animator (2023 - Present) - Created 2D character sheets, background illustrations, and sprite sheets."
    ),
    "Sample 3: Full-Stack Developer": (
        "Alex Morgan | Senior Full-Stack Engineer\n\n"
        "SUMMARY:\n"
        "Full-Stack Developer with strong experience in building web applications and backend APIs.\n\n"
        "TECHNICAL SKILLS:\n"
        "Backend: C#, .NET Core, Node.js, Python, FastAPI\n"
        "Frontend: React, TypeScript, Next.js, HTML5, CSS3, JavaScript\n"
        "Databases: PostgreSQL, MySQL, Redis, MongoDB\n"
        "DevOps & Tools: Docker, Kubernetes, Git, AWS (EC2, S3), CI/CD pipelines\n\n"
        "EXPERIENCE:\n"
        "Software Engineer (2021 - Present) - Designed REST APIs and microservices."
    ),
    "Sample 4: Data Analyst": (
        "Amit Patel | Data Analyst\n\n"
        "SUMMARY:\n"
        "Data Analyst with 3 years of experience in business reporting, statistical analysis, and dashboard development.\n\n"
        "TECHNICAL SKILLS:\n"
        "Databases & Querying: SQL, PostgreSQL, MySQL, Microsoft SQL Server, Oracle Database\n"
        "BI & Visualization: Power BI, Tableau, Microsoft Excel (VLOOKUP, Pivot Tables, Macros)\n"
        "Programming: Python (Pandas, NumPy, Matplotlib, Seaborn), R\n"
        "Analytics: KPI Dashboards, A/B Testing, Cohort Analysis, ETL Pipelines\n\n"
        "EXPERIENCE:\n"
        "Data Analyst (2022 - Present) - Automated KPI reporting."
    )
}


# ----------------------------------------------------------------------
# TOP: SIMPLE HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">AI Resume Screening & Job Recommendation System</div>
    <div class="header-subtitle">Upload your resume to analyze your skills and find matching job roles.</div>
</div>
""", unsafe_allow_html=True)

if load_errors:
    st.error("System assets could not be loaded:")
    for err in load_errors:
        st.write(f"- {err}")
    st.stop()


# ----------------------------------------------------------------------
# SECTION: ABOUT PROJECT (IF NAVIGATED VIA SIDEBAR)
# ----------------------------------------------------------------------
if nav_selection == "About Project":
    st.markdown("### About Project")
    st.write(
        "This project uses NLP and machine learning techniques to compare resume "
        "information with job information and recommend potentially relevant roles."
    )
    st.write(
        "The recommendations are based on text similarity and available resume/job "
        "information. They should not be treated as a final hiring decision."
    )
    st.markdown("---")
    st.markdown("**Dataset Information:**")
    st.write("- Resumes Dataset: 600 verified real resume records (Djinni Recruitment Corpus).")
    st.write("- Jobs Dataset: 1,016 verified occupation records (O*NET 31.0 Database).")
    st.write("- Model: TF-IDF Vectorizer (ngram_range=(1,2), max_features=30,000) and Cosine Similarity.")
    st.stop()


# ----------------------------------------------------------------------
# 1. RESUME UPLOAD SECTION
# ----------------------------------------------------------------------
st.markdown("### Resume Upload")

upload_mode = st.radio("Input method:", ["Upload PDF Resume", "Paste Resume Text"], horizontal=True, label_visibility="collapsed")

resume_text = ""
uploaded_file_name = None

col_upload, col_status = st.columns([3, 2])

with col_upload:
    if upload_mode == "Upload PDF Resume":
        uploaded_pdf = st.file_uploader("Choose PDF Resume", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            uploaded_file_name = uploaded_pdf.name
            with st.spinner("Extracting text from PDF..."):
                extracted_text, extract_err = extract_text_from_pdf(uploaded_pdf)

            if extract_err:
                st.error(extract_err)
            else:
                resume_text = extracted_text
        elif sample_choice != "None":
            uploaded_file_name = f"{sample_choice.split(':')[0].strip().lower()}.txt"
            resume_text = sample_texts[sample_choice]
    else:
        # Paste Resume Text
        initial_val = sample_texts[sample_choice] if sample_choice != "None" else ""
        pasted_text = st.text_area("Paste Resume Text", value=initial_val, height=140, placeholder="Paste resume text here...", label_visibility="collapsed")
        if pasted_text.strip():
            uploaded_file_name = "pasted_resume.txt"
            resume_text = pasted_text.strip()

with col_status:
    if resume_text:
        st.markdown(f"<div class='file-name-text'>File: {uploaded_file_name or 'resume.pdf'}</div>", unsafe_allow_html=True)
        st.markdown("<div class='status-text'>Status: Resume loaded successfully</div>", unsafe_allow_html=True)
        st.write(f"Length: {len(resume_text.split())} words ({len(resume_text)} characters)")

# Expandable extracted text section
if resume_text:
    with st.expander("View extracted text"):
        st.text_area("Raw Extracted Content", resume_text, height=140, disabled=True, label_visibility="collapsed")

# Simple Analyze Button
analyze_clicked = st.button("Analyze Resume", type="primary")

# Persist analysis state in session_state
if analyze_clicked and resume_text.strip():
    with st.spinner("Processing resume text and computing job matches..."):
        recs, cleaned_res = compute_job_recommendations(
            raw_resume_text=resume_text,
            vectorizer=vectorizer,
            job_vectors=job_vectors,
            df_jobs=df_jobs,
            top_n=top_n
        )
        st.session_state["recs"] = recs
        st.session_state["resume_text"] = resume_text

elif analyze_clicked and not resume_text.strip():
    st.warning("Please upload a PDF resume or enter resume text to analyze.")


# ----------------------------------------------------------------------
# RESULTS DISPLAY (WHEN RESUME HAS BEEN ANALYZED)
# ----------------------------------------------------------------------
if "recs" in st.session_state and st.session_state["recs"]:
    recs = st.session_state["recs"]
    curr_text = st.session_state.get("resume_text", "")

    calibration_factor = 7.5

    # ------------------------------------------------------------------
    # 2. RESUME ANALYSIS SECTION
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Resume Analysis")
    
    cand_skills, cand_edu, cand_exp = extract_resume_details(curr_text, vectorizer)

    col_res_a, col_res_b = st.columns(2)
    with col_res_a:
        st.markdown("**Education:**")
        st.write(cand_edu)
        st.markdown("**Experience:**")
        st.write(cand_exp)

    with col_res_b:
        st.markdown("**Identified Skills:**")
        if cand_skills:
            tags_html = "".join([f"<span class='tag-general'>{s}</span>" for s in cand_skills])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("No standard technical keywords identified directly.")

    # ------------------------------------------------------------------
    # 3. RECOMMENDED JOBS TABLE
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Recommended Jobs")

    # Prepare clean table data: Rank | Job Role | Match Score | Matching Skills
    table_data = []
    for r in recs:
        raw_val = r['similarity_percentage']
        if score_mode == "Match Score (%)":
            score_disp = f"{min(round(raw_val * calibration_factor, 1), 99.0)}%"
        else:
            score_disp = f"{raw_val:.2f}%"

        matching_sample = ", ".join(r['matching_skills'][:5]) if r['matching_skills'] else "None"
        if len(r['matching_skills']) > 5:
            matching_sample += f" (+{len(r['matching_skills']) - 5} more)"

        table_data.append({
            "Rank": r['rank'],
            "Job Role": r['job_title'],
            "Match Score": score_disp,
            "Matching Skills": matching_sample
        })

    df_display = pd.DataFrame(table_data)
    st.table(df_display)

    # ------------------------------------------------------------------
    # 4 & 5. JOB DETAILS & WHY THIS JOB WAS RECOMMENDED
    # ------------------------------------------------------------------
    st.markdown("### Job Details")

    job_options = [f"Rank {r['rank']}: {r['job_title']}" for r in recs]
    selected_job_label = st.selectbox("Select a job to view details:", job_options, index=0)

    selected_idx = int(selected_job_label.split(":")[0].replace("Rank", "").strip()) - 1
    selected_job = recs[selected_idx]

    raw_s = selected_job['similarity_percentage']
    score_text = f"{min(round(raw_s * calibration_factor, 1), 99.0)}%" if score_mode == "Match Score (%)" else f"{raw_s:.2f}%"

    col_jd1, col_jd2 = st.columns([3, 2])

    with col_jd1:
        st.markdown(f"**Job Role:** {selected_job['job_title']}")
        st.markdown(f"**Match Score:** <span class='score-badge'>{score_text}</span> (Text Similarity Score)", unsafe_allow_html=True)
        st.markdown(f"**Job Description:**\n{selected_job['job_description']}")

    with col_jd2:
        st.markdown(f"**Education Requirement:**\n{selected_job['education']}")
        st.markdown(f"**Experience Requirement:**\n{selected_job['experience']}")

    # 5. Why This Job Was Recommended
    st.markdown("#### Why This Job Was Recommended")
    st.caption("The score is a text similarity score based on resume and job vocabulary overlap, not an absolute hiring decision.")

    col_why1, col_why2 = st.columns(2)

    with col_why1:
        st.markdown("**Matching Skills:**")
        if selected_job['matching_skills']:
            for s in selected_job['matching_skills'][:15]:
                st.write(f"- {s}")
            if len(selected_job['matching_skills']) > 15:
                st.caption(f"...and {len(selected_job['matching_skills']) - 15} more matching skills.")
        else:
            st.write("No explicit technical skills directly matched.")

    with col_why2:
        st.markdown("**Potential Missing Skills:**")
        if selected_job['missing_skills']:
            for s in selected_job['missing_skills'][:15]:
                st.write(f"- {s}")
            if len(selected_job['missing_skills']) > 15:
                st.caption(f"...and {len(selected_job['missing_skills']) - 15} more skills listed in job profile.")
        else:
            st.write("All listed skills were identified in the resume.")

    # ------------------------------------------------------------------
    # 6. JOB COMPARISON SECTION
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Job Comparison")
    st.caption("Compare two recommended jobs side by side:")

    all_roles = [r['job_title'] for r in recs]
    col_cmp_a, col_cmp_b = st.columns(2)

    with col_cmp_a:
        job_a_title = st.selectbox("Select Job 1:", all_roles, index=0)
    with col_cmp_b:
        job_b_title = st.selectbox("Select Job 2:", all_roles, index=min(1, len(all_roles) - 1))

    job_a_data = next((item for item in recs if item['job_title'] == job_a_title), recs[0])
    job_b_data = next((item for item in recs if item['job_title'] == job_b_title), recs[1] if len(recs) > 1 else recs[0])

    def format_score(raw):
        return f"{min(round(raw * calibration_factor, 1), 99.0)}%" if score_mode == "Match Score (%)" else f"{raw:.2f}%"

    comparison_rows = [
        {"Attribute": "Job Role", "Job 1": job_a_data['job_title'], "Job 2": job_b_data['job_title']},
        {"Attribute": "Text Similarity Score", "Job 1": format_score(job_a_data['similarity_percentage']), "Job 2": format_score(job_b_data['similarity_percentage'])},
        {"Attribute": "Matching Skills Count", "Job 1": f"{len(job_a_data['matching_skills'])} skills", "Job 2": f"{len(job_b_data['matching_skills'])} skills"},
        {"Attribute": "Sample Matching Skills", "Job 1": ", ".join(job_a_data['matching_skills'][:4]) or "None", "Job 2": ", ".join(job_b_data['matching_skills'][:4]) or "None"},
        {"Attribute": "Potential Missing Skills Count", "Job 1": f"{len(job_a_data['missing_skills'])} skills", "Job 2": f"{len(job_b_data['missing_skills'])} skills"},
        {"Attribute": "Sample Missing Skills", "Job 1": ", ".join(job_a_data['missing_skills'][:4]) or "None", "Job 2": ", ".join(job_b_data['missing_skills'][:4]) or "None"},
        {"Attribute": "Education", "Job 1": str(job_a_data['education'])[:80] + "...", "Job 2": str(job_b_data['education'])[:80] + "..."},
        {"Attribute": "Experience", "Job 1": str(job_a_data['experience'])[:80] + "...", "Job 2": str(job_b_data['experience'])[:80] + "..."}
    ]

    st.table(pd.DataFrame(comparison_rows).set_index("Attribute"))

    # ------------------------------------------------------------------
    # 7. CAREER/SKILL SUGGESTIONS SECTION
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Skills to Consider Learning")
    st.caption("Based on potential missing skills identified across your top recommended job roles:")

    # Count occurrences of missing skills across all top recommended jobs
    missing_freq = {}
    for r in recs:
        for sk in r['missing_skills']:
            missing_freq[sk] = missing_freq.get(sk, 0) + 1

    # Sort descending by frequency
    sorted_missing = sorted(missing_freq.items(), key=lambda x: x[1], reverse=True)

    if sorted_missing:
        st.write("The following skills are frequently required by your top matching roles but were not identified in your resume:")
        col_sugg1, col_sugg2 = st.columns(2)
        half = min(6, len(sorted_missing))
        
        with col_sugg1:
            for skill_name, freq in sorted_missing[:half]:
                st.write(f"- **{skill_name}** (appears in {freq} of your top {len(recs)} matches)")

        with col_sugg2:
            for skill_name, freq in sorted_missing[half:half*2]:
                st.write(f"- **{skill_name}** (appears in {freq} of your top {len(recs)} matches)")
    else:
        st.write("No missing skills identified among your top recommended roles.")


# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("""
<div class="footer-box">
    AI-Based Resume Screening and Job Recommendation System | College ML Project
</div>
""", unsafe_allow_html=True)
