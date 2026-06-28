import streamlit as st
import json
import gzip
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_core import is_honeypot, evaluate_candidates, generate_custom_reasoning, SERVICE_COMPANIES

print("\n=== STARTING SCRIPT RERUN ===")
t_total_start = time.time()

# Page Configuration
st.set_page_config(
    page_title="Redrob Talent Discovery Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Mode, Glassmorphism, Custom Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1f122e 0%, #0d0614 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid #331f4d;
        box-shadow: 0 8px 32px 0 rgba(13, 6, 20, 0.37);
        position: relative;
        overflow: hidden;
    }
    
    .header-title {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #ff7bf2 0%, #9e5fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        color: #b0a3c4;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 20, 45, 0.4);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(130, 80, 220, 0.15);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(130, 80, 220, 0.4);
    }
    
    .metric-value {
        font-weight: 800;
        font-size: 2rem;
        color: #ff7bf2;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b0a3c4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Candidate Profile Cards */
    .profile-card {
        background: rgba(25, 15, 38, 0.5);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(158, 95, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
    }
    
    .profile-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    
    .profile-title {
        font-size: 1.2rem;
        color: #9e5fff;
        font-weight: 600;
    }
    
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-core {
        background: rgba(158, 95, 255, 0.2);
        color: #d1b3ff;
        border: 1px solid rgba(158, 95, 255, 0.4);
    }
    
    .badge-location {
        background: rgba(255, 123, 242, 0.2);
        color: #ffb3f7;
        border: 1px solid rgba(255, 123, 242, 0.4);
    }
    
    /* Reasoning Quote Box */
    .reasoning-box {
        background: rgba(130, 80, 220, 0.08);
        border-left: 4px solid #ff7bf2;
        padding: 1.2rem;
        border-radius: 0 10px 10px 0;
        font-style: italic;
        color: #e3d9ff;
        margin-top: 1rem;
    }
    
    /* Score Radar Block */
    .score-breakdown-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .score-label {
        color: #b0a3c4;
    }
    
    .score-val {
        font-weight: 600;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Application Heading
st.markdown("""
<div class="header-container">
    <div class="header-title">REDROB INTEL-RANKER</div>
    <div class="header-subtitle">Founding Senior AI Engineer Candidate Discovery Portal</div>
</div>
""", unsafe_allow_html=True)

# Cache Candidate Data
@st.cache_resource
def load_candidates_cache(filepath):
    candidates = []
    open_func = gzip.open if filepath.endswith(".gz") else open
    with open_func(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
    return candidates

# Load data
data_path = "./candidates.jsonl"
if not os.path.exists(data_path):
    data_path = "./candidates.jsonl.gz"
    
if not os.path.exists(data_path):
    # Try sample candidates if full file not found (fallback for local spaces testing)
    data_path = "./sample_candidates.json"

if not os.path.exists(data_path):
    st.error("No candidate dataset found in current directory! Please ensure candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json exists.")
    st.stop()

t_start = time.time()
with st.spinner("Loading candidate database..."):
    all_candidates = load_candidates_cache(data_path)
print(f"--- app.py: load_candidates_cache returned in {time.time() - t_start:.4f}s")

# Extract technical profiles text for semantic TF-IDF
@st.cache_resource
def get_tfidf_matrices(_candidates_list):
    profile_texts = []
    for c in _candidates_list:
        prof = c.get("profile", {})
        skills_str = " ".join([s.get("name", "") for s in c.get("skills", [])])
        career_str = " ".join([job.get("title", "") + " " + job.get("description", "") for job in c.get("career_history", [])])
        full_text = f"{prof.get('headline', '')} {prof.get('summary', '')} {skills_str} {career_str}"
        profile_texts.append(full_text)
        
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(profile_texts)
    return vectorizer, vectors

t_tfidf = time.time()
vectorizer, candidate_vectors = get_tfidf_matrices(all_candidates)
print(f"--- app.py: get_tfidf_matrices returned in {time.time() - t_tfidf:.4f}s")

# Fixed parameters for candidate ranking
w_yoe = 1.0
w_loc = 1.0
w_title = 1.0
w_skill = 1.0
w_sem = 1.0
w_beh = 1.0

jd_editor = """Senior AI Engineer — Founding Team. Embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5), vector databases (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS), Python, evaluation frameworks (NDCG, MRR, MAP, A/B testing), nice to have fine-tuning LLMs (LoRA, QLoRA, PEFT)."""

# Re-evaluate based on weights
@st.cache_data(show_spinner=False)
def get_custom_scores(jd_text, _all_candidates, _vectorizer, _candidate_vectors, w_yoe, w_loc, w_title, w_skill, w_sem, w_beh):
    # Compute semantic cosine similarity
    jd_vector = _vectorizer.transform([jd_text])
    similarities = cosine_similarity(_candidate_vectors, jd_vector).flatten()
    
    SERVICE_COMPANIES = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree', 'mphasis'}
    AI_TITLES = ["ai engineer", "ml engineer", "machine learning engineer", "nlp engineer", "search engineer", "applied ml engineer", "recommendation systems engineer", "senior ai engineer", "staff machine learning engineer", "senior applied scientist", "senior nlp engineer", "lead ai engineer", "senior machine learning engineer", "senior ml engineer", "data scientist", "senior data scientist"]
    
    CORE_SKILL_GROUPS = {
        "embeddings": ["embedding", "sentence-transformer", "sentence transformer", "bge", "e5", "word2vec"],
        "vector_dbs": ["vector db", "vector search", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "pgvector", "chroma"],
        "retrieval": ["rag", "retrieval", "retrieval-augmented generation", "dense retrieval", "sparse retrieval", "bm25", "information retrieval"],
        "ranking": ["ranking", "ndcg", "mrr", "map", "learning to rank", "learning-to-rank", "ltr", "re-ranking", "reranking"],
        "python": ["python", "pytorch", "tensorflow", "scikit-learn"]
    }
    
    NICE_SKILL_GROUPS = {
        "llm_tuning": ["lora", "qlora", "peft", "fine-tuning", "fine-tune", "llm", "large language model"],
        "mlops": ["mlflow", "kubeflow", "dbt", "airflow", "docker", "gcp", "aws", "azure"]
    }
    
    results = []
    
    for idx, c in enumerate(_all_candidates):
        cid = c.get("candidate_id")
        
        # Blacklist
        if is_honeypot(c):
            continue
            
        profile = c.get("profile", {})
        career = c.get("career_history", [])
        skills = c.get("skills", [])
        signals = c.get("redrob_signals", {})
        
        current_title = profile.get("current_title", "").lower()
        headline = profile.get("headline", "").lower()
        summary = profile.get("summary", "").lower()
        
        # Heuristic filters
        all_companies = {job.get("company", "").lower() for job in career if job.get("company")}
        if all_companies and all_companies.issubset(SERVICE_COMPANIES):
            continue
            
        non_tech_titles = {'marketing manager', 'accountant', 'hr manager', 'operations manager', 'graphic designer', 'civil engineer', 'customer support', 'sales executive', 'project manager'}
        if current_title in non_tech_titles:
            continue
            
        # YOE Score
        yoe = profile.get("years_of_experience", 0.0)
        if yoe < 4.0 or yoe > 13.0:
            yoe_score = 2.0
        elif 6.0 <= yoe <= 8.0:
            yoe_score = 10.0
        else:
            yoe_score = 7.5
            
        # Location Score
        loc = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        relocate = signals.get("willing_to_relocate", False)
        in_target_city = any(city in loc for city in ["noida", "pune", "gurgaon", "delhi", "ncr"])
        in_tier1_india = any(city in loc for city in ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai", "kolkata", "chandigarh", "ahmedabad", "coimbatore", "kochi", "trivandrum", "indore"]) or country == "india"
        
        if in_target_city:
            loc_score = 10.0
        elif in_tier1_india:
            loc_score = 8.0 if relocate else 6.0
        else:
            if relocate:
                loc_score = 1.0
            else:
                continue
                
        # Title score
        title_score = 2.0
        for t in AI_TITLES:
            if t in current_title or t in headline:
                title_score = 10.0
                break
        if title_score == 2.0:
            if any(t in current_title or t in headline for t in ["data engineer", "backend", "software engineer", "full stack"]):
                title_score = 5.0
                
        # NLP Skills score
        skills_names = [s.get("name", "").lower() for s in skills]
        skills_str = " ".join(skills_names) + " " + headline + " " + summary
        
        matched_cores = []
        for group, keywords in CORE_SKILL_GROUPS.items():
            if any(kw in skills_str for kw in keywords):
                matched_cores.append(group)
                
        matched_nices = []
        for group, keywords in NICE_SKILL_GROUPS.items():
            if any(kw in skills_str for kw in keywords):
                matched_nices.append(group)
                
        nlp_skill_score = (len(matched_cores) * 3.0) + (len(matched_nices) * 1.5)
        nlp_skill_score = min(nlp_skill_score, 20.0)
        
        # Semantic Score
        sem_score = similarities[idx] * 40.0
        
        # Behavioral score
        last_active = signals.get("last_active_date")
        active_score = 0.0
        if last_active:
            try:
                dt = datetime.strptime(last_active, "%Y-%m-%d")
                days_inactive = (datetime(2026, 6, 23) - dt).days
                if days_inactive <= 30:
                    active_score = 3.0
                elif days_inactive <= 90:
                    active_score = 2.0
                elif days_inactive <= 180:
                    active_score = 1.0
            except:
                pass
        resp_rate = signals.get("recruiter_response_rate", 0.0)
        resp_score = resp_rate * 3.0
        open_to_work = signals.get("open_to_work_flag", False)
        otw_score = 2.0 if open_to_work else 0.0
        github = signals.get("github_activity_score", -1)
        github_score = 0.0
        if github > 50:
            github_score = 2.0
        elif github > 0:
            github_score = 1.0
            
        behavioral_score = active_score + resp_score + otw_score + github_score
        
        # Combine with custom weights
        final_score = (
            w_yoe * yoe_score +
            w_loc * loc_score +
            w_title * title_score +
            w_skill * nlp_skill_score +
            w_sem * sem_score +
            w_beh * behavioral_score
        )
        
        c["match_details"] = {
            "yoe_score": round(yoe_score * w_yoe, 1),
            "loc_score": round(loc_score * w_loc, 1),
            "title_score": round(title_score * w_title, 1),
            "nlp_skill_score": round(nlp_skill_score * w_skill, 1),
            "sem_score": round(sem_score * w_sem, 1),
            "behavioral_score": round(behavioral_score * w_beh, 1),
            "matched_skills": [s.get("name") for s in skills if any(kw in s.get("name", "").lower() for keywords in list(CORE_SKILL_GROUPS.values()) + list(NICE_SKILL_GROUPS.values()) for kw in keywords)]
        }
        results.append((cid, final_score, c))
        
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:100]

# Perform Discovery
t_scores = time.time()
shortlist = get_custom_scores(jd_editor, all_candidates, vectorizer, candidate_vectors, w_yoe, w_loc, w_title, w_skill, w_sem, w_beh)
print(f"--- app.py: get_custom_scores returned in {time.time() - t_scores:.4f}s")

# Statistics Row
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(all_candidates):,}</div>
        <div class="metric-label">Total Scanned Candidates</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">108</div>
        <div class="metric-label">Filtered Honeypots</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">100</div>
        <div class="metric-label">Shortlisted Candidates</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid Layout
col_profile, col_table = st.columns([1.2, 0.8])

with col_table:
    st.markdown("### 🏆 Candidate Discovery Shortlist (Top 100)")
    
    # Construct dataframe for display
    table_data = []
    for rank_idx, (cid, score, c) in enumerate(shortlist):
        prof = c.get("profile", {})
        table_data.append({
            "Rank": rank_idx + 1,
            "ID": cid,
            "Name": prof.get("anonymized_name"),
            "Title": prof.get("current_title"),
            "YOE": prof.get("years_of_experience"),
            "Location": prof.get("location"),
            "Match Score": round(score, 2)
        })
    df = pd.DataFrame(table_data)
    
    # Display interactive selection box
    selected_idx = st.selectbox(
        "Select a candidate to inspect their profile:",
        options=df.index,
        format_func=lambda idx: f"Rank {df.loc[idx, 'Rank']}: {df.loc[idx, 'Name']} ({df.loc[idx, 'Title']} - {df.loc[idx, 'Match Score']} pts)"
    )
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# Selected candidate profile details
_, _, sel_cand = shortlist[selected_idx]
sel_prof = sel_cand.get("profile", {})
sel_sigs = sel_cand.get("redrob_signals", {})
sel_match = sel_cand.get("match_details", {})
sel_rank = selected_idx + 1
sel_score = df.loc[selected_idx, "Match Score"]

# Compute reasoning
sel_reason = generate_custom_reasoning(sel_cand, sel_rank, sel_score)

with col_profile:
    st.markdown("### 👤 Candidate Profiler")
    # Premium Profile Card UI
    st.markdown(f"""<div class="profile-card">
<div style="display: flex; justify-content: space-between; align-items: start;">
<div>
<div class="profile-name">{sel_prof.get('anonymized_name')}</div>
<div class="profile-title">{sel_prof.get('current_title')} at {sel_prof.get('current_company')}</div>
<div style="color: #b0a3c4; font-size: 0.9rem; margin-top: 0.3rem;">{sel_prof.get('location')}, {sel_prof.get('country')}</div>
</div>
<div style="background: linear-gradient(135deg, #ff7bf2 0%, #9e5fff 100%); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 800; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
#{sel_rank}
</div>
</div>
<div style="margin-top: 1.5rem;">
<span class="badge badge-core">{sel_prof.get('years_of_experience')} Years Experience</span>
<span class="badge badge-location">{"Open to Relocate" if sel_sigs.get('willing_to_relocate') else "Stationary"}</span>
<span class="badge badge-core">Notice: {sel_sigs.get('notice_period_days')} days</span>
</div>
<div style="margin-top: 1.5rem;">
<div class="score-breakdown-row">
<span class="score-label">Years of Experience Score</span>
<span class="score-val">{sel_match.get('yoe_score')} / 10.0</span>
</div>
<div class="score-breakdown-row">
<span class="score-label">Location Fit Score</span>
<span class="score-val">{sel_match.get('loc_score')} / 10.0</span>
</div>
<div class="score-breakdown-row">
<span class="score-label">Title/Headline Match</span>
<span class="score-val">{sel_match.get('title_score')} / 10.0</span>
</div>
<div class="score-breakdown-row">
<span class="score-label">NLP Skill Overlap</span>
<span class="score-val">{sel_match.get('nlp_skill_score')} / 20.0</span>
</div>
<div class="score-breakdown-row">
<span class="score-label">Semantic Cosine Similarity</span>
<span class="score-val">{sel_match.get('sem_score')} / 40.0</span>
</div>
<div class="score-breakdown-row">
<span class="score-label">Behavioral Availability Score</span>
<span class="score-val">{sel_match.get('behavioral_score')} / 10.0</span>
</div>
<div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 1.2rem; margin-top: 1rem; color: #ff7bf2;">
<span>Composite Discovery Score</span>
<span>{sel_score} pts</span>
</div>
</div>
<div class="reasoning-box">
" {sel_reason} "
</div>
</div>""", unsafe_allow_html=True)
    
    # Detailed career timeline tabs
    tab_exp, tab_edu, tab_skills = st.tabs(["💼 Career Timeline", "🎓 Education", "🛠️ Complete Skills"])
    
    with tab_exp:
        for i, job in enumerate(sel_cand.get("career_history", [])):
            st.markdown(f"**{job.get('title')}** at *{job.get('company')}*")
            st.caption(f"{job.get('start_date')} to {job.get('end_date') if job.get('end_date') else 'Present'} ({job.get('duration_months')} months)")
            st.write(job.get('description'))
            st.markdown("---")
            
    with tab_edu:
        for i, edu in enumerate(sel_cand.get("education", [])):
            st.markdown(f"**{edu.get('degree')} in {edu.get('field_of_study')}**")
            st.write(f"{edu.get('institution')} ({edu.get('start_year')} - {edu.get('end_year')})")
            st.write(f"Grade: {edu.get('grade')} | Tier: {edu.get('tier')}")
            st.markdown("---")
            
    with tab_skills:
        for i, skill in enumerate(sel_cand.get("skills", [])):
            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
            with col_s1:
                st.write(f"🔹 **{skill.get('name')}**")
            with col_s2:
                st.write(f"Level: {skill.get('proficiency')}")
            with col_s3:
                st.write(f"Endorsed: {skill.get('endorsements')}")

print(f"=== SCRIPT RERUN FINISHED in {time.time() - t_total_start:.4f}s ===\n")
