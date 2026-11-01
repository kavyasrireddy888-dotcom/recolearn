
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import json
from datetime import datetime
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RecoLearn",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. DESIGN / CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F7F5FC;
}
/* FIX MAIN PAGE TEXT VISIBILITY */
.stApp [data-testid="stMain"] {
    color: #211B43 !important;
}

.stApp [data-testid="stMain"] label,
.stApp [data-testid="stMain"] p,
.stApp [data-testid="stMain"] h1,
.stApp [data-testid="stMain"] h2,
.stApp [data-testid="stMain"] h3,
.stApp [data-testid="stMain"] h4,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: #211B43 !important;
}
[data-testid="stSidebar"] {
    background-color: #17152F;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
.main-title {
    color: #211B43;
    font-size: 36px;
    font-weight: 800;
}
.subtitle {
    color: #8175B5;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #E9E3F7;
}
.stButton > button[kind="primary"] {
    background-color: #8068C9;
    border: none;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD PROJECT FILES
# ==========================================

@st.cache_resource
def load_project_data():
    df = pd.read_csv("phase4_prepared_dataset.csv")
    matrix = load_npz("tfidf_matrix.npz")

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return df, matrix, vectorizer

df, tfidf_matrix, vectorizer = load_project_data()

# ==========================================
# 4. DATABASE
# ==========================================

def get_connection():
    return sqlite3.connect("recolearn.db")

def setup_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            query TEXT,
            method TEXT,
            top_n INTEGER,
            results TEXT,
            searched_at TEXT
        )
    """)

    conn.commit()
    conn.close()

setup_database()

def save_history(username, query, method, top_n, baseline, mmr):
    conn = get_connection()

    saved_results = {
        "baseline": [[int(i), float(s)] for i, s in baseline],
        "mmr": [[int(i), float(s)] for i, s in mmr]
    }

    conn.execute("""
        INSERT INTO history
        (username, query, method, top_n, results, searched_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        username,
        query,
        method,
        top_n,
        json.dumps(saved_results),
        datetime.now().strftime("%d %b %Y, %I:%M %p")
    ))

    conn.commit()
    conn.close()

def get_user_history(username):
    conn = get_connection()

    history = pd.read_sql_query(
        "SELECT * FROM history WHERE username = ? ORDER BY id DESC",
        conn,
        params=(username,)
    )

    conn.close()
    return history

# ==========================================
# 5. BASELINE RECOMMENDATION
# ==========================================

def baseline_recommendations(query, top_n=5):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    indices = np.argsort(scores)[::-1]

    results = []

    for idx in indices:
        if scores[idx] <= 0:
            break

        results.append((idx, scores[idx]))

        if len(results) >= top_n:
            break

    return results

# ==========================================
# 6. MMR RECOMMENDATION
# ==========================================

def mmr_recommendations(query, top_n=5, lambda_param=0.7):
    query_vector = vectorizer.transform([query])

    relevance = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    candidate_indices = np.argsort(relevance)[::-1][:100]

    if len(candidate_indices) == 0:
        return []

    if relevance[candidate_indices[0]] <= 0:
        return []

    candidate_matrix = tfidf_matrix[candidate_indices]
    similarity_matrix = cosine_similarity(candidate_matrix)

    selected = []
    remaining = list(range(len(candidate_indices)))

    while remaining and len(selected) < top_n:
        mmr_scores = []

        for candidate in remaining:
            relevance_score = relevance[candidate_indices[candidate]]

            if selected:
                redundancy = max(
                    similarity_matrix[candidate][selected]
                )
            else:
                redundancy = 0

            score = (
                lambda_param * relevance_score
                - (1 - lambda_param) * redundancy
            )

            mmr_scores.append((candidate, score))

        best = max(mmr_scores, key=lambda x: x[1])[0]

        selected.append(best)
        remaining.remove(best)

    return [
        (candidate_indices[i], relevance[candidate_indices[i]])
        for i in selected
    ]

# ==========================================
# 7. EVALUATION
# ==========================================

def evaluate_recommendations(results):
    if not results:
        return 0, 0, 0

    indices = [item[0] for item in results]

    relevance = np.mean([item[1] for item in results])
    categories = df.iloc[indices]["Category"].nunique()

    if len(indices) > 1:
        selected_matrix = tfidf_matrix[indices]
        similarity = cosine_similarity(selected_matrix)

        upper = similarity[
            np.triu_indices(len(indices), k=1)
        ]

        redundancy = np.mean(upper)
    else:
        redundancy = 0

    return relevance, redundancy, categories

# ==========================================
# 8. DISPLAY RESOURCES
# ==========================================

def display_results(results, heading):
    st.subheader(heading)

    if not results:
        st.info("No matching resources found.")
        return

    for rank, (idx, score) in enumerate(results, 1):
        resource = df.iloc[idx]

        with st.container(border=True):
            st.markdown(
                f"### {rank}. {resource['Resource_Name']}"
            )

            st.caption(f"Category: {resource['Category']}")

            st.write(str(resource["Description"])[:400])

            st.caption(f"Relevance Score: {score:.4f}")

            url = resource["Resource_URL"]

            if pd.notna(url) and str(url).startswith("http"):
                st.link_button(
                    "Open Learning Resource ↗",
                    str(url)
                )

# ==========================================
# 9. LOGIN / SIGNUP
# ==========================================

if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:

    st.markdown("<h1 style='text-align:center;color:#211B43;'>📚 RecoLearn</h1>",
                unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#8175B5;'>Discover More. Learn Better.</p>",
        unsafe_allow_html=True
    )

    st.write("")
    left, center, right = st.columns([1, 1.3, 1])

    with center:
        st.subheader("Welcome to RecoLearn")
        st.write("Enter a username to access your learning space.")

        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username"
            )

            submitted = st.form_submit_button(
                "Continue",
                type="primary",
                use_container_width=True
            )

            if submitted:
                if username.strip():
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a username.")

    st.stop()

# ==========================================
# 10. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.markdown("# 📚 RecoLearn")
    st.caption("Discover More. Learn Better.")

    st.divider()

    st.write(f"Welcome, **{st.session_state.username}**!")

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "My Dashboard",
            "Find Resources",
            "Recommendation History",
            "Feedback",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()

    if st.button("Log out", use_container_width=True):
        st.session_state.username = None
        st.rerun()

# ==========================================
# 11. HOME
# ==========================================

if page == "Home":

    st.markdown(
        "<div class='main-title'>Welcome to RecoLearn 👋</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Discover learning resources that match your interests.</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "Explore relevant study materials with our "
        "Baseline and Maximum Marginal Relevance (MMR) methods."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Find Resources")
        st.write("Search for subjects, concepts and learning topics.")

        if st.button("Start Exploring", type="primary"):
            st.session_state.go_to_search = True
            st.rerun()

    with col2:
        st.markdown("### 📊 My Dashboard")
        st.write("View your learning activity and search history.")

        if st.button("View Dashboard"):
            st.session_state.go_to_dashboard = True
            st.rerun()

    if st.session_state.pop("go_to_search", False):
        page = "Find Resources"

    if st.session_state.pop("go_to_dashboard", False):
        page = "My Dashboard"

# ==========================================
# 12. DASHBOARD
# ==========================================

if page == "My Dashboard":

    st.markdown(
        "<div class='main-title'>My Dashboard</div>",
        unsafe_allow_html=True
    )

    st.write(f"Your learning activity, {st.session_state.username}.")

    history = get_user_history(st.session_state.username)

    total_searches = len(history)

    unique_topics = (
        history["query"].nunique() if total_searches else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Searches", total_searches)
    col2.metric("Topics Explored", unique_topics)
    col3.metric("Available Resources", len(df))

    st.divider()
    st.subheader("🕒 Recent Activity")

    if history.empty:
        st.info("You haven't searched for any resources yet. Start exploring!")

    else:
        for _, item in history.head(5).iterrows():
            with st.container(border=True):
                st.markdown(f"**{item['query']}**")
                st.caption(
                    f"{item['method']} • {item['searched_at']}"
                )

# ==========================================
# 13. FIND RESOURCES
# ==========================================

if page == "Find Resources":

    st.markdown(
        "<div class='main-title'>Find Study Resources 🔍</div>",
        unsafe_allow_html=True
    )

    query = st.text_input(
        "What would you like to learn?",
        placeholder="Example: Python, Machine Learning, Data Science..."
    )

    top_n = st.selectbox(
        "Number of recommendations",
        [3, 5, 10],
        index=1
    )

    method = st.radio(
        "Recommendation method",
        ["Baseline", "MMR", "Compare Both"],
        horizontal=True
    )

    if st.button("Find Recommendations", type="primary"):

        if not query.strip():
            st.warning("Please enter a study topic.")

        else:
            with st.spinner("Finding relevant resources..."):

                baseline = baseline_recommendations(query, top_n)
                mmr = mmr_recommendations(query, top_n, 0.7)

            if not baseline:
                st.warning("No matching resources found. Try another topic.")

            else:
                save_history(
                    st.session_state.username,
                    query,
                    method,
                    top_n,
                    baseline,
                    mmr
                )

                st.session_state.last_search = {
                    "query": query,
                    "method": method,
                    "baseline": baseline,
                    "mmr": mmr
                }

    if "last_search" in st.session_state:
        result = st.session_state.last_search

        st.divider()

        if result["method"] == "Baseline":
            display_results(
                result["baseline"],
                "📖 Baseline Recommendations"
            )

        elif result["method"] == "MMR":
            display_results(
                result["mmr"],
                "✨ MMR Recommendations"
            )

        else:
            tab1, tab2 = st.tabs(["Baseline", "MMR"])

            with tab1:
                display_results(
                    result["baseline"],
                    "Baseline Recommendations"
                )

            with tab2:
                display_results(
                    result["mmr"],
                    "MMR Recommendations"
                )

        baseline_metrics = evaluate_recommendations(result["baseline"])
        mmr_metrics = evaluate_recommendations(result["mmr"])

        st.divider()
        st.subheader("📊 Recommendation Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Baseline Relevance", f"{baseline_metrics[0]:.3f}")
        col2.metric("MMR Relevance", f"{mmr_metrics[0]:.3f}")
        col3.metric("MMR Category Diversity", mmr_metrics[2])

        comparison = pd.DataFrame({
            "Metric": [
                "Average Relevance",
                "Redundancy",
                "Category Diversity"
            ],
            "Baseline": baseline_metrics,
            "MMR": mmr_metrics
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# 14. RECOMMENDATION HISTORY
# ==========================================

if page == "Recommendation History":

    st.markdown(
        "<div class='main-title'>Recommendation History</div>",
        unsafe_allow_html=True
    )

    history = get_user_history(st.session_state.username)

    if history.empty:
        st.info("Your previous searches will appear here.")

    else:
        st.write(f"You have made {len(history)} searches.")

        for _, item in history.iterrows():

            with st.expander(
                f"🔍 {item['query']} — {item['searched_at']}"
            ):

                st.write(f"**Method:** {item['method']}")
                st.write(f"**Number requested:** {item['top_n']}")

                saved = json.loads(item["results"])

                if item["method"] in ["Baseline", "Compare Both"]:
                    display_results(
                        saved["baseline"],
                        "Baseline Results"
                    )

                if item["method"] in ["MMR", "Compare Both"]:
                    display_results(
                        saved["mmr"],
                        "MMR Results"
                    )

# ==========================================
# 15. FEEDBACK
# ==========================================

if page == "Feedback":

    st.markdown(
        "<div class='main-title'>Your Feedback 💜</div>",
        unsafe_allow_html=True
    )

    st.write(
        "We plan to add recommendation ratings and "
        "user feedback collection in a future update."
    )

    st.info(
        "Feedback collection and feedback analytics "
        "are planned features. They are not active yet."
    )

# ==========================================
# 16. ABOUT PROJECT
# ==========================================

if page == "About Project":

    st.markdown(
        "<div class='main-title'>About RecoLearn</div>",
        unsafe_allow_html=True
    )

    st.write(
        """
        RecoLearn is an intelligent study resource
        recommendation system designed to help learners
        discover relevant educational materials while
        reducing repetitive recommendations.
        """
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - **Python** — Core programming language
    - **TF-IDF** — Text feature extraction
    - **Cosine Similarity** — Resource relevance
    - **MMR** — Redundancy-aware recommendations
    - **Streamlit** — Website interface
    - **SQLite** — Search history storage
    """)

    st.subheader("Our Goal")

    st.write(
        "To balance recommendation relevance and category "
        "diversity, helping learners explore more useful resources."
    )

# ==========================================
# 17. FOOTER
# ==========================================

st.divider()

st.caption(
    "RecoLearn | Intelligent Study Resource Recommendation System"
)

st.caption(
    "Powered by TF-IDF, Cosine Similarity and Maximum Marginal Relevance."
)
