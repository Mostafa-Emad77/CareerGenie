# %%
import os
import shutil
import json
import http.client
import duckdb
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ---------------------------
# Set up LLM and Embedding Model
# ---------------------------
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=60.0)
embed_model = OllamaEmbedding(
    model_name="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.llm = llm
Settings.embed_model = embed_model

# ---------------------------
# Prepare DuckDB Database
# ---------------------------
db_file = "datacamp.duckdb"
if os.path.exists(db_file):
    os.remove(db_file)
con = duckdb.connect(db_file)
con.close()

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="CareerGenie", layout="wide")
st.sidebar.title("Navigation")
st.sidebar.markdown("Upload your CV and find relevant job opportunities!")

# ---------------------------
# Prepare Document Folder
# ---------------------------
folder_name = "document"
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.makedirs(folder_name, exist_ok=True)

# ---------------------------
# File Upload Section
# ---------------------------
st.title("🧞 CareerGenie")
st.markdown("Upload your CV and get job recommendations based on your skills and experience.")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
if uploaded_file:
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File {uploaded_file.name} uploaded successfully!")
    st.session_state.cv_uploaded = True

# ---------------------------
# Extract Information and Get Job Suggestions
# ---------------------------
def extract_cv_info():
    documents = SimpleDirectoryReader(folder_name).load_data()
    vector_store = DuckDBVectorStore(
        database_name=db_file,
        table_name="cv",
        persist_dir="./",
        embed_dim=1536
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("Extract technical skills, experience, and job title from the CV. Answer must start with 'Answer:'")
    return str(response).split("Answer:")[-1].strip()

# Job API Handling
def get_job_suggestions(query, location, num_results):
    connection = http.client.HTTPSConnection("jooble.org")
    headers = {"Content-type": "application/json"}
    body = json.dumps({"keywords": query, "location": location})
    connection.request('POST', f'/api/566748cd-136e-454c-9726-4f0a4dde3b9a', body, headers)
    response = connection.getresponse()
    if response.status == 200:
        return json.loads(response.read().decode()).get("jobs", [])[:num_results]
    return []

# Compute Similarity Scores
def compute_similarity(cv_summary, job_list):
    try:
        cv_vector = np.array(embed_model.get_text_embedding(cv_summary)).reshape(1, -1)
        job_vectors = np.array([
            embed_model.get_text_embedding(f"{job.get('title', '')} {job.get('company', '')} {job.get('description', '')}")
            for job in job_list
        ])
        return cosine_similarity(cv_vector, job_vectors)[0]
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return []

# ---------------------------
# Job Suggestions Section
# ---------------------------
st.subheader("📌 Extracted Information and Job Suggestions")

if "job_location" not in st.session_state:
    st.session_state.job_location = "New York"
if "num_jobs" not in st.session_state:
    st.session_state.num_jobs = 5

location_input = st.text_input("📍 Enter job location", st.session_state.job_location)
num_jobs = st.slider("📌 Number of jobs to fetch", 1, 15, st.session_state.num_jobs)

st.session_state.job_location = location_input
st.session_state.num_jobs = num_jobs

if st.button("🚀 Get Job Suggestions"):
    if not st.session_state.get("cv_uploaded", False):
        st.warning("⚠️ Please upload a CV before getting job suggestions.")
    else:
        with st.spinner("fetching job suggestions..."):
            try:
                if "cv_summary" not in st.session_state:
                    st.session_state.cv_summary = extract_cv_info()
                    st.success("✅ CV Summary Generated Successfully!")
                    st.info(st.session_state.cv_summary)

                job_list = get_job_suggestions(st.session_state.cv_summary, st.session_state.job_location, st.session_state.num_jobs)

                if job_list:
                    similarity_scores = compute_similarity(st.session_state.cv_summary, job_list)
                    for i, job in enumerate(job_list):
                        job["similarity"] = similarity_scores[i] if i < len(similarity_scores) else 0
                    job_list.sort(key=lambda x: x["similarity"], reverse=True)

                    st.markdown(f"### **Suggested Jobs in {st.session_state.job_location} (Sorted by Relevance):**")
                    for job in job_list:
                        st.markdown(f"- **{job.get('title', 'Unknown Title')}** at {job.get('company', 'Unknown Company')} in **{job.get('location', st.session_state.job_location)}** ([Apply Here]({job.get('link', '#')}))")
                        st.markdown(f"  - 🔥 **Similarity Score:** {round(job.get('similarity', 0) * 100, 2)}%")
                else:
                    st.warning(f"No job suggestions found for '{st.session_state.job_location}'. Try modifying the CV or keywords.")
            except Exception as e:
                st.error(f"Error processing CV: {e}")
    st.session_state.cv_uploaded = False







