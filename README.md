# 🧞 CareerGenie – AI-Powered Job Recommendation System

CareerGenie is an AI-powered platform that analyzes CVs and provides personalized job recommendations based on skills and experience. By leveraging **LLM-powered embeddings**, **vector search with DuckDB**, and real-time **job API integration**, CareerGenie helps job seekers find the best career opportunities effortlessly.

---

## 🚀 Features

✅ **AI-Powered CV Analysis** – Extracts key skills, experience, and job titles from uploaded resumes.  
✅ **Smart Job Matching** – Uses **vector embeddings & cosine similarity** to recommend relevant job postings.  
✅ **Real-Time Job Fetching** – Retrieves job listings from **Jooble API** based on location and expertise.  
✅ **Interactive UI** – Built with **Streamlit** for a seamless user experience.  
✅ **Customizable Search** – Users can specify job location and number of job suggestions.  

---

## 🏗️ Tech Stack

- **Python** 🐍  
- **Streamlit** (UI)  
- **LlamaIndex** (Document processing & Vector search)  
- **Ollama (DeepSeek-r1:1.5b)** (LLM & Embedding Model)  
- **DuckDB** (Vector database for storing & querying CV data)  
- **Jooble API** (Job data retrieval)  
- **Scikit-learn** (Cosine similarity for job ranking)  
- **NumPy** (Data manipulation)  

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/CareerGenie.git
cd CareerGenie
```
### 2️⃣ Set Up a Virtual Environment (Optional)
```bash
pip install -r requirements.txt
```
### 4️⃣ Run Ollama for LLM & Embeddings
Ensure you have Ollama installed and running locally.
```bash
ollama run deepseek-r1:1.5b
```
### 5️⃣ Start the Application
```bash
streamlit run app.py
```
# 📦 Project Dependencies – CareerGenie

Below are the dependencies required to run **CareerGenie**, an AI-powered job recommendation system.

## 🛠️ Required Packages

```plaintext
streamlit
llama-index
duckdb
numpy
scikit-learn
ollama
requests
```
### 🤝 Contributing
Pull requests and suggestions are welcome! If you'd like to contribute, fork the repository and submit a PR.
