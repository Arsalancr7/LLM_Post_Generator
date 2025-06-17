# 🧠 LLM_Post_Generator

LLM_Post_Generator is a lightweight and extensible system that uses Large Language Models (LLMs) to analyze and enrich social media content. This tool extracts metadata from raw posts, unifies tags for downstream analysis, and provides a Streamlit-based UI for reviewing and exploring results.

---

## 🚀 Features

- ✅ **Metadata Extraction**: Uses LLMs to extract the number of lines, language, and relevant hashtags from each post.
- ✅ **Tag Normalization**: Groups similar tags into unified categories using an LLM-guided heuristic.
- ✅ **Robust Parsing**: Handles malformed JSON and long-context LLM responses with safe fallbacks and chunking.
- ✅ **Streamlit Dashboard**: Visualizes enriched post data interactively.

---

## 📁 Directory Structure

LLM_Post_Generator/
│
├── raw_posts.json # Input: raw LinkedIn/TikTok-style post content
├── data/
│ └── processed_posts.json # Output: enriched and tag-normalized post data
│
├── pre_process.py # Main preprocessing script using LangChain + LLM
├── llm_helper.py # LLM pipeline configuration
├── main.py # Streamlit UI to browse enriched posts
├── requirements.txt # Python dependencies
└── README.md # You're here!
