# 📈 Stock Price Data App with Hybrid RAG System

This **Streamlit-based application** allows users to download stock price data and ask questions using a **Hybrid Retrieval-Augmented Generation (RAG) system**. The app combines **keyword-based heuristics**, **graph-based reasoning**, **BM25**, and **FAISS vector search** with **Gemini LLM embeddings** to deliver accurate and contextual answers for stock data queries.

---

## Features

### 1. Stock Price Downloader
- Add or select **multiple stock ticker symbols**.
- Choose a **custom date range** for historical stock prices.
- Download the data as a **CSV file** for further analysis.

### 2. Stock Data QA System
- Upload your **CSV file** with stock prices.
- Ask questions using **Hybrid RAG**:
  - **Keyword-based heuristics**: Fast, accurate answers for structured queries (e.g., highest, lowest, average prices).
  - **GraphRAG**: Temporal and trend-based reasoning (e.g., largest increase in stock prices).
  - **BM25 Retriever**: Keyword-heavy fallback search for text chunks.
  - **FAISS + Gemini Embeddings**: Semantic vector search for open-ended or paraphrased queries.
- Provides **sample questions** and supports manual queries.
- Answers indicate which retrieval strategy was used: `(Keyword)`, `(GraphRAG)`, `(BM25)`, `(FAISS)`.

### 3. RAG System Description
- Explains the **different retrieval strategies** and their use cases.
- Illustrates the workflow:
  `keyword search → graph reasoning → BM25 → FAISS → Gemini` for final answer generation.

---

## Tech Stack
- **Frontend**: Streamlit
- **Data**: Yahoo Finance API (`yfinance`)
- **Machine Learning / LLM**: Google Gemini, LangChain, FAISS, BM25, GraphRAG
- **Data Handling**: Pandas
- **Graph Processing**: NetworkX

---

## How It Works
1. **Download Stock Data**: Users select tickers and date ranges, then download CSV files.
2. **Upload CSV**: Users upload CSV files to initialize retrievers.
3. **Hybrid RAG Retrieval**:
   - Keyword queries are resolved first.
   - Graph-based reasoning answers temporal questions.
   - BM25 and FAISS handle semantic or keyword-heavy queries.
4. **Answer Generation**: Contextual answers are generated using **Gemini LLM**.

---

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-rag-app.git
