import streamlit as st

def description_page():
    st.subheader("🔍 Hybrid RAG System Description")
    st.markdown("This application uses a combination of retrieval strategies to provide accurate and comprehensive answers based on stock data.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧮 Keyword-Based Heuristics")
        st.markdown("""
        - **Use case:** Exact questions like:
          - “What is the highest price for TCS?”
          - “Average stock price of Apple?”
        - **How it works:** Uses string matching and pandas.
        - **Advantage:** Very fast and accurate for structured queries.
        """)

        st.subheader("📈 GraphRAG (Graph-Based Retrieval)")
        st.markdown("""
        - **Use case:** Trend or comparative questions:
          - “Which stock had the largest increase?”
        - **How it works:** 
          - Builds a graph with stocks and dates.
          - Uses edges with prices to reason over time.
        - **Advantage:** Captures temporal relationships and trends.
        """)

    with col2:
        st.subheader("🧠 FAISS + Gemini Embeddings")
        st.markdown("""
        - **Use case:** Open-ended questions:
          - “Tell me about stock trends.”
        - **How it works:**
          - Splits CSV text into chunks.
          - Uses Gemini to generate embeddings.
          - Retrieves relevant chunks via FAISS.
        - **Advantage:** Handles vague or paraphrased questions.
        """)

        st.subheader("🔤 BM25 Retriever")
        st.markdown("""
        - **Use case:** Keyword-heavy questions where exact words matter.
        - **How it works:**
          - BM25 ranks text chunks based on keyword frequency.
        - **Advantage:** Strong keyword matching fallback when vector search misses.
        """)

    st.markdown("---")
    st.markdown("""
    ### 🤖 Answer Generation with Gemini
    - Retrieved context (from BM25 or FAISS) is passed to **Gemini** using a custom prompt.
    - Ensures the answer is well-grounded and detailed.
    """)

    st.markdown("""
    ### ⚙️ Retrieval Strategy Flow
    1. Keyword-Based Answer (fastest)
    2. GraphRAG (temporal reasoning)
    3. BM25 (keyword fallback)
    4. FAISS (semantic fallback)
    """)


