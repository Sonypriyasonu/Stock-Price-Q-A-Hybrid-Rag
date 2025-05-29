import os
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
import networkx as nx #Graphrag

# Set your GEMINI_API_KEY directly here

def stock_data_qa_system():
    
    api_key = "0000000000000000000000000000000"  # Replace with your actual Gemini API key

    # Check if the API key is set
    if not api_key:
        st.error("GEMINI_API_KEY is not set.")
        st.stop()

    def get_csv_text(csv_file):
        try:
            df = pd.read_csv(csv_file)
            text = df.to_string(index=False)
            return df, text
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, ""

    def get_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        return splitter.split_text(text)

    def create_vector_store(chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        return FAISS.from_texts(chunks, embedding=embeddings)

    def create_bm25_retriever(chunks):
        documents = [Document(page_content=chunk) for chunk in chunks]
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = 4
        return retriever

    def create_graph(df):
        G = nx.DiGraph()
        for _, row in df.iterrows():
            date = row['Date']
            for stock in df.columns[1:]:
                price = row[stock]
                G.add_node(date, type='date')
                G.add_node(stock, type='stock')
                G.add_edge(stock, date, price=price)
        return G

    def query_graph(graph, question):
        try:
            question = question.lower()
            if "largest increase" in question:
                max_increase = 0
                max_stock = ""
                for stock in set(n for n, d in graph.nodes(data=True) if d.get('type') == 'stock'):
                    dates = list(graph.neighbors(stock))
                    if len(dates) >= 2:
                        prices = [graph[stock][date]['price'] for date in sorted(dates)]
                        change = prices[-1] - prices[0]
                        if change > max_increase:
                            max_increase = change
                            max_stock = stock
                return f"{max_stock} had the largest increase of {max_increase:.2f} over the dataset."
        except:
            pass
        return None

    def get_conversational_chain():
        prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details.
        If the answer is not in the provided context just say \"answer is not available in the context\". 

        Context:
        {context}

        Question: 
        {question}

        Answer:"""
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=api_key, temperature=0.5)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def extract_stock_name(question, df):
        for col in df.columns:
            if col.lower() in question:
                return col
        return df.columns[1] if len(df.columns) > 1 else None

    def extract_date(question, df):
        import re
        try:
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', question)
            for date in dates:
                if date in df['Date'].values:
                    return date
        except:
            pass
        return None

    def keyword_based_answer(question: str, df: pd.DataFrame):
        try:
            question = question.lower()
            stock = extract_stock_name(question, df)

            if "highest" in question:
                val = df[stock].max()
                date = df[df[stock] == val]['Date'].values[0]
                return f"The highest price for {stock} was {val} on {date}."
            elif "lowest" in question:
                val = df[stock].min()
                date = df[df[stock] == val]['Date'].values[0]
                return f"The lowest price for {stock} was {val} on {date}."
            elif "average" in question:
                val = df[stock].mean()
                return f"The average price for {stock} is {val:.2f}."
            elif "price on" in question:
                date = extract_date(question, df)
                if date:
                    val = df[df['Date'] == date][stock].values[0]
                    return f"The price of {stock} on {date} was {val}."
            elif "last date" in question:
                last = df['Date'].max()
                val = df[df['Date'] == last][stock].values[0]
                return f"The price of {stock} on the last date ({last}) was {val}."
            elif "change from" in question and "to" in question:
                start = df['Date'].min()
                end = df['Date'].max()
                val1 = df[df['Date'] == start][stock].values[0]
                val2 = df[df['Date'] == end][stock].values[0]
                return f"{stock} changed from {val1} on {start} to {val2} on {end}, a change of {val2 - val1:.2f}."
        except:
            pass
        return None

    def get_answer(question, vector_store, bm25_retriever, df, graph):
        keyword_ans = keyword_based_answer(question, df)
        if keyword_ans:
            return f"(Keyword)\n{keyword_ans}"

        graph_ans = query_graph(graph, question)
        if graph_ans:
            return f"(GraphRAG)\n{graph_ans}"

        if bm25_retriever:
            bm25_docs = bm25_retriever.get_relevant_documents(question)
            if bm25_docs:
                chain = get_conversational_chain()
                res = chain({"input_documents": bm25_docs, "question": question}, return_only_outputs=True)
                if res and res["output_text"].strip().lower() != "answer is not available in the context":
                    return f"(BM25)\n{res['output_text']}"

        if vector_store:
            docs = vector_store.similarity_search(question)
            chain = get_conversational_chain()
            res = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
            return f"(FAISS)\n{res['output_text']}"

        return "No answer could be generated from any retriever."

    def process_csv_file(csv_file):
        df, raw_text = get_csv_text(csv_file)
        if df is not None:
            chunks = get_chunks(raw_text)
            vector_store = create_vector_store(chunks)
            bm25_retriever = create_bm25_retriever(chunks)
            graph = create_graph(df)
            stock_names = df.columns[1:]
            start_date = df['Date'].min()
            end_date = df['Date'].max()

            questions = [
                f"What was the highest stock price for {stock_names[0]} in the given data?",
                f"What was the lowest stock price for {stock_names[0]} in the given data?",
                f"What is the average stock price for {stock_names[0]} in the dataset?",
                f"What was the stock price of {stock_names[0]} on {start_date}?",
                f"When did {stock_names[0]} stock price first cross a specific value?",
                f"What was the stock price of {stock_names[0]} on the last date in the dataset?",
                f"How did the {stock_names[0]} stock price change from {start_date} to {end_date}?",
                f"Which stock had the largest increase in price from start to end?"
            ]

            return vector_store, bm25_retriever, stock_names, start_date, end_date, questions, df, graph
        return None, None, None, None, None, [], None, None

    # -------------------- Streamlit UI --------------------


    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        vs, bm25, stock_names, start_date, end_date, questions, df, graph = process_csv_file(uploaded_file)
        st.session_state.vector_store = vs
        st.session_state.bm25_retriever = bm25
        st.session_state.df = df
        st.session_state.graph = graph

        if vs:
            st.success("File processed and retrievers initialized.")
            st.subheader("Stock Information")
            st.write(f"Stocks: {', '.join(stock_names)}")
            st.write(f"Start Date: {start_date}")
            st.write(f"End Date: {end_date}")

            st.write("### Sample Questions (Click to auto-fill below)")
            question_selection = st.selectbox("Choose a sample question", [""] + questions)
            manual_question = st.text_input("Or ask your own question:")

            if st.button("Get Answer"):
                final_question = manual_question.strip() or question_selection
                if final_question:
                    answer = get_answer(final_question, st.session_state.vector_store,
                                        st.session_state.bm25_retriever,
                                        st.session_state.df,
                                        st.session_state.graph)
                    st.write(f"**Answer:**\n{answer}")
                else:
                    st.warning("Please ask or select a question.")
