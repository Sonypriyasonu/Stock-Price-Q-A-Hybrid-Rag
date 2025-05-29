# app.py
import yfinance as yf
from datetime import datetime
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
from page2 import stock_data_qa_system
from page1 import stock_price_downloader
from page0 import description_page

# Set page config
st.title("📈 Stock Price Data App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["RAG System Description","Stock Price Downloader", "Stock Data QA System"])

if page == "RAG System Description":
    description_page()
    
# Stock Price Downloader Page
elif page == "Stock Price Downloader":
    st.subheader("📥 Stock Price Data Downloader")

    stock_price_downloader()

# Stock Data QA System Page
elif page == "Stock Data QA System":
    st.subheader("🔍 Stock Data QA System")

    stock_data_qa_system()
