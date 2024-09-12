import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#load openAI api key
os.environ['OPENAI_API_KEY'] = ''

# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500) 

loaders = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/business/stocks/accumulate-tata-motors-target-of-rs-1075-prabhudas-lilladher-12538061.html",
        "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-1188-sharekhan-12411611.html",
        "https://www.moneycontrol.com/news/business/stocks/reduce-tata-motors-target-of-rs-901-icici-securities-12411521.html"
        ]
)
data = loaders.load() 


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)

# Create the embeddings of the chunks using openAIEmbeddings
embeddings = OpenAIEmbeddings()

# Pass the documents and embeddings inorder to create FAISS vector index
vectorindex_openai = FAISS.from_documents(docs, embeddings)

pkl = vectorindex_openai.serialize_to_bytes()

# Storing vector index create in local
file_path="vector_index_1.pkl"
with open(file_path, "wb") as f:
    pickle.dump(pkl, f)

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)

vectorIndex = FAISS.deserialize_from_bytes(embeddings=OpenAIEmbeddings(), serialized=vectorIndex,allow_dangerous_deserialization=True)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())


query = "What is the least and highest suggested value for tata motors"

langchain.debug=True

chain({"question": query}, return_only_outputs=True)

