from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import streamlit as st


MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "Life_insurance_RAG"
COLLECTION_NAME = "embeddings"
ATLAS_VECTOR_SEARCH = "vector_index_life_insurance"


def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = GoogleGenerativeAIEmbeddings(model = "model/embeddings-001")

    vector_store = MongoDBAtlasVectorSearch(collection=collection,embedding=embeddings,index_name=ATLAS_VECTOR_SEARCH)

    return vector_store


def ingest_text(text_content):
    vector_store = get_vector_store()

    # document in mongoDB => bascially a json object which gets loaded in mongoDB database
    docs = Document(text_content)
    vector_store.add_documents([docs])
    return True


def get_rag_response(query):
    vector_store = get_vector_store()
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {k:3})

    prompt_template = """ Use the following context from the user in order to provide an accurate and concise response to the query.
    If you don't know the answer, just say that you don't know, don't try to make up an answer."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(llm = llm,chain_stuff = 'stuff', retriever = retriever)

    response = qa_chain.invoke({"query" : query})

    return response