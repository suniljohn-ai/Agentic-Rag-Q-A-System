
# 📦 Custom LLM-Based Question Answering System with RAG & Agentic AI

# ======== step1: Install all required Libraries from the requirement.txt file =========
#follow this command pip install -r requirement.txt

# ======== step2: importing the required libraries ========

import os
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChian
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import pinecone
import chromadb
import keys
from utils import document_loader
from utils import agent_tools


# ======== step3: setting up the environment with the api keys ========

os.environ["OPENAI_API-_KEY"] = keys.open_ai_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys.hface_api_key
pinecone.init(api_key = keys.pinecone_api_key ,environment = keys.pinecone_env_key)

# ======= step4: Load and preprocess the documents (Pdf/Webside loader) ========

docs = document_loader()

# ======= step5: Creating Embeddings and vector stores =======

embedding_model = OpenAIEmbeddings()  #embedding_model

#if you want you can also switch to huggingfacehub embeddings
#embedding_model = HuggingFaceEmbeddings(model_name = "the repo Id")

pinecone_index_name = "agentic-rag"

if pinecone_index_name not in pinecone.list_indexes():
    Pinecone.from_documents(docs,embedding_model, index_name = pinecone_index_name)

vector_stores = Pinecone.from_existing_index(pinecone_index_name, embedding_model) #vectorstores
#if you want alternate use chromadb
# vector_stores = Chroma.from_documents(docs,embedding_model, persist_directory = "chroma_store")


# ======= step6: Setup Conversational Chain with memory =======

memory = ConversationBufferMemory(memory_key = "chat_history",return_messeges = True)
retriever = vector_stores.as_retriever()

#creating an llm option A OpenAI

llm = ChatOpenAI(temparature = 0.8)

#alternate Option B
#llm HuggingFaceHub(repo_id = "repo_id ", model_kwargs = {"temperature" : 0.8, "max_length" : 512})

qa_chain = ConversationalRetrievalChian.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory
)


# ======= step7: Defining Agentic Behaviour (with custom tools) =======

agent = initialize_agent(
    tools = agent_tools,
    llm = llm,
    agent = "chat-conversational-react-description",
    memory = memory,
    verbose = True
)


