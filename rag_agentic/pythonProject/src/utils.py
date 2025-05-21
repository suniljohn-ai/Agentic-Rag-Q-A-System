from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool
from file import qa_chain

#function to load and preprocess the documents (Pdf/Website support)
def document_loader():
    loader = DirectoryLoader("./data", glob = "*.pdf", loader_cls = PyPDFLoader)
    web_loader = WebBaseLoader("https://www.india.gov.in/spotlight/union-budget-2025-2026")
    documents = loader.load() + web_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    return splitter.split_documents(documents)


#function to take the input query
def fetch_summary(query):
    return qa_chain.run(query)

#agent behaviour and tools

agent_tools = [
    Tool(name = "RAG_QA", func = fetch_summary, description = "Answer the Question using the domain documents"),
    Tool(name = "Simple_Search",func = lambda q:"Search the web for: "+q, description = "Dummy search tool")
]

