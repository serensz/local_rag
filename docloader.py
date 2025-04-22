from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_docs(doc_path):
    loader = PyMuPDFLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=100)

    return text_splitter.split_documents(documents)