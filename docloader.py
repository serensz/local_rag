from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import hashlib

def load_docs(doc_path):
    loader = PyMuPDFLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    for doc in splits:
        uid = f"{doc_path}_{doc.metadata.get('page','')}"
        doc.metadata["doc_id"] = hashlib.md5(uid.encode()).hexdigest()
    return splits