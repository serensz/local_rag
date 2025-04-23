from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(model="bge-m3:latest")