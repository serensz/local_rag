import lancedb
from langchain_community.vectorstores import LanceDB
from embedding import get_embedding_function

def store_embed(docs):
    db = lancedb.connect("./lancedb")

    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=get_embedding_function(),
        connection=db,
        table_name="docs"
    )
    
    return vectorstore