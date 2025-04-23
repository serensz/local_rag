import os
import json
import lancedb
from langchain_community.vectorstores import LanceDB
from embedding import get_embedding_function
from docloader import load_docs

EMBED_LOG_PATH = "./embedded_files.json"

def load_embed_log():
    if os.path.exists(EMBED_LOG_PATH):
        with open(EMBED_LOG_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_embed_log(file_set):
    with open(EMBED_LOG_PATH, "w") as f:
        json.dump(list(file_set), f)

def store_all_pdfs_in_docs():
    db = lancedb.connect("./lancedb")
    
    if not db:
        raise ValueError("Failed to connect to the database.")
    
    embedding_function = get_embedding_function()

    if "docs" in db.table_names():
        table = db.open_table("docs")
        vectorstore = LanceDB(table=table, connection=db, embedding=embedding_function)
    else:
        # Create empty table if it doesn't exist
        vectorstore = LanceDB.from_documents(
            documents=[], 
            embedding=embedding_function, 
            connection=db, 
            table_name="docs"
        )

    embedded_files = load_embed_log()
    current_files = set(f for f in os.listdir("./docs") if f.endswith(".pdf"))
    new_files = current_files - embedded_files

    if new_files:
        for filename in new_files:
            print(f"Embedding new file: {filename}")
            path = os.path.join("./docs", filename)
            docs = load_docs(path)
            if docs:
                vectorstore.add_documents(docs)
                embedded_files.add(filename)
            else:
                print(f"Skipping {filename}, no documents found.")

        save_embed_log(embedded_files)
    else:
        print("No new PDFs to embed.")

    return vectorstore
