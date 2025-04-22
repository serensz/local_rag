from docloader import load_docs
from store import store_embed
from rag_chain import build_chain

def main():
    docs = load_docs("./docs/cs_psu.pdf")
    vectorstore = store_embed(docs)
    chain = build_chain(vectorstore)

    while True:
        query = input("Enter the question: ")
        response = chain.invoke({"question": query})
        print(response)

if __name__ == '__main__':
    main()
