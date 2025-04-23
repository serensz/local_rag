from store import store_all_pdfs_in_docs
from rag_chain import build_chain

def main():
    vectorstore = store_all_pdfs_in_docs()
    chain = build_chain()

    while True:
        query = input("Enter the question: ")
        response = chain.invoke({"question": query})
        print(response)

if __name__ == '__main__':
    main()
