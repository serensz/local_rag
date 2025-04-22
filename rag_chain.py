from langchain.chains import RetrievalQA 
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import Ollama

def build_chain(vectorstore):
    ollama_llm = Ollama(model="llama3.2",temperature=0.7)

    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def character_prompt(question):
        character = ("You are the assistant at Prince of Songkla University, Computer Science Department")       
        return character + question
     
    chain=(
        RunnableLambda(lambda x: x["question"]) |
        retriever |
        RunnableLambda(format_docs) |
        RunnableLambda(character_prompt) |
        ollama_llm
    )
    return chain