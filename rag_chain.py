from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import Ollama
from store import store_all_pdfs_in_docs  

def build_chain():
    vectorstore = store_all_pdfs_in_docs()
    ollama_llm = Ollama(model="mistral:latest", temperature=0.2)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def create_prompt(question_and_context):
        question = question_and_context["question"]
        context = question_and_context["context"]

        prompt = f"""
        You are a developer for Typhoon LLM model, you need to explain me how things are working depend on what user asked

        Document:
        {context}

        Question:
        {question}

        Answer:
        """
        return prompt.strip()
    
    chain = (
        RunnableLambda(lambda x: {"question": x["question"], "context": format_docs(retriever.invoke(x["question"]))}) |
        RunnableLambda(create_prompt) |
        ollama_llm
    )
    return chain
