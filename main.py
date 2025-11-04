import os
from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
#from langchain_classic.chains import create_retrieval_chain
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == "__main__":
    print("Hello from ragapp!")
    pdf_path = "reAct.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_reaAct")

    new_vectorstore = FAISS.load_local("faiss_index_reaAct", embeddings, allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(model="gpt-4o-mini", temperature=0), retrieval_qa_chat_prompt)

    rag_chain = create_retriever_tool(new_vectorstore.as_retriever(), combine_docs_chain)

    res = rag_chain.invoke({"input": "What is the main idea of the document?"})
    print(res)