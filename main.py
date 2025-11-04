import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic import hub

if __name__ == "__main__":
    print("Hello from ragapp!")

    # 1) Load & split
    pdf_path = "reAct.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # 2) Embed & index
    embeddings = OpenAIEmbeddings(  # optionally specify model="text-embedding-3-small"
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_path = "faiss_index_reaAct"
    vectorstore.save_local(index_path)

    # 3) Reload index & build RAG chain
    new_vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Chain to stuff retrieved docs into the prompt
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )

    # Retrieval chain (this is the key: use create_retrieval_chain)
    retriever = new_vectorstore.as_retriever()
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 4) Ask a question
    res = rag_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences?"})
    print(res["answer"])
