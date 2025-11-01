import os

from dotenv import load_dotenv
load_dotenv()


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate



if __name__ == "__main__":
    print("Retrieving the data from the vector database")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    
    # Connect to the existing Pinecone index
    index_name = os.environ["INDEX_NAME"]
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    
    # Retrieve relevant documents
    query = "what is pinecone in machine learning?"
    docs = vectorstore.similarity_search(query, k=3)
    
    # Create a prompt that includes the retrieved documents
    template = """You are a helpful assistant. Use the following pieces of context to answer the question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer based on the context above:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Combine the documents into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create the chain and run it
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": query})
    print(result.content)