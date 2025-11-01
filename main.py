import os

from dotenv import load_dotenv
load_dotenv()


from langchain_core import embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain



if __name__ == "__main__":
    print("Retrieving the data from the vector database")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()


    query = "what is pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)