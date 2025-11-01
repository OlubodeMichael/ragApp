from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



if __name__ == "__main__":
    loader = TextLoader("/Users/lacatel/Desktop/Ai Agent/RagApp/blogText.txt")
    documents = loader.load()
    #print(documents)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    #PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])