import os
import openai
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

# Authenticate with OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)

# Load opportunity text and create vector stores
text_loader = TextLoader("data/astronauts/astronauts.txt")
documents = text_loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(texts, embeddings, collection_name="astronauts")

# Create vector store agents
vector_store_info = VectorStoreInfo(
    name="astronauts",
    description="Information about astronauts",
    vectorstore=vector_store,
)
toolkit = VectorStoreToolkit(vectorstore_info=vector_store_info)
agent = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)

# Get the response
response = agent.run("What challenges Astronauts face?")
print(response)
