import os
import openai
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
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

# Load text documents and create vector stores
astronauts_loader = TextLoader("data/astronauts/astronauts-538.txt")
astronauts_documents = astronauts_loader.load()
astronauts_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
astronauts_texts = astronauts_text_splitter.split_documents(astronauts_documents)
astronauts_embeddings = OpenAIEmbeddings()
astronauts_store = Chroma.from_documents(astronauts_texts, astronauts_embeddings, collection_name="astronauts")

# Create vector store agents
astronauts_vectorstore_info = VectorStoreInfo(
    name="astronauts",
    description="Information about astronauts",
    vectorstore=astronauts_store,
)
astronauts_toolkit = VectorStoreToolkit(vectorstore_info=astronauts_vectorstore_info)


astronauts_agent = create_vectorstore_agent(
    llm=llm,
    toolkit=astronauts_toolkit,
    verbose=True,
)

# Get the response
astronauts_agent.run("What challenges Astronauts face?")