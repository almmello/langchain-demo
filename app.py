import openai
import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.document_loaders import DirectoryLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

from langchain.document_loaders import TextLoader

from langchain.agents import initialize_agent

# Authenticate with OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Defining the llm
llm = OpenAI(temperature=0)
             #, model_name="gpt-3.5-turbo")
print("LLM Model Debug:", llm)

# Load text documents and create vector stores
planet_loader = TextLoader("data/planet/planet.txt")
planet_documents = planet_loader.load()
planet_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
planet_texts = planet_text_splitter.split_documents(planet_documents)
planet_embeddings = OpenAIEmbeddings()
planet_vectorstore = FAISS.from_documents(planet_texts, planet_embeddings)

astronauts_loader = TextLoader("data/astronauts/astronauts.txt")
astronauts_documents = astronauts_loader.load()
astronauts_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
astronauts_texts = astronauts_text_splitter.split_documents(astronauts_documents)
astronauts_embeddings = OpenAIEmbeddings()
astronauts_vectorstore = FAISS.from_documents(astronauts_texts, astronauts_embeddings)

spaceships_loader = TextLoader("data/spaceships/spaceships.txt")
spaceships_documents = spaceships_loader.load()
spaceships_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
spaceships_texts = spaceships_text_splitter.split_documents(spaceships_documents)
spaceships_embeddings = OpenAIEmbeddings()
spaceships_vectorstore = FAISS.from_documents(spaceships_texts, spaceships_embeddings)


# Define the tools to be used by the agent
tools = [
    Tool(
        name="Planet Vector Store",
        func=planet_vectorstore.similarity_search,
        description="Useful when you need to answer questions about the planet.",
    ),
    Tool(
        name="Astronauts Vector Store",
        func=astronauts_vectorstore.similarity_search,
        description="Useful when you need to answer questions about astronauts.",
    ),
    Tool(
        name="Spaceships Vector Store",
        func=spaceships_vectorstore.similarity_search,
        description="Useful when you need to answer questions about spaceships.",
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Test query
prompt = "Write a 10 days detailed travel jornal for those astronauts getting in the planet. Talk about their experiences and about the spaceships every day. Each day must have its own page with hourly description of the mission"
response = agent.run(prompt)
print(response)
