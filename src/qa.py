import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from src.config import *

from langchain.document_loaders import (
    PyPDFLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredXMLLoader,
    JSONLoader,
    # S3DirectoryLoader,
    # S3FileLoader,
    # AzureBlobStorageContainerLoader,
    # GCSDirectoryLoader,
    # GCSFileLoader,
    # GoogleDriveLoader,
    # HuggingFaceDatasetLoader,
    # NotionDirectoryLoader,
    # NotionDBLoader,
    # SlackDirectoryLoader,
    # DirectoryLoader,
)

from src.config import (
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    DEVICE_TYPE,
)

EXTENSION_MAPPING = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".xml": UnstructuredXMLLoader,
    ".json": JSONLoader,
    ".md": UnstructuredMarkdownLoader,
}

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import tempfile


def bot(db, history, llm_name):
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model=llm_name,
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
    )
    chat_history = [message["content"] for message in history]
    response = qa.run({"question": chat_history[-1], "chat_history": chat_history[:-1]})
    return response
