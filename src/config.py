import os
from chromadb.config import Settings

# Define the folder for storing database
ROOT_DIRECTORY = "YOUR-ROOT-DIRECTORY"
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"
PERSIST_DIRECTORY = f"{SOURCE_DIRECTORY}/db"
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

DEVICE_TYPE = "cpu"
