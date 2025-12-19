import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CHAT_API_KEY = os.getenv("CHAT_API_KEY")
    CHAT_BASE_URL = os.getenv("CHAT_BASE_URL")
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
    
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
    
settings = Config()