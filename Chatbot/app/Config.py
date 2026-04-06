import os
from dotenv import load_dotenv

class Config:

 load_dotenv()

 GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 QDRANT_URL = "http://localhost:6333"
 COLLECTION_NAME = "documents"
 REDIS_URL = "redis://localhost:6379/0"
 CACHE_THRESHOLD = 0.90
 CACHE_TTL = 86400
 SECRET_KEY = os.getenv("SECRET_KEY")
 ALGORITHM = "HS256"
 ACCESS_TOKEN_EXPIRE_MINUTES = 60
 DATABASE_URL = os.getenv("DATABASE_URL")
 POSTGRES_USER = "postgres"
 POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
 POSTGRES_HOST = "127.0.0.1"
 POSTGRES_PORT = 5432
 POSTGRES_DB = "chatbot_database"
 BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
 LOGOUT_TOKEN_EXPIRE = 1
 DOCUMENT_COLLECTION_NAME = "documents_collection"
 ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".docx", ".pptx", ".html", ".md"}
 MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
 MIN_CONN = 5
 MAX_CONN = 20
 ALLOWED_MIME_TYPES = {
        "application/pdf", "text/plain", "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "image/jpeg", "image/png",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html", "text/markdown"
    }
