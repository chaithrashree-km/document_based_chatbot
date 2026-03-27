import logging
import os
import redis
import json
import uuid
from datetime import datetime
import shutil
from app.core.Rate_Limiter import limiter
from celery.result import AsyncResult
from fastapi import APIRouter,Request
from app.services.Query_Service import Retrieve
from app.services.Session_Service import SessionManagement
from app.models.Chat_Model import Chat
from app.tasks.Async_Upload import celery,async_upload
from app.security.jwt_authentication import Authentication
from fastapi import Depends, HTTPException,UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.Config import Config
from app.services.ChatHistory_Service import ChatHistoryService
from app.services.User_Service import UserService
from app.models.User_Model import UserSignup, UserLogin
from app.services.LLM_Service import Response
from app.services.Metrics_Service import MetricsService

router = APIRouter()
session = SessionManagement()
retrieve = Retrieve()
auth = Authentication()
config = Config()
response = Response()
chat_db = ChatHistoryService()
user_service = UserService()
_metrics = MetricsService()

redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
security = HTTPBearer()
  
@router.post("/signup")
@limiter.limit("10/minute")
def signup(request: Request, user: UserSignup):

    existing_user = user_service.get_user_by_email(user.email)

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = user_service.create_user(
        user.username,
        user.email,
        user.password
    )

    token = auth.create_access_token({"sub": str(user_id)})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, user: UserLogin):

    db_user = user_service.get_user_by_email(user.email)

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found. Please sign up")

    user_id,email,password_hash = db_user

    if not user_service.verify_password(user.password,password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token({ "sub": str(user_id) })

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@router.post("/upload")
@limiter.limit("5/minute")
def ingest(request: Request, file: UploadFile = File(...), user=Depends(auth.verify_token)):
    user_id = user["sub"]

    logging.info(f"Uploading/Ingesting {file.filename}")

    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if file.content_type not in config.ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    contents = file.file.read()
    if len(contents) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 30MB limit")
    file.file.seek(0) 

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_location = os.path.join(config.UPLOAD_DIR, unique_name)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = async_upload.delay(file_location)
    return {
        "user_id": user_id,
        "task_id": task.id,
        "message": "File uploaded and processing started"
    }

@router.get("/upload/status/{task_id}")
@limiter.limit("30/minute")
def get_task_status(request: Request,task_id: str, user=Depends(auth.verify_token)):
    logging.info(f"Checking status of task with if {task_id}")
    result = AsyncResult(task_id, app=celery)
    return {
        "task_id": task_id,
        "status": result.status,
    }

@router.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, chat_request: Chat, user=Depends(auth.verify_token)):
    user_id = user["sub"]
    query_recieved_time = datetime.now()
    logging.info(f"Query Recieved: {chat_request.question}")
    session_id,start, end = session.get_or_create_session(user_id)
   
    logging.info(f"Checking intent of the query")
    intent =  response.detect_intent(chat_request.question)
    logging.info(f"Intent checking completed")
    answer = await retrieve.query_docs(chat_request.question)

    logging.info("Response recieved at controller")
    logging.info("Saving to DB.")
    chat_db.store_chat(user_id=user_id, session_id=session_id,session_start = start, session_end = end, question=chat_request.question, response=answer)
    response_sending_time = datetime.now()
    time_taken = response_sending_time - query_recieved_time
    logging.info(f"Time taken for generating the response: {time_taken}")
    logging.info(f"Sending Response...")
    return {
        "session_id":session_id,
        "intent": intent,
        "Response": answer
    }

@router.post("/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    auth.revoke_token(token)

    return {
        "message": "Successfully logged out"
    }

@router.post("/new_chat")
def new_session(user=Depends(auth.verify_token)):
    user_id = user["sub"]
    session_id, start, end = session.create_new_session(user_id)

    return {
        "session_id": session_id,
        "session_start": start,
        "session_end": end
    }

@router.get("/cache_list")
def list_Cache(user=Depends(auth.verify_token)):
   keys = redis_client.keys("semantic_cache:*")
   cache_data = []
   for key in keys:
        value = redis_client.get(key)
        data = json.loads(value)
        cache_data.append({
            "key": key,
            "response": data["response"],
            # "embedding_length": len(data["embedding"])
        })
   return cache_data

@router.delete("/clear_cache")
def clear_semantic_cache(user=Depends(auth.verify_token)):
    keys = redis_client.keys("semantic_cache:*")
    if keys:
        redis_client.delete(*keys)
        logging.info("Semantic cache cleared") 
    return "cache cleared successfully!"       

@router.get("/count_chats")
def count_chats_by_user(user=Depends(auth.verify_token)):
     user_id = user["sub"]
     count=chat_db.count_user_chats(user_id)
     logging.info(f"No. of chats by user {user_id} is {count}")
     return count

@router.delete("/delete_chats")
def delete_chats_for_user(user=Depends(auth.verify_token)):
     user_id = user["sub"]
     chat_db.delete_chats_by_user(user_id)
     logging.info(f"chats of user with user-id {user_id} has been deleted")
     return "deletion completed successfully!"

@router.get("/get_chats_by_user")
def get_all_chats_by_user(user=Depends(auth.verify_token)):
     user_id = user["sub"]
     chats=chat_db.get_chats_by_user(user_id)
     logging.info(f"No. of chats by user with {user_id} is: {len(chats)}")
     return chats

@router.delete("/delete_session")
def delete_session_by_sessionId(session_id, user=Depends(auth.verify_token)):
     user_id = user["sub"]
     if not chat_db.session_belongs_to_user(session_id, user_id):
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")
     chat_db.delete_session(session_id)
     logging.info(f"session with session-id {session_id} deleted")
     return "session deleted successfully!"

@router.get("/metrics/summary")
def metrics_summary(last_n: int = 100, user=Depends(auth.verify_token)):
    return _metrics.summary(last_n=last_n)
 
 
@router.get("/metrics/recent")
def metrics_recent(n: int = 20, user=Depends(auth.verify_token)):
    return _metrics.recent(n=n)