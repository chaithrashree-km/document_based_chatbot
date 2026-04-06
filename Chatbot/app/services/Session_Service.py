import uuid
import redis
import logging
import json
from datetime import datetime, timedelta
from app.Config import Config

class SessionManagement:
 config = Config()
 SESSION_TIMEOUT = 3600
 redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
 logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

 def get_or_create_session(self, user_id):
        session_key = f"user_session:{user_id}"
        existing = self.redis_client.get(session_key)
        
        if existing:
            session_data = json.loads(existing)
            logging.info(f"Reusing existing session {session_data['session_id']}")
            logging.info(f"Existing session start-time: {session_data['start']}")
            logging.info(f"Existing session end-time: {session_data['end']}")
            return (
                session_data["session_id"],
                datetime.fromisoformat(session_data["start"]),
                datetime.fromisoformat(session_data["end"])
            )

        session_id = str(uuid.uuid4())
        session_start = datetime.now()
        session_end = session_start + timedelta(seconds=self.SESSION_TIMEOUT)

        session_data = {
            "session_id": session_id,
            "start": session_start.isoformat(),
            "end": session_end.isoformat()
        }

        self.redis_client.setex(session_key, self.SESSION_TIMEOUT, json.dumps(session_data))
        logging.info(f"New session created {session_id}")
      
        return session_id, session_start, session_end
 
 def create_new_session(self, user_id):

        session_key = f"user_session:{user_id}"

        session_id = str(uuid.uuid4())
        session_start = datetime.now()
        session_end = session_start + timedelta(seconds=self.SESSION_TIMEOUT)

        session_data = {
            "session_id": session_id,
            "start": session_start.isoformat(),
            "end": session_end.isoformat()
        }

        self.redis_client.setex(session_key, self.SESSION_TIMEOUT, json.dumps(session_data))

        logging.info(f"Created a new session with id {session_id} for new chat.")
        logging.info(f"session_start-time:{session_start} \n session_end-time: {session_end} ")
        return session_id, session_start, session_end

 def delete_session_from_redis(self, user_id: str, session_id: str):

        session_key = f"user_session:{user_id}"
        existing = self.redis_client.get(session_key)
        if existing:
            session_data = json.loads(existing)
            if session_data.get("session_id") == session_id:
                self.redis_client.delete(session_key)
                logging.info(f"Redis session key removed for user {user_id} (session {session_id})") 