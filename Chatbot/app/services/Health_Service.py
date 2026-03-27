import redis
from fastapi import APIRouter
from app.Config import Config
from app.db.Postgres_Database import get_pool
from app.db.Vector_Database import VectorDatabase

router = APIRouter(prefix="/health", tags=["Health"])

class Health_Service:
    config = Config()

    def check_redis(self):
        try:
          r = redis.from_url(self.config.REDIS_URL)
          r.ping()
          r.close()
          return {"status": "up"}
        except Exception as e:
          return {"status": "down", "error": str(e)}

    def check_postgres(self):
       pool = None
       conn = None
       try:
         pool = get_pool()
         conn = pool.getconn()
         cursor = conn.cursor()
         cursor.execute("SELECT 1")
         cursor.close()
         pool.putconn(conn)
         conn = None 
         return {"status": "up"}
       except Exception as e:
          return {"status": "down", "error": str(e)}
       finally:
          if pool and conn:
            pool.putconn(conn)

    def check_qdrant(self):
        try:
          vector_db = VectorDatabase()
          collections = vector_db.client.get_collections()
          return {
            "status": "up",
            "collections": len(collections.collections)
          }
        except Exception as e:
          return {"status": "down", "error": str(e)}