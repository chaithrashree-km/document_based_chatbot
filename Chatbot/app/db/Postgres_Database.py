import threading
from psycopg2 import pool
from app.Config import Config

_pool = None
_pool_lock = threading.Lock()

def get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:   
                config = Config()
                _pool = pool.ThreadedConnectionPool(
                    minconn=config.MIN_CONN,
                    maxconn=config.MAX_CONN,
                    dbname=config.POSTGRES_DB,
                    user=config.POSTGRES_USER,
                    password=config.POSTGRES_PASSWORD,
                    host=config.POSTGRES_HOST,
                    port=config.POSTGRES_PORT
                )
    return _pool


class Database:

    def __init__(self):
        self.conn = get_pool().getconn() 
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def return_to_pool(self):
        self.cursor.close()
        self.cursor = None
        get_pool().putconn(self.conn) 
        self.conn = None