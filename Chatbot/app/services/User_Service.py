from passlib.context import CryptContext
from app.db.Postgres_Database import Database

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:

    def hash_password(self, password):
        return pwd_context.hash(password)

    def verify_password(self, plain, hashed):
        return pwd_context.verify(plain, hashed)

    def create_user(self, username, email, password):

        hashed = self.hash_password(password)

        query = """
        INSERT INTO users (username,email,password_hash)
        VALUES (%s,%s,%s)
        RETURNING id
        """

        db = Database()
        try:
            db.cursor.execute(query, (username, email, hashed))
            user_id = db.cursor.fetchone()[0]
            return user_id
        finally:
            db.return_to_pool()

    def get_user_by_email(self, email):
        query = "SELECT id, email, password_hash FROM users WHERE email=%s"
        db = Database()
        try:
            db.cursor.execute(query, (email,))
            return db.cursor.fetchone()
        finally:
            db.return_to_pool()