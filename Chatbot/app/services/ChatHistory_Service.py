from app.db.Postgres_Database import Database

class ChatHistoryService:

    def store_chat(self, user_id, session_id, session_start, session_end, question, response, intent):

        query = """
        INSERT INTO chat_history
        (user_id, session_id, session_start, session_end, question, response, intent)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        values = (user_id, session_id, session_start, session_end, question, response,intent)

        # A single Database instance can't be reused after return_to_pool() is called, so we create a fresh one per method call instead.
        db = Database() 
        try:
            db.cursor.execute(query, values)
            db.conn.commit()
        finally:
            db.return_to_pool()

    def get_chats_by_user(self, user_id):
        query = """
                SELECT * FROM chat_history
                WHERE user_id = %s
                ORDER BY session_start DESC
                """
        db = Database()
        try:
            db.cursor.execute(query, (user_id,))
            return db.cursor.fetchall()
        finally:
            db.return_to_pool()       
    
    def get_chats_by_session(self, session_id):
        query = """
         SELECT question, response
         FROM chat_history
         WHERE session_id = %s
         ORDER BY session_start
         """
        db = Database()
        try:
            db.cursor.execute(query, (session_id,))
            return db.cursor.fetchall()
        finally:
            db.return_to_pool()
    
    def delete_chats_by_user(self, user_id):
        query = """
         DELETE FROM chat_history
         WHERE user_id = %s
         """
        db = Database()
        try:
            db.cursor.execute(query, (user_id,))
            db.conn.commit()
        finally:
            db.return_to_pool()

    def delete_session(self, session_id):
        query = """
        DELETE FROM chat_history
        WHERE session_id = %s
         """
        db = Database()
        try:
            db.cursor.execute(query, (session_id,))
            db.conn.commit()
        finally:
            db.return_to_pool()

    def count_user_chats(self, user_id):
        query = """
        SELECT COUNT(*) 
        FROM chat_history
        WHERE user_id = %s
        """
        db = Database()
        try:
            db.cursor.execute(query, (user_id,))
            return db.cursor.fetchone()[0]
        finally:
            db.return_to_pool()

    def get_session_owner(self, session_id: str) -> str | None:
        query = """ SELECT user_id FROM chat_history
                    WHERE session_id = %s
                    LIMIT 1
                    """
        db = Database()
        try:
           db.cursor.execute(query, (session_id,))
           row = db.cursor.fetchone()
           return str(row[0]) if row else None
        finally:
           db.return_to_pool()

    def get_chats_by_session_id(self, session_id: str) -> str | None:
        query = """SELECT question, response FROM chat_history
                   WHERE session_id = %s
                """
        db = Database()
        try:
            db.cursor.execute(query, (session_id,))
            return db.cursor.fetchall()
        finally:
            db.return_to_pool()

    def get_session_meta(self, session_id: str):
        query = """
            SELECT session_start, session_end
            FROM chat_history
            WHERE session_id = %s
            LIMIT 1
        """
        db = Database()
        try:
            db.cursor.execute(query, (session_id,))
            return db.cursor.fetchone()
        finally:
            db.return_to_pool()
 
    def delete_message(self, session_id: str, question: str):
        query = """
            DELETE FROM chat_history
            WHERE ctid = (
                SELECT ctid FROM chat_history
                WHERE session_id = %s AND question = %s
                LIMIT 1
            )
        """
        db = Database()
        try:
            db.cursor.execute(query, (session_id, question))
            db.conn.commit()
        finally:
            db.return_to_pool()

    def get_sessions_by_user(self, user_id: str):
        query = """SELECT DISTINCT ON (session_id) 
                   session_id, intent, question
                   FROM chat_history
                   WHERE user_id = %s
                   ORDER BY session_id, session_start ASC
                """
        db = Database()
        try:
           db.cursor.execute(query, (user_id,))
           rows = db.cursor.fetchall()
           return [
              {"session_id": row[0], "intent": row[1], "question": row[2]}
              for row in rows
            ]
        finally:
          db.return_to_pool()       

