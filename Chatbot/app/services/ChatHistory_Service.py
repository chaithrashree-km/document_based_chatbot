from app.db.Postgres_Database import Database

class ChatHistoryService:

    def store_chat(self, user_id, session_id, session_start, session_end, question, response):

        query = """
        INSERT INTO chat_history
        (user_id, session_id, session_start, session_end, question, response)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (user_id, session_id, session_start, session_end, question, response)

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


